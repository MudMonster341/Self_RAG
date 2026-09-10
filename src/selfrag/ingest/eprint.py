"""Fetch and safely extract an arXiv e-print (LaTeX source) archive.

``https://arxiv.org/e-print/{id}`` returns one of three things: a gzipped
tar of the submission's source files, a bare gzipped single file (papers
submitted as one ``.tex`` file get no tar wrapper), or -- for old papers
where only a PDF was ever kept -- a raw PDF. All three are handled; the
third is reported as ``is_pdf_only`` rather than raising, because it is a
normal, expected outcome the caller needs to fall back on (see
``pdf_fallback.py``, owned elsewhere), not a bug in this module.

The archive is untrusted input from the internet. Two defences are applied
before anything is written to disk from inside it:

1. **Path-traversal rejection.** A tar member whose name or symlink target
   would resolve outside the destination directory (``../../etc/passwd``,
   an absolute path, an absolute or escaping symlink) is refused. This is
   checked for *every* member before *any* member is extracted, so a
   malicious archive cannot get partial extraction of its safe-looking
   members before the check reaches the unsafe one.
2. **A size cap**, checked twice: once cheaply against the compressed file
   already on disk (a compressed archive over the cap can only decompress
   to something larger), and once against the sum of member sizes recorded
   in the tar header before extracting any of them (catching a small
   compressed archive that claims to unpack to something huge).
"""

from __future__ import annotations

import gzip
import tarfile
from dataclasses import dataclass
from pathlib import Path

from selfrag.ingest.arxiv_client import ArxivClient
from selfrag.paths import raw_dir

EPRINT_BASE_URL = "https://arxiv.org/e-print"

#: LaTeX source bundles for a single paper are almost always a few MB;
#: 200 MB is generous headroom for figure-heavy submissions while still
#: refusing anything that looks like abuse of the extractor.
DEFAULT_MAX_EXTRACTED_BYTES = 200 * 1024 * 1024

_PDF_MAGIC = b"%PDF"
_GZIP_MAGIC = b"\x1f\x8b"


class EprintFetchError(Exception):
    """A malformed or unsafe e-print archive. Never raised for a plain PDF-only paper."""


@dataclass(frozen=True)
class EprintResult:
    """What came back for one document, typed so callers never have to guess.

    ``main_tex_file`` is a path relative to ``dest_dir``, found by the
    heuristic the task calls for: the extracted ``.tex`` file whose content
    contains ``\\documentclass``. ``None`` means no such file was found
    (e.g. every ``.tex`` file present is an included sub-file), which the
    caller should treat as "parsing this one needs a different strategy,"
    not as an error.
    """

    doc_id: str
    dest_dir: Path
    extracted_files: tuple[str, ...]
    main_tex_file: str | None
    is_pdf_only: bool
    total_bytes: int


def fetch_eprint(
    client: ArxivClient,
    doc_id: str,
    *,
    dest_root: Path | None = None,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    base_url: str = EPRINT_BASE_URL,
) -> EprintResult:
    """Download and extract the LaTeX source for one arXiv document.

    ``doc_id`` is used verbatim in the request URL and as the destination
    directory name, so it should already be canonicalised (see
    ``selfrag.ids.canonical_arxiv_id``) by the caller.

    Raises:
        EprintFetchError: the response is neither a recognisable archive nor
            a PDF, or extracting it would violate the path-traversal or
            size-cap guards.
    """
    dest_root = dest_root or (raw_dir() / "eprint")
    dest_dir = dest_root / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    download_path = dest_dir / "_download.bin"
    total_bytes = client.stream_to_file(f"{base_url}/{doc_id}", download_path)

    try:
        with download_path.open("rb") as f:
            header = f.read(4)

        if header.startswith(_PDF_MAGIC):
            return EprintResult(doc_id, dest_dir, (), None, True, total_bytes)

        if not header.startswith(_GZIP_MAGIC):
            raise EprintFetchError(
                f"{doc_id}: e-print response is neither gzip nor PDF (first bytes {header!r})"
            )

        compressed_size = download_path.stat().st_size
        if compressed_size > max_extracted_bytes:
            raise EprintFetchError(
                f"{doc_id}: compressed e-print is {compressed_size} bytes, over the "
                f"{max_extracted_bytes}-byte cap -- refusing to even attempt extraction"
            )

        try:
            extracted = _extract_tar_gz(download_path, dest_dir, max_extracted_bytes)
        except tarfile.ReadError:
            extracted = _extract_bare_gzip(download_path, dest_dir, doc_id, max_extracted_bytes)
    finally:
        download_path.unlink(missing_ok=True)

    main_tex = _find_main_tex(dest_dir, extracted)
    return EprintResult(doc_id, dest_dir, tuple(sorted(extracted)), main_tex, False, total_bytes)


def _is_unsafe_absolute(name: str) -> bool:
    """POSIX absolute (``/etc/passwd``) or Windows drive-letter absolute (``C:\\...``)."""
    if name.startswith("/"):
        return True
    return len(name) > 1 and name[1] == ":"


def _check_safe_member(member: tarfile.TarInfo, dest_root: Path) -> None:
    """Reject a tar member that would write or link outside ``dest_root``.

    Checked against the *resolved* destination path rather than by
    pattern-matching ``".."`` in the name, because a name can escape the
    destination without containing a literal ``".."`` segment in ways a
    naive string check would miss, and conversely a name containing ``".."``
    that still resolves safely inside the destination should not be
    rejected on sight.
    """
    name = member.name
    if _is_unsafe_absolute(name):
        raise EprintFetchError(f"unsafe tar member with absolute path: {name!r}")

    resolved = (dest_root / name).resolve()
    try:
        resolved.relative_to(dest_root)
    except ValueError:
        raise EprintFetchError(f"unsafe tar member escapes destination directory: {name!r}") from None

    if member.issym() or member.islnk():
        link_target = member.linkname
        if _is_unsafe_absolute(link_target):
            raise EprintFetchError(f"unsafe tar member with absolute link target: {link_target!r}")
        resolved_link = (dest_root / Path(name).parent / link_target).resolve()
        try:
            resolved_link.relative_to(dest_root)
        except ValueError:
            raise EprintFetchError(f"unsafe tar member link escapes destination directory: {name!r}") from None


def _extract_tar_gz(archive_path: Path, dest_dir: Path, max_extracted_bytes: int) -> list[str]:
    dest_root = dest_dir.resolve()
    extracted: list[str] = []
    with tarfile.open(archive_path, mode="r:gz") as tar:
        members = tar.getmembers()

        total_size = sum(m.size for m in members if m.isfile())
        if total_size > max_extracted_bytes:
            raise EprintFetchError(
                f"archive claims {total_size} bytes of extracted content, over the "
                f"{max_extracted_bytes}-byte cap -- refusing to extract"
            )

        # Validate every member before extracting any of them: a malicious
        # archive must not get partial extraction of "safe" members ahead
        # of the unsafe one that would have aborted the whole operation.
        for member in members:
            _check_safe_member(member, dest_root)

        for member in members:
            if not member.isfile():
                continue
            # filter="data" (PEP 706) is defense-in-depth on top of
            # _check_safe_member above: every member reaching this line has
            # already passed our own explicit, testable guard, so this
            # should never actually trigger in practice.
            tar.extract(member, path=dest_dir, filter="data")
            extracted.append(member.name)
    return extracted


def _looks_like_tex(data: bytes) -> bool:
    return b"\\documentclass" in data or b"\\begin{document}" in data


def _extract_bare_gzip(archive_path: Path, dest_dir: Path, doc_id: str, max_extracted_bytes: int) -> list[str]:
    """Single-file submissions: a bare gzip of one file, no tar wrapper."""
    with gzip.open(archive_path, "rb") as gz:
        data = gz.read(max_extracted_bytes + 1)
    if len(data) > max_extracted_bytes:
        raise EprintFetchError(
            f"{doc_id}: single-file e-print exceeds the {max_extracted_bytes}-byte cap"
        )
    safe_stem = doc_id.replace("/", "_")
    name = f"{safe_stem}.tex" if _looks_like_tex(data) else f"{safe_stem}.txt"
    (dest_dir / name).write_bytes(data)
    return [name]


def _find_main_tex(dest_dir: Path, extracted_names: list[str]) -> str | None:
    """The extracted ``.tex`` file containing ``\\documentclass``, if any."""
    for name in sorted(extracted_names):
        if not name.lower().endswith(".tex"):
            continue
        try:
            content = (dest_dir / name).read_bytes()
        except OSError:
            continue
        if b"\\documentclass" in content:
            return name
    return None

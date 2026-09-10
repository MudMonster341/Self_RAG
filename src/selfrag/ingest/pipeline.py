"""The ingest pipeline orchestrator -- Phase 1's exit criterion in one place.

Wires together every module under ``selfrag.ingest`` into one coherent,
resumable flow::

    seed manifest -> acquire (e-print) -> parse (latex, PDF fallback)
      -> freeze canonical -> assess parse quality -> chunk (via registry)
      -> dedup chunks -> persist -> record run

**Idempotency is the whole point.** Re-running this module over the same
documents must add zero new chunk ids (the Phase 1 exit criterion -- see
``tests/integration/test_ingest_idempotency.py``). That is achieved by
leaning on state that already exists rather than inventing new tracking:

- ``selfrag.ingest.manifest.Manifest`` status (``needs_acquisition``) says
  whether a document needs fetching at all.
- A document already at ``PARSED`` status has canonical text already frozen
  on disk (``selfrag.ingest.canonical``); this module never re-acquires or
  re-parses it. Quality assessment and chunking both need only that frozen
  text, so they run every time, for every live document, from that text --
  which is safe *because* chunk ids and ledger upserts are themselves
  idempotent (``selfrag.ids.chunk_uid`` hashes coordinates, and
  ``Ledger.upsert_chunks``/``upsert_documents`` are ``ON CONFLICT`` upserts).
  Recomputing the same deterministic thing twice and upserting it is
  indistinguishable, from the ledger's point of view, from computing it
  once.
- A document that is already ``ACQUIRED`` but was not yet ``PARSED`` (e.g.
  the process died between those two steps on a previous run) is resumed by
  re-locating its already-downloaded source files on disk, via the same
  deterministic ``dest_root/doc_id`` layout every ``DocumentSource`` in this
  module writes to -- never by re-fetching.

**Canonical text is genuinely immutable (CLAUDE.md invariant 4).** This
module never tries to make a "changed document" re-freeze under the same
``doc_id`` -- ``freeze_canonical`` itself refuses that
(``CanonicalMismatchError``) and this module does not work around it. A
document whose content needs to change is a new ``doc_id`` (a new arXiv
version is a different document already, since ``chunk_uid``/``doc_id`` are
version-stripped by ``selfrag.ids.canonical_arxiv_id`` upstream of this
module) or an explicit ``tombstone_document`` followed by a fresh ingest.

**No network in the core.** Every network-capable operation is reached only
through an injected ``DocumentSource`` (see below); ``run_ingest`` itself
imports nothing from ``httpx``/``selfrag.ingest.arxiv_client`` except to
type the one real implementation, ``ArxivDocumentSource``. Tests drive the
whole orchestrator through a fixture ``DocumentSource`` and never reach a
socket.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from selfrag import registry
from selfrag.ids import content_hash
from selfrag.ingest import canonical
from selfrag.ingest.arxiv_client import ArxivClient
from selfrag.ingest.dedup import ChunkDeduplicator, DedupConfig
from selfrag.ingest.eprint import DEFAULT_MAX_EXTRACTED_BYTES, fetch_eprint
from selfrag.ingest.latex import LatexParseError, ParsedDocument, parse_latex_source
from selfrag.ingest.manifest import Manifest, ManifestEntry, ManifestStatus
from selfrag.ingest.pdf_fallback import parse_pdf
from selfrag.ingest.quality import ParseQualityReport, aggregate_reports, assess_quality
from selfrag.ledger import Ledger
from selfrag.paths import raw_dir
from selfrag.schema import Chunk, Document, Namespace

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class IngestConfig(BaseModel):
    """Everything one ``selfrag ingest run`` invocation needs.

    Not itself a ``RunManifest`` field or a registry component: ingest
    *produces* the corpus a ``RunManifest.corpus_snapshot`` points at,
    rather than being one stage of a retrieval pipeline. ``doc_ids`` are
    already version-stripped arXiv base ids (the same key
    ``selfrag.schema.Document.doc_id`` and ``ManifestEntry.doc_id`` use) --
    resolving a raw id/URL to that form is the caller's job (see
    ``selfrag.ids.canonical_arxiv_id``), not this config's.
    """

    corpus_name: str = "dev"
    namespace: Namespace = Namespace.PAPERS
    doc_ids: list[str] = Field(default_factory=list)
    chunker: dict = Field(default_factory=lambda: {"name": "fixed", "config": {}})
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES


def manifest_path_for(corpus_name: str) -> Path:
    """Where the manifest for ``corpus_name`` lives, given the current data dir."""
    return raw_dir() / "manifest" / f"{corpus_name}.manifest.parquet"


def default_dest_root() -> Path:
    """Where acquired source archives are extracted -- matches ``eprint.py``'s own default."""
    return raw_dir() / "eprint"


# ---------------------------------------------------------------------------
# Acquisition abstraction -- the only door to the network in this module.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquiredDocument:
    """What one document's acquisition step produced, before parsing.

    ``latex_dir``/``main_tex_file`` are set when a usable LaTeX main file
    was found; ``pdf_path`` is set when a PDF is the only thing available.
    Both may be set (a PDF fetched as a hedge) but the common case sets
    exactly one -- the pipeline tries LaTeX first when it is present, and
    only asks the source for a PDF fallback on demand (see
    ``DocumentSource.acquire_pdf_fallback``) rather than always fetching
    both, since e-print/PDF requests share arXiv's rate limit.
    """

    source_url: str
    content_hash: str
    latex_dir: Path | None = None
    main_tex_file: str | None = None
    pdf_path: Path | None = None


class DocumentSource(Protocol):
    """Abstraction over "how do we get one document's raw content onto disk."

    The only thing standing between ``run_ingest`` and a real network call.
    Tests supply an in-memory/fixture implementation of this protocol and
    ``run_ingest`` never imports anything network-capable directly, so a
    network call is structurally unreachable from a unit test that drives
    the pipeline through a fixture source.
    """

    def acquire(self, doc_id: str, dest_root: Path) -> AcquiredDocument:
        """Fetch (or otherwise make available) one document's raw content.

        Implementations should write extracted/downloaded files under
        ``dest_root / doc_id`` -- that convention is what lets a resumed run
        re-locate an already-``ACQUIRED`` document's files without calling
        this method again (see ``_locate_acquired``).
        """
        ...

    def acquire_pdf_fallback(self, doc_id: str, dest_root: Path) -> Path:
        """Fetch a PDF for ``doc_id`` specifically, for use when LaTeX parsing failed."""
        ...


def _hash_files(paths: Sequence[Path]) -> str:
    """Deterministic content hash over a set of files, order-independent.

    Sorted by path so the hash does not depend on filesystem iteration
    order -- the same set of files always hashes the same way, regardless
    of which OS or directory-listing order produced ``paths``.
    """
    hasher = hashlib.sha256()
    for p in sorted(paths, key=lambda p: p.as_posix()):
        hasher.update(p.read_bytes())
    return hasher.hexdigest()


@dataclass
class ArxivDocumentSource:
    """The real ``DocumentSource``: arXiv e-print archives, with a PDF fallback.

    Built on ``selfrag.ingest.eprint.fetch_eprint`` (owned elsewhere) for
    the LaTeX-source path, and ``ArxivClient.stream_to_file`` directly
    against ``https://arxiv.org/pdf/{doc_id}`` for the PDF path --
    ``fetch_eprint`` itself deletes its downloaded bytes once it has
    classified them (see its module docstring), so a PDF-only submission's
    actual bytes have to be fetched again from the dedicated PDF endpoint
    rather than salvaged from the e-print response.
    """

    client: ArxivClient
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES

    def acquire(self, doc_id: str, dest_root: Path) -> AcquiredDocument:
        result = fetch_eprint(
            self.client, doc_id, dest_root=dest_root, max_extracted_bytes=self.max_extracted_bytes
        )
        source_url = f"https://arxiv.org/abs/{doc_id}"

        if result.is_pdf_only or result.main_tex_file is None:
            pdf_path = self.acquire_pdf_fallback(doc_id, dest_root)
            return AcquiredDocument(
                source_url=source_url, content_hash=_hash_files([pdf_path]), pdf_path=pdf_path
            )

        extracted_paths = [result.dest_dir / name for name in result.extracted_files]
        return AcquiredDocument(
            source_url=source_url,
            content_hash=_hash_files(extracted_paths),
            latex_dir=result.dest_dir,
            main_tex_file=result.main_tex_file,
        )

    def acquire_pdf_fallback(self, doc_id: str, dest_root: Path) -> Path:
        dest_dir = dest_root / doc_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = dest_dir / f"{doc_id.replace('/', '_')}.pdf"
        self.client.stream_to_file(f"https://arxiv.org/pdf/{doc_id}", pdf_path)
        return pdf_path


def _find_main_tex_on_disk(tex_files: list[Path]) -> Path | None:
    """Minimal, re-scan-only rediscovery of the main ``.tex`` file among ``tex_files``.

    Deliberately not a full reimplementation of ``eprint._find_main_tex``
    (private to that module, and this module must not modify it): by the
    time a resumed run gets here, the files were already extracted and
    classified once by a ``DocumentSource.acquire`` call earlier in this
    corpus's history, so this only needs to be good enough to relocate the
    same answer, not to make the original classification decision.
    """
    if len(tex_files) == 1:
        return tex_files[0]
    for p in tex_files:
        if p.name == "main.tex":
            return p
    for p in tex_files:
        try:
            if b"\\documentclass" in p.read_bytes():
                return p
        except OSError:
            continue
    return None


def _locate_acquired(doc_id: str, dest_root: Path) -> AcquiredDocument:
    """Re-derive what ``DocumentSource.acquire`` returned, from disk alone.

    Used when a document is already ``ACQUIRED`` (from an earlier, possibly
    interrupted, run) so acquisition never runs twice: every
    ``DocumentSource`` in this module writes to the same deterministic
    ``dest_root / doc_id`` layout, so the files are still exactly where
    ``acquire`` left them.

    Raises:
        FileNotFoundError: no usable source files were found at the
            expected location -- the manifest says this document was
            acquired, but its files are gone (deleted out from under the
            pipeline). That is a real inconsistency and must surface, not
            be silently treated as "nothing to parse."
    """
    dest_dir = dest_root / doc_id
    source_url = f"https://arxiv.org/abs/{doc_id}"

    tex_files = sorted(dest_dir.rglob("*.tex")) if dest_dir.is_dir() else []
    if tex_files:
        main_tex = _find_main_tex_on_disk(tex_files)
        if main_tex is not None:
            return AcquiredDocument(
                source_url=source_url,
                content_hash=_hash_files(tex_files),
                latex_dir=dest_dir,
                main_tex_file=str(main_tex.relative_to(dest_dir)),
            )

    pdf_files = sorted(dest_dir.glob("*.pdf")) if dest_dir.is_dir() else []
    if pdf_files:
        return AcquiredDocument(
            source_url=source_url, content_hash=_hash_files(pdf_files), pdf_path=pdf_files[0]
        )

    raise FileNotFoundError(
        f"doc_id={doc_id!r} is marked ACQUIRED but no source files were found under {dest_dir}"
    )


# ---------------------------------------------------------------------------
# Parsing, with the parser-selection contract the task calls out explicitly.
# ---------------------------------------------------------------------------


def _parse_document(
    doc_id: str, acquired: AcquiredDocument, source: DocumentSource, dest_root: Path
) -> ParsedDocument:
    """Parse one acquired document: LaTeX source when available, PDF otherwise.

    Parser choice is not left implicit -- it is exactly "LaTeX when a main
    ``.tex`` file was found, PDF fallback otherwise" (including "LaTeX was
    found but failed to parse"), and the resulting ``ParsedDocument.parser_id``
    is what ends up on the persisted ``Document`` row, so which path ran for
    a given document is always attributable after the fact.

    Raises:
        LatexParseError / PdfParseError: whatever the last parser actually
            attempted raised, when no avenue succeeds. The caller
            (``run_ingest``) dead-letters this per document rather than
            aborting the whole corpus.
    """
    if acquired.latex_dir is not None and acquired.main_tex_file is not None:
        try:
            return parse_latex_source(acquired.latex_dir, acquired.main_tex_file)
        except LatexParseError:
            pdf_path = acquired.pdf_path or source.acquire_pdf_fallback(doc_id, dest_root)
            return parse_pdf(pdf_path)

    if acquired.pdf_path is not None:
        return parse_pdf(acquired.pdf_path)

    raise ValueError(
        f"doc_id={doc_id!r}: acquired document has neither a LaTeX source nor a PDF to parse"
    )


def _reconstruct_parsed_document(doc_id: str) -> ParsedDocument:
    """Rebuild a ``ParsedDocument`` view of an already-frozen canonical document.

    Used for every document that is already ``PARSED`` (skip re-parsing
    entirely) and by ``compute_quality_report`` (the standalone ``ingest
    quality`` report, computed after the fact from the ledger). Citations
    are not part of ``CanonicalMeta`` and are not needed by either caller
    (quality assessment and chunking both only look at ``text``/``sections``),
    so they come back empty rather than approximated.
    """
    meta = canonical.load_canonical_meta(doc_id)
    text = canonical.load_canonical(doc_id)
    return ParsedDocument(
        text=text,
        parser_id=meta.parser_id,
        title=meta.title,
        abstract=meta.abstract,
        sections=list(meta.sections),
        citations=[],
        bibliography_char_start=meta.bibliography_char_start,
    )


# ---------------------------------------------------------------------------
# The sacred offset assertion.
# ---------------------------------------------------------------------------


class SpanIntegrityError(RuntimeError):
    """A chunk's stored text does not round-trip through its own offsets.

    Deliberately **not** caught by the per-document dead-letter loop in
    ``run_ingest``. Every document's chunk offsets are produced by the same
    chunker against the same kind of frozen canonical text, so a mismatch
    here means the offset invariant itself is broken (a chunker bug, or
    canonical text that changed out from under offsets already computed
    against it) -- not a property of the one unlucky document. Continuing
    to ingest the rest of the corpus under a broken offset invariant would
    silently manufacture more corrupted data, exactly what CLAUDE.md
    invariant 4 ("offsets are sacred") exists to prevent. It must raise
    loudly and stop the run, never be swallowed into a "failed" manifest
    row as if it were an ordinary parse problem.
    """


def _assert_chunk_span_matches(doc_id: str, chunk: Chunk) -> None:
    """Verify ``chunk.text`` is exactly ``canonical.get_span(doc_id, start, end)``.

    Raises:
        SpanIntegrityError: the chunk's stored text does not match what its
            own offsets resolve to in the frozen canonical text.
    """
    actual = canonical.get_span(doc_id, chunk.char_start, chunk.char_end)
    if actual != chunk.text:
        raise SpanIntegrityError(
            f"chunk {chunk.chunk_uid} for doc_id={doc_id!r} does not round-trip: "
            f"canonical.get_span({chunk.char_start}, {chunk.char_end}) != chunk.text -- "
            "refusing to persist a chunk whose offsets do not index the frozen canonical text."
        )


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureRecord:
    """One dead-lettered document: which stage failed, and why."""

    doc_id: str
    stage: str
    reason: str


@dataclass(frozen=True)
class IngestReport:
    """What one ``run_ingest`` call did, typed so a caller never has to guess.

    ``parse_quality`` is ``None`` rather than an empty dict when nothing was
    parsed this run (zero live documents), matching
    ``selfrag.ingest.quality.aggregate_reports``'s own refusal to average
    zero reports -- an empty-but-present dict here would look like a real
    (and misleadingly perfect) measurement.
    """

    ingest_run_id: str
    corpus_snapshot: str
    n_seen: int
    n_acquired: int
    n_parsed: int
    n_failed: int
    n_skipped: int
    n_chunks_created: int
    n_chunks_duplicate: int
    parse_quality: dict[str, float] | None
    elapsed_seconds: float
    failures: tuple[FailureRecord, ...] = ()
    dry_run: bool = False


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# The orchestrator.
# ---------------------------------------------------------------------------


def _build_document_row(
    doc_id: str, parsed: ParsedDocument, source_url: str, config: IngestConfig, ingest_run_id: str
) -> Document:
    """A ``Document`` row from a parsed document -- never re-reads canonical
    metadata from disk, since everything needed is already on ``parsed``
    (and is exactly what ``freeze_canonical`` itself would have computed)."""
    return Document(
        doc_id=doc_id,
        namespace=config.namespace,
        source_url=source_url,
        title=parsed.title,
        abstract=parsed.abstract,
        doc_text_sha256=content_hash(parsed.text),
        parser_id=parsed.parser_id,
        char_len=len(parsed.text),
        ingest_run_id=ingest_run_id,
    )


def run_ingest(
    config: IngestConfig,
    *,
    source: DocumentSource | None = None,
    ledger: Ledger | None = None,
    manifest_path: Path,
    dest_root: Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> IngestReport:
    """Run the full ingest pipeline over ``config.doc_ids``.

    Args:
        source: where document content comes from. Required unless
            ``dry_run`` is true -- a dry run never fetches anything.
        ledger: where documents/chunks/the ingest-run record land. Required
            unless ``dry_run`` is true -- a dry run never writes anything.
        manifest_path: path to this corpus's manifest Parquet file.
        dest_root: where acquired source archives are extracted. Defaults
            to ``default_dest_root()``.
        limit: process at most this many of ``config.doc_ids`` (applied
            before seeding, so a limited dry run and a limited real run
            agree on which documents they are talking about).
        dry_run: report what would happen -- how many documents would need
            acquisition vs. are already done -- without fetching, parsing,
            or writing the manifest or the ledger.

    Raises:
        ValueError: ``source``/``ledger`` are missing and ``dry_run`` is
            false.
        SpanIntegrityError: see that class's docstring -- aborts the run.
    """
    if not dry_run and (source is None or ledger is None):
        raise ValueError("run_ingest requires both `source` and `ledger` unless dry_run=True")

    start = time.monotonic()
    dest_root = dest_root or default_dest_root()

    manifest = Manifest(manifest_path)
    doc_ids = list(config.doc_ids)
    if limit is not None:
        doc_ids = doc_ids[:limit]

    for doc_id in doc_ids:
        manifest.seed_pending(doc_id, f"https://arxiv.org/abs/{doc_id}")

    if dry_run:
        to_acquire = sum(1 for d in doc_ids if manifest.needs_acquisition(d))
        elapsed = time.monotonic() - start
        return IngestReport(
            ingest_run_id="",
            corpus_snapshot=manifest.snapshot_id(),
            n_seen=len(doc_ids),
            n_acquired=0,
            n_parsed=0,
            n_failed=0,
            n_skipped=len(doc_ids) - to_acquire,
            n_chunks_created=0,
            n_chunks_duplicate=0,
            parse_quality=None,
            elapsed_seconds=elapsed,
            failures=(),
            dry_run=True,
        )

    assert source is not None and ledger is not None  # narrowed by the guard above

    ingest_run_id = f"ingest-{uuid.uuid4().hex[:16]}"
    ledger.record_ingest_run(ingest_run_id, source=config.corpus_name)

    chunker = registry.build_from_dict("chunker", config.chunker)
    dedup = ChunkDeduplicator(config.dedup)

    failures: list[FailureRecord] = []
    n_acquired = n_parsed = n_skipped = 0
    quality_reports: list[ParseQualityReport] = []
    parsed_by_doc: dict[str, ParsedDocument] = {}
    documents_to_persist: list[Document] = []

    # Sorted, not seed order: dedup's "keep the earliest occurrence" and the
    # exit criterion's "same input -> same output" both depend on a stable,
    # reproducible processing order across separate runs.
    for doc_id in sorted(doc_ids):
        entry = manifest.get(doc_id)
        assert entry is not None, f"{doc_id!r} was just seeded above"

        if entry.status == ManifestStatus.TOMBSTONED:
            n_skipped += 1
            continue

        if entry.status == ManifestStatus.PARSED:
            try:
                parsed = _reconstruct_parsed_document(doc_id)
            except canonical.CanonicalNotFoundError as exc:
                manifest.mark_failed(doc_id, f"persist: {exc}")
                failures.append(FailureRecord(doc_id, "persist", str(exc)))
                continue
            n_skipped += 1
            source_url = entry.source_url
        else:
            acquired, acquire_failure = _acquire_one(doc_id, entry, manifest, source, dest_root)
            if acquire_failure is not None:
                failures.append(acquire_failure)
                continue
            assert acquired is not None
            if entry.status != ManifestStatus.ACQUIRED:
                n_acquired += 1

            try:
                parsed = _parse_document(doc_id, acquired, source, dest_root)
            except Exception as exc:  # dead-letter boundary -- one document's parser failure must not abort the whole corpus ingest
                manifest.mark_failed(doc_id, f"parse: {exc}")
                failures.append(FailureRecord(doc_id, "parse", str(exc)))
                continue

            try:
                canonical.freeze_canonical(doc_id, parsed)
            except Exception as exc:  # dead-letter boundary -- a freeze conflict for one document must not abort the whole corpus ingest
                manifest.mark_failed(doc_id, f"freeze: {exc}")
                failures.append(FailureRecord(doc_id, "freeze", str(exc)))
                continue

            manifest.mark_parsed(doc_id, parser_used=parsed.parser_id)
            n_parsed += 1
            source_url = acquired.source_url

        quality_reports.append(assess_quality(doc_id, parsed))
        parsed_by_doc[doc_id] = parsed
        documents_to_persist.append(_build_document_row(doc_id, parsed, source_url, config, ingest_run_id))

    manifest.save()
    corpus_snapshot = manifest.snapshot_id()

    all_chunks: list[Chunk] = []
    for doc_id in sorted(parsed_by_doc):
        text = parsed_by_doc[doc_id].text
        if not text:
            manifest.mark_failed(doc_id, "chunk: canonical text is empty, nothing to chunk")
            failures.append(FailureRecord(doc_id, "chunk", "canonical text is empty, nothing to chunk"))
            continue
        try:
            chunks = chunker.chunk(doc_id, text)
        except Exception as exc:  # dead-letter boundary -- one document's chunker failure must not abort the whole corpus ingest
            manifest.mark_failed(doc_id, f"chunk: {exc}")
            failures.append(FailureRecord(doc_id, "chunk", str(exc)))
            continue

        for chunk in chunks:
            _assert_chunk_span_matches(doc_id, chunk)  # loud on purpose -- see SpanIntegrityError
            dedup.add(chunk.chunk_uid, chunk.text)
            all_chunks.append(chunk)

    manifest.save()

    dup_of_map = dedup.resolve()
    n_duplicate = sum(1 for v in dup_of_map.values() if v is not None)
    final_chunks = [
        chunk.model_copy(
            update={"dup_of": dup_of_map.get(chunk.chunk_uid), "raw_text_sha256": content_hash(chunk.text)}
        )
        for chunk in all_chunks
    ]

    ledger.upsert_documents(documents_to_persist)
    ledger.upsert_chunks(final_chunks)

    n_failed = len(failures)
    elapsed = time.monotonic() - start
    ledger.finish_ingest_run(ingest_run_id, n_documents=len(documents_to_persist), n_errors=n_failed)
    for f in failures:
        ledger.record_error(None, f.stage, f"{f.doc_id}: {f.reason}")

    return IngestReport(
        ingest_run_id=ingest_run_id,
        corpus_snapshot=corpus_snapshot,
        n_seen=len(doc_ids),
        n_acquired=n_acquired,
        n_parsed=n_parsed,
        n_failed=n_failed,
        n_skipped=n_skipped,
        n_chunks_created=len(final_chunks),
        n_chunks_duplicate=n_duplicate,
        parse_quality=aggregate_reports(quality_reports) if quality_reports else None,
        elapsed_seconds=elapsed,
        failures=tuple(failures),
        dry_run=False,
    )


def _acquire_one(
    doc_id: str,
    entry: ManifestEntry,
    manifest: Manifest,
    source: DocumentSource,
    dest_root: Path,
) -> tuple[AcquiredDocument | None, FailureRecord | None]:
    """Acquire (or re-locate) one document's raw content.

    Isolated from ``run_ingest``'s main loop only to keep that loop's
    control flow readable -- this still fully owns the
    fetch-vs-relocate decision and the manifest mutation that follows it.
    """
    if manifest.needs_acquisition(doc_id):
        try:
            acquired = source.acquire(doc_id, dest_root)
        except Exception as exc:  # dead-letter boundary -- one document's acquisition failure must not abort the whole corpus ingest
            manifest.mark_failed(doc_id, f"acquire: {exc}")
            return None, FailureRecord(doc_id, "acquire", str(exc))
        manifest.mark_acquired(
            doc_id, source_url=acquired.source_url, content_hash=acquired.content_hash, version=entry.version
        )
        return acquired, None

    # Already ACQUIRED from an earlier run: re-locate on disk, never re-fetch.
    try:
        acquired = _locate_acquired(doc_id, dest_root)
    except FileNotFoundError as exc:
        manifest.mark_failed(doc_id, f"acquire: {exc}")
        return None, FailureRecord(doc_id, "acquire", str(exc))
    return acquired, None


# ---------------------------------------------------------------------------
# Standalone parse-quality reporting -- how the LaTeX-vs-PDF call gets made.
# ---------------------------------------------------------------------------


def compute_quality_report(ledger: Ledger, *, parser_id: str | None = None) -> dict[str, float] | None:
    """Aggregate parse-quality metrics across every live document in the ledger.

    Recomputed fresh from each document's frozen canonical text on every
    call -- ``selfrag.ingest.quality`` is deterministic and has no external
    state, so there is nothing worth caching, and this can never drift from
    what is actually on disk the way a once-computed, stored report could.

    Returns:
        ``None`` if there are no matching live documents to report on.
    """
    reports: list[ParseQualityReport] = []
    for doc in ledger.live_documents():
        if parser_id is not None and doc.parser_id != parser_id:
            continue
        try:
            parsed = _reconstruct_parsed_document(doc.doc_id)
        except canonical.CanonicalNotFoundError:
            continue
        reports.append(assess_quality(doc.doc_id, parsed))

    if not reports:
        return None
    return aggregate_reports(reports)

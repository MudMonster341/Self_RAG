"""Freeze parsed documents into immutable canonical text on disk.

Every offset that exists anywhere downstream of ingest -- qrels, chunk
spans, citation-verification spans -- indexes into the text this module
writes (see ``selfrag.ingest.__init__`` and
``decisions/0001-qrels-are-document-character-spans.md``).
``freeze_canonical`` is therefore the one place that gets to write it, and
it enforces the invariant that makes everything else safe to trust: once a
``doc_id`` has been frozen, its text can never silently change underneath
an offset that already points into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from selfrag.ids import content_hash, normalize_text
from selfrag.ingest.latex import ParsedDocument, Section
from selfrag.paths import canonical_dir


class CanonicalMismatchError(ValueError):
    """``doc_id`` is already frozen with text that hashes differently.

    Canonical text is write-once (CLAUDE.md invariant 4): every stored
    offset -- qrels, ``chunk_uid``s, citation spans -- indexes into
    whatever text was frozen first for this ``doc_id``. Silently
    overwriting it with different text would leave every one of those
    offsets pointing at the wrong characters with nothing anywhere to show
    it happened. This is the one place in the system that can catch that
    before it happens, so it raises instead of overwriting.
    """


class CanonicalNotFoundError(FileNotFoundError):
    """No canonical text has been frozen yet for this ``doc_id``."""


def _safe_doc_path(doc_id: str, suffix: str) -> Path:
    """Resolve ``doc_id`` to a path under ``canonical_dir()``, refusing escape.

    ``doc_id`` may legitimately contain ``/`` -- old-style arXiv ids look
    like ``"hep-th/9901001"`` -- and ``Path``'s own ``/`` operator turns
    that into a nested directory, which is intentional and handled by
    ``mkdir(parents=True)`` at write time. What must never happen is a
    ``doc_id`` containing a ``..`` segment that resolves outside
    ``canonical_dir()`` entirely; by the time a doc_id reaches this
    function it carries no proof of where it came from, so that is checked
    explicitly rather than trusted.
    """
    root = canonical_dir().resolve()
    candidate = (root / f"{doc_id}{suffix}").resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"doc_id resolves outside the canonical data dir: {doc_id!r}")
    return candidate


def _text_path(doc_id: str) -> Path:
    return _safe_doc_path(doc_id, ".txt")


def _meta_path(doc_id: str) -> Path:
    return _safe_doc_path(doc_id, ".meta.json")


@dataclass(frozen=True)
class CanonicalMeta:
    """Sidecar record for one frozen document -- everything but the text itself.

    The text lives in its own ``.txt`` file rather than inside this JSON
    so that ``load_canonical``/``get_span`` -- the hot path for citation
    verification and the MCP ``get_span`` tool -- never have to parse JSON
    to reach it.
    """

    doc_id: str
    doc_text_sha256: str
    parser_id: str
    char_len: int
    title: str = ""
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    bibliography_char_start: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_text_sha256": self.doc_text_sha256,
            "parser_id": self.parser_id,
            "char_len": self.char_len,
            "title": self.title,
            "abstract": self.abstract,
            "sections": [
                {
                    "section_path": s.section_path,
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                }
                for s in self.sections
            ],
            "citations": self.citations,
            "bibliography_char_start": self.bibliography_char_start,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CanonicalMeta:
        return cls(
            doc_id=d["doc_id"],
            doc_text_sha256=d["doc_text_sha256"],
            parser_id=d["parser_id"],
            char_len=d["char_len"],
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
            sections=[
                Section(
                    section_path=s["section_path"],
                    char_start=s["char_start"],
                    char_end=s["char_end"],
                )
                for s in d.get("sections", [])
            ],
            citations=list(d.get("citations", [])),
            bibliography_char_start=d.get("bibliography_char_start"),
        )


def freeze_canonical(doc_id: str, parsed: ParsedDocument) -> CanonicalMeta:
    """Write ``parsed.text`` to disk as the permanent canonical text for ``doc_id``.

    Idempotent when the text already frozen for ``doc_id`` is byte-identical
    to ``parsed.text`` -- re-running ingest on a document that has not
    changed must not be an error, and the *existing* metadata is returned
    unchanged in that case (this call never updates a frozen record, even
    when only non-text fields like ``sections`` would differ; "immutable"
    means the whole record, not just the text file).

    Raises:
        CanonicalMismatchError: ``doc_id`` is already frozen with different
            text. See that class's docstring for why this must be loud.
        ValueError: ``parsed.text`` is not a fixed point of
            ``normalize_text`` -- i.e. some parser skipped or double-applied
            normalisation. This is exactly the bug class
            ``selfrag.ingest.__init__``'s contract exists to catch before
            it reaches disk and corrupts every offset computed against it.
    """
    if normalize_text(parsed.text) != parsed.text:
        raise ValueError(
            f"parsed.text for doc_id={doc_id!r} is not normalised -- "
            "normalize_text(parsed.text) != parsed.text. Every parser's "
            "final step must be normalize_text(); see "
            "selfrag.ingest.__init__ for the contract this violates."
        )

    new_hash = content_hash(parsed.text)
    text_path = _text_path(doc_id)

    if text_path.exists():
        existing_hash = content_hash(text_path.read_text(encoding="utf-8"))
        if existing_hash != new_hash:
            raise CanonicalMismatchError(
                f"doc_id={doc_id!r} is already frozen with different text "
                f"(existing sha256={existing_hash}, new sha256={new_hash}). "
                "Canonical text is immutable once written -- re-freezing "
                "different text under the same id would invalidate every "
                "offset and qrel that already points into it."
            )
        return load_canonical_meta(doc_id)

    meta = CanonicalMeta(
        doc_id=doc_id,
        doc_text_sha256=new_hash,
        parser_id=parsed.parser_id,
        char_len=len(parsed.text),
        title=parsed.title,
        abstract=parsed.abstract,
        sections=list(parsed.sections),
        citations=list(parsed.citations),
        bibliography_char_start=parsed.bibliography_char_start,
    )

    text_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path(doc_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    text_path.write_text(parsed.text, encoding="utf-8")
    meta_path.write_bytes(orjson.dumps(meta.to_dict(), option=orjson.OPT_INDENT_2))
    return meta


def load_canonical(doc_id: str) -> str:
    """The full frozen canonical text for ``doc_id``.

    Reads fresh from disk on every call rather than caching, matching
    ``selfrag.paths``'s own philosophy: a cache keyed only on ``doc_id``
    would go stale the moment ``SELFRAG_DATA_DIR`` changes underneath it
    (exactly what happens between tests), and canonical documents are small
    enough (single papers, not the whole corpus) that re-reading is cheap.
    """
    path = _text_path(doc_id)
    if not path.exists():
        raise CanonicalNotFoundError(f"no canonical text frozen for doc_id={doc_id!r} at {path}")
    return path.read_text(encoding="utf-8")


def load_canonical_meta(doc_id: str) -> CanonicalMeta:
    """Sidecar metadata for ``doc_id`` (hash, parser, sections, ...)."""
    path = _meta_path(doc_id)
    if not path.exists():
        raise CanonicalNotFoundError(f"no canonical metadata frozen for doc_id={doc_id!r} at {path}")
    return CanonicalMeta.from_dict(orjson.loads(path.read_bytes()))


def get_span(doc_id: str, char_start: int, char_end: int) -> str:
    """Exact, bounds-checked slice of canonical text.

    Citation verification and the MCP ``get_span`` tool both call this
    directly, so it deliberately does no fuzzy clamping: an out-of-range
    request means a stale offset or a corrupted qrel somewhere upstream,
    and that has to surface as an error, never get silently clipped into
    something that merely looks plausible.

    Raises:
        ValueError: ``char_start`` is negative, the span is empty or
            inverted, or ``char_end`` exceeds the document's length.
        CanonicalNotFoundError: no canonical text frozen for ``doc_id``.
    """
    if char_start < 0:
        raise ValueError(f"char_start must be >= 0, got {char_start}")
    if char_end <= char_start:
        raise ValueError(f"empty or inverted span [{char_start}, {char_end})")
    text = load_canonical(doc_id)
    if char_end > len(text):
        raise ValueError(
            f"span [{char_start}, {char_end}) exceeds canonical text length "
            f"{len(text)} for doc_id={doc_id!r}"
        )
    return text[char_start:char_end]

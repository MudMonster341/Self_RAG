"""PyMuPDF fallback parser for source-less (PDF-only) arXiv submissions.

Used only when no LaTeX source is available to parse, or ``latex.py``
raised a :class:`~selfrag.ingest.latex.LatexParseError` on the source that
does exist -- LaTeX source parsing is always preferred (see the module
docstring on ``latex.py``: it beats PDF extraction on quality and CPU cost,
and gives structure almost for free).

Follows the same normalise-once-then-offset contract as ``latex.py`` (see
``selfrag.ingest.__init__``) and returns the exact same
:class:`~selfrag.ingest.latex.ParsedDocument`/``Section`` types, so a caller
cannot distinguish a LaTeX parse from a PDF-fallback parse except via
``ParsedDocument.parser_id``.

**Deliberately not handled:**

- Header/footer noise (running heads, bare page numbers, ``arXiv:NNNN.NNNNN``
  stamps) is *not* stripped here. ``selfrag.ingest.quality`` exists
  specifically to measure that contamination rate so LaTeX-vs-PDF can be
  decided empirically; cleaning it up in this module would zero out the
  one signal that comparison depends on.
- Tables: ``pymupdf4llm`` renders them as markdown pipe tables, which are
  passed through unmodified. No attempt is made to extract cell text as
  separate retrievable spans.
- Multi-column reflow errors, OCR of scanned (non-text) PDFs, and
  mid-sentence page-break artifacts are all inherited as-is from
  ``pymupdf4llm`` -- they are exactly the kind of extraction damage
  ``quality.py`` is built to quantify, not this module's job to fix.
- Title/abstract detection is heuristic (see ``parse_pdf``'s docstring) and
  is best-effort metadata, not something later code should treat as
  guaranteed accurate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm

from selfrag.ids import normalize_text
from selfrag.ingest.latex import ParsedDocument, Section, advance_heading

PARSER_ID = "pdf_fallback"

_MARK = ""
_MAX_NUMBERED_LEVEL = 3
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LEADING_NUMBER_RE = re.compile(r"^[\d.]+\s+")


class PdfParseError(ValueError):
    """The PDF could not be converted to text at all.

    Covers a missing file, a corrupt/unreadable PDF, and an unexpected
    ``pymupdf4llm`` return shape -- anything that means there is no text to
    work with, as opposed to text that came out low quality (which is
    ``quality.py``'s job to measure, not this module's job to reject).
    """


def _marker(payload: str) -> str:
    if _MARK in payload:
        raise PdfParseError(
            "internal marker character leaked into extracted content; refusing "
            "to compute offsets against text that could collide with our own "
            "bookkeeping"
        )
    return f"{_MARK}{payload}{_MARK}"


def _strip_images(text: str) -> str:
    """Markdown image placeholders (``![](...)``) carry no retrievable text."""
    return _IMAGE_RE.sub("", text)


def _process_headings(text: str) -> str:
    """Replace markdown ``#``-headings with numbered heading markers.

    Uses the same ``advance_heading`` numbering convention as ``latex.py``
    (levels 1-3 numbered, deeper levels bare) so ``section_path`` strings
    from both parsers are shaped identically and comparable side by side.
    """
    counters = [0] * _MAX_NUMBERED_LEVEL
    # (level, display) pairs. advance_heading needs the level of each frame so
    # it can pop every frame at or below an incoming heading's own level --
    # without it, consecutive siblings nest instead of replacing each other.
    path_stack: list[tuple[int, str]] = []

    def _repl(m: re.Match[str]) -> str:
        level = len(m.group(1))
        title = " ".join(m.group(2).split())
        display, section_path = advance_heading(level, title, True, counters, path_stack)
        return f"\n\n{_marker(section_path)}{display}"

    return _HEADING_RE.sub(_repl, text)


_MARKER_RE = re.compile(_MARK + r"(.*?)" + _MARK)


def _extract_markers(text: str) -> tuple[str, list[Section]]:
    """Same technique as ``latex._extract_markers``, without a bibliography
    marker: PDF-fallback text has no equivalent structural signal for where
    a reference list starts, so ``ParsedDocument.bibliography_char_start``
    is always ``None`` for this parser."""
    sections: list[Section] = []
    out_parts: list[str] = []
    cursor = 0
    running_len = 0
    pending: list[tuple[str, int]] = []

    for m in _MARKER_RE.finditer(text):
        chunk = text[cursor : m.start()]
        out_parts.append(chunk)
        running_len += len(chunk)
        pending.append((m.group(1), running_len))
        cursor = m.end()
    out_parts.append(text[cursor:])
    cleaned = "".join(out_parts)

    for i, (section_path, start) in enumerate(pending):
        end = pending[i + 1][1] if i + 1 < len(pending) else len(cleaned)
        if end > start:
            sections.append(Section(section_path=section_path, char_start=start, char_end=end))

    return cleaned, sections


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """Extract text and section structure from a PDF via ``pymupdf4llm``.

    ``title`` is a heuristic: the first level-1 markdown heading
    ``pymupdf4llm`` emits, which is usually (not always) the paper's title
    since it is normally the largest font on page 1. ``abstract`` is
    populated when a heading's leaf title (after stripping any numbering
    this module added) is exactly ``"abstract"``, case-insensitively --
    when no such heading is found, both fields are simply empty strings
    rather than a guess.

    Raises:
        PdfParseError: ``pdf_path`` does not exist, ``pymupdf4llm`` could
            not open/convert it, or it returned something other than a
            single markdown string (this module always calls it in
            whole-document string mode; anything else means a
            ``pymupdf4llm`` API change this module has not accounted for).
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise PdfParseError(f"no such PDF file: {path}")

    try:
        markdown = pymupdf4llm.to_markdown(str(path))
    except Exception as exc:  # pymupdf4llm/PyMuPDF raise their own assorted types
        raise PdfParseError(f"pymupdf4llm failed to extract {path}: {exc}") from exc

    if not isinstance(markdown, str):
        raise PdfParseError(
            f"expected pymupdf4llm.to_markdown to return str, got {type(markdown).__name__}"
        )

    text = _strip_images(markdown)

    first_heading = _HEADING_RE.search(text)
    title = ""
    if first_heading is not None and len(first_heading.group(1)) == 1:
        title = " ".join(first_heading.group(2).split())

    text = _process_headings(text)

    normalized = normalize_text(text)
    cleaned, sections = _extract_markers(normalized)

    abstract = ""
    for sec in sections:
        leaf = sec.section_path.rsplit(" > ", 1)[-1]
        leaf_title = _LEADING_NUMBER_RE.sub("", leaf).strip()
        if leaf_title.lower() == "abstract":
            abstract = cleaned[sec.char_start : sec.char_end].strip()
            break

    return ParsedDocument(
        text=cleaned,
        parser_id=PARSER_ID,
        title=title,
        abstract=abstract,
        sections=sections,
        citations=[],
        bibliography_char_start=None,
    )

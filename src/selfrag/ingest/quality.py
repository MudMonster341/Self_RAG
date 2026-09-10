"""Deterministic, no-LLM parse-quality metrics -- ablation axis 0.

Bad extraction dominates every 1-3 point delta the rest of the ladder will
ever measure, so this has to be assessed *before* anything else, and it has
to be assessed the same way regardless of which parser produced the text --
that is what makes ``ParseQualityReport`` a fair basis for deciding
LaTeX-vs-PDF empirically instead of by assertion.

Every metric here operates on ``ParsedDocument.text`` and
``ParsedDocument.sections`` alone (plus a deterministic paragraph split),
never on chunks. Chunking is a separate ablation axis (CLAUDE.md invariant
1); if this module depended on it, comparing two chunkers would also
silently compare two different quality scores for the *same* parsed
document, which would make the two axes impossible to separate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from selfrag.ingest.latex import _BUILTIN_UNARY_MACROS, ParsedDocument

_ARXIV_STAMP_RE = re.compile(r"arxiv:\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
_BARE_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
_REFERENCE_ITEM_RE = re.compile(r"^\s*(?:\[\d+\]|\(\d+\)|\d+[.)])\s")

# The three specific residue shapes `latex.py` is supposed to have already
# removed -- not "any backslash command", which would flag every
# intentionally-preserved unknown macro and make every LaTeX document look
# broken by design (see `latex.py`'s module docstring on why an unrecognised
# command is left in place on purpose, not a defect).
_MACRO_DEFINITION_RE = re.compile(r"\\(?:new|renew|provide)command\b|\\DeclareMathOperator\b")
_PASSTHROUGH_MACRO_RE = re.compile(
    r"\\(?:" + "|".join(re.escape(name) for name in _BUILTIN_UNARY_MACROS) + r")(?![A-Za-z])"
)
_STRAY_EMPTY_BRACES_RE = re.compile(r"(?<=[\w}])\{\}")

_MIN_SPAN_LEN_FOR_SHORT = 40
_MIN_REPEAT_COUNT_FOR_RUNNING_HEAD = 3
_MAX_RUNNING_HEAD_LEN = 80


@dataclass(frozen=True)
class ParseQualityReport:
    """One document's worth of deterministic parse-quality signal.

    Meant to be recorded per-document in the ledger and then averaged
    across a corpus by the caller (see ``aggregate_reports``) to compare
    parsers -- a single report on one small document is not, by itself,
    meaningful.
    """

    doc_id: str
    parser_id: str
    n_chars: int
    n_spans: int
    header_footer_contamination_rate: float
    broken_math_span_rate: float
    unexpanded_macro_rate: float
    reference_only_span_rate: float
    mean_section_label_coverage: float
    mean_span_length_chars: float
    short_span_rate: float


def _split_into_spans(text: str) -> list[str]:
    """Paragraph-level spans: blank-line-delimited stretches of text.

    Deliberately independent of any chunker -- see the module docstring.
    A blank-line-delimited paragraph is the smallest unit that both parsers
    in this package reliably preserve as one piece of prose: LaTeX's own
    paragraph breaks survive comment-stripping/macro-expansion untouched,
    and PyMuPDF4LLM's markdown output uses blank lines the same way.
    """
    return [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]


def _is_broken_math(span: str) -> bool:
    """A span with an odd ``$`` count, unbalanced braces, or an unmatched
    ``\\(``/``\\[`` delimiter -- signals of mid-formula truncation, the
    dominant math-corruption failure mode in PDF extraction.

    Deliberately does **not** flag a lone backslash on its own: both
    parsers in this package leave escaped characters and unrecognised
    commands in place rather than mangling them (see ``latex.py``'s module
    docstring), so a generic backslash count would flag correct,
    intentional output as broken.
    """
    if span.count("$") % 2 != 0:
        return True
    if span.count("{") != span.count("}"):
        return True
    if span.count("\\(") != span.count("\\)"):
        return True
    return span.count("\\[") != span.count("\\]")


def _has_unexpanded_macro_residue(span: str) -> bool:
    """A span containing residue ``latex.py`` is supposed to have removed.

    Three narrow, specific signals -- deliberately not "contains a
    backslash command", which would flag every intentionally-preserved
    unknown macro and make every LaTeX document look broken by design
    rather than by defect:

    - a ``\\newcommand``-family *definition* that leaked past extraction
      (an in-body definition whose closing brace the parser could not
      find used to survive verbatim -- the dominant LaTeX failure mode
      this metric exists to catch);
    - a literal, still-backslashed occurrence of one of ``latex.py``'s own
      passthrough/font-wrapper builtins (``\\ensuremath``, ``\\emph``,
      ``\\tiny``, ...) -- these are commands the parser specifically
      unwraps, so a surviving one is that unwrapping failing on its own
      stated job (typically a nested-brace argument its single-level
      substitution could not span), not an unrecognised command left in
      place on purpose;
    - a stray empty ``{}`` directly after a word character -- the same
      zero-argument-macro residue ``latex.py`` cleans up post-expansion,
      still present if that cleanup did not reach it.
    """
    return bool(
        _MACRO_DEFINITION_RE.search(span)
        or _PASSTHROUGH_MACRO_RE.search(span)
        or _STRAY_EMPTY_BRACES_RE.search(span)
    )


def _is_header_footer_line(line: str, line_counts: dict[str, int]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _ARXIV_STAMP_RE.search(stripped):
        return True
    if _BARE_PAGE_NUMBER_RE.match(stripped):
        return True
    return (
        len(stripped) <= _MAX_RUNNING_HEAD_LEN
        and line_counts.get(stripped, 0) >= _MIN_REPEAT_COUNT_FOR_RUNNING_HEAD
    )


def _is_reference_only_span(span: str) -> bool:
    lines = [line for line in span.split("\n") if line.strip()]
    if not lines:
        return False
    matches = sum(1 for line in lines if _REFERENCE_ITEM_RE.match(line))
    return (matches / len(lines)) >= 0.5


def assess_quality(doc_id: str, parsed: ParsedDocument) -> ParseQualityReport:
    """Compute every quality signal for one parsed document.

    - ``header_footer_contamination_rate``: fraction of non-blank *lines*
      that look like a running head, a bare page number, or an
      ``arXiv:NNNN.NNNNN`` stamp. A line short enough to plausibly be a
      running head (<= 80 chars) and repeated 3+ times verbatim anywhere in
      the document counts, in addition to the two fixed patterns.
    - ``broken_math_span_rate`` / ``unexpanded_macro_rate`` /
      ``reference_only_span_rate`` / ``short_span_rate`` /
      ``mean_span_length_chars``: computed over paragraph-level spans (see
      ``_split_into_spans``).
    - ``unexpanded_macro_rate``: fraction of spans containing residue that
      should not have survived parsing -- a leaked ``\\newcommand``-family
      definition, a known passthrough/font wrapper still backslashed, or a
      stray empty ``{}`` (see ``_has_unexpanded_macro_residue``). Always
      0.0 for ``pdf_fallback`` output, which never contains LaTeX source.
    - ``mean_section_label_coverage``: fraction of document characters that
      fall inside *some* ``Section`` -- i.e. under a named heading rather
      than an unlabelled blob. 0.0 for a document with no sections at all.
    """
    text = parsed.text
    n_chars = len(text)

    lines = text.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    nonempty_lines = [line for line in lines if line.strip()]
    contaminated = sum(1 for line in nonempty_lines if _is_header_footer_line(line, line_counts))
    header_footer_rate = contaminated / len(nonempty_lines) if nonempty_lines else 0.0

    spans = _split_into_spans(text)
    n_spans = len(spans)
    if n_spans:
        broken_math_rate = sum(1 for s in spans if _is_broken_math(s)) / n_spans
        unexpanded_macro_rate = sum(1 for s in spans if _has_unexpanded_macro_residue(s)) / n_spans
        reference_only_rate = sum(1 for s in spans if _is_reference_only_span(s)) / n_spans
        mean_span_length = sum(len(s) for s in spans) / n_spans
        short_span_rate = sum(1 for s in spans if len(s) < _MIN_SPAN_LEN_FOR_SHORT) / n_spans
    else:
        broken_math_rate = 0.0
        unexpanded_macro_rate = 0.0
        reference_only_rate = 0.0
        mean_span_length = 0.0
        short_span_rate = 0.0

    if parsed.sections and n_chars:
        covered = sum(
            max(0, min(sec.char_end, n_chars) - max(sec.char_start, 0)) for sec in parsed.sections
        )
        coverage = min(1.0, covered / n_chars)
    else:
        coverage = 0.0

    return ParseQualityReport(
        doc_id=doc_id,
        parser_id=parsed.parser_id,
        n_chars=n_chars,
        n_spans=n_spans,
        header_footer_contamination_rate=header_footer_rate,
        broken_math_span_rate=broken_math_rate,
        unexpanded_macro_rate=unexpanded_macro_rate,
        reference_only_span_rate=reference_only_rate,
        mean_section_label_coverage=coverage,
        mean_span_length_chars=mean_span_length,
        short_span_rate=short_span_rate,
    )


_MEAN_FIELDS = (
    "header_footer_contamination_rate",
    "broken_math_span_rate",
    "unexpanded_macro_rate",
    "reference_only_span_rate",
    "mean_section_label_coverage",
    "mean_span_length_chars",
    "short_span_rate",
)


def aggregate_reports(reports: Sequence[ParseQualityReport]) -> dict[str, float]:
    """Mean of every rate/length field across a corpus of reports.

    This is the number that actually decides LaTeX-vs-PDF: any single
    document's report is noisy (a two-paragraph document can look
    "perfect" by accident), but the corpus mean is what the ladder records
    for each parser and compares. Callers wanting a per-parser comparison
    filter ``reports`` to one ``parser_id`` before calling this.

    Raises:
        ValueError: ``reports`` is empty -- there is no meaningful mean of
            zero documents, and returning e.g. all-zero would look like a
            real (and misleadingly excellent) measurement.
    """
    if not reports:
        raise ValueError("cannot aggregate an empty sequence of quality reports")
    n = len(reports)
    return {field: sum(getattr(r, field) for r in reports) / n for field in _MEAN_FIELDS}

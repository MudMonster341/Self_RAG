"""Tests for selfrag.ingest.pdf_fallback.

Two layers, deliberately: the marker/offset machinery
(``_process_headings``/``_extract_markers``) is tested directly against
plain synthetic markdown strings, because it is *this module's* logic and
must be deterministic regardless of what ``pymupdf4llm`` decides to do with
any particular PDF's font sizes. ``parse_pdf`` itself is then tested
end-to-end against small PDFs built at test time with ``pymupdf`` -- real
files, no network, no checked-in binary fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from selfrag.ids import normalize_text
from selfrag.ingest.latex import ParsedDocument
from selfrag.ingest.pdf_fallback import (
    PdfParseError,
    _extract_markers,
    _marker,
    _process_headings,
    _strip_images,
    parse_pdf,
)


def _make_pdf(tmp_path: Path, blocks: list[tuple[str, int]], name: str = "doc.pdf") -> Path:
    """Write a minimal single-page PDF, one text line per ``(text, fontsize)``."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    for text, fontsize in blocks:
        page.insert_text((72, y), text, fontsize=fontsize)
        y += max(30, fontsize * 2)
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def _assert_invariants(doc: ParsedDocument) -> None:
    """The four invariants every parser output in this package must satisfy."""
    assert normalize_text(doc.text) == doc.text

    prev_end = -1
    for sec in doc.sections:
        assert sec.char_end > sec.char_start, f"empty/inverted span: {sec}"
        assert sec.char_start >= prev_end, f"out of order or overlapping: {sec}"
        assert doc.text[sec.char_start : sec.char_end] != "", f"slices to empty text: {sec}"
        prev_end = sec.char_end

    cursor = 0
    reconstructed: list[str] = []
    for sec in doc.sections:
        reconstructed.append(doc.text[cursor : sec.char_start])
        reconstructed.append(doc.text[sec.char_start : sec.char_end])
        cursor = sec.char_end
    reconstructed.append(doc.text[cursor:])
    assert "".join(reconstructed) == doc.text


class TestStripImages:
    def test_strips_markdown_image_syntax(self):
        text = "Before ![alt text](image.png) after."
        assert _strip_images(text) == "Before  after."

    def test_leaves_ordinary_text_untouched(self):
        text = "No images here at all."
        assert _strip_images(text) == text


class TestProcessHeadings:
    def test_single_heading_becomes_a_marker(self):
        text = "# Title\n\nBody text.\n"
        out = _process_headings(text)
        cleaned, sections = _extract_markers(normalize_text(out))
        assert [s.section_path for s in sections] == ["1 Title"]
        assert "Body text." in cleaned[sections[0].char_start : sections[0].char_end]

    def test_heading_levels_nest_and_number_independently(self):
        text = "# Intro\nintro body\n## Sub\nsub body\n# Next\nnext body\n"
        out = _process_headings(text)
        cleaned, sections = _extract_markers(normalize_text(out))
        assert [s.section_path for s in sections] == ["1 Intro", "1 Intro > 1.1 Sub", "2 Next"]

    def test_heading_deeper_than_level_three_is_unnumbered(self):
        text = "# A\n#### deep heading\nbody\n"
        out = _process_headings(text)
        cleaned, sections = _extract_markers(normalize_text(out))
        assert sections[-1].section_path == "1 A > deep heading"

    def test_non_heading_text_is_untouched(self):
        text = "Just a plain paragraph, no headings at all."
        assert _process_headings(text) == text


class TestExtractMarkers:
    def test_computes_correct_offsets_and_strips_marker_characters(self):
        text = f"prefix {_marker('1 A')}A body here.\n\n{_marker('2 B')}B body."
        normalized = normalize_text(text)
        cleaned, sections = _extract_markers(normalized)

        assert "" not in cleaned
        assert [s.section_path for s in sections] == ["1 A", "2 B"]
        assert cleaned[sections[0].char_start : sections[0].char_end] == "A body here.\n\n"
        assert cleaned[sections[1].char_start : sections[1].char_end] == "B body."

    def test_no_markers_returns_text_unchanged_and_no_sections(self):
        text = "Nothing structured about this at all."
        cleaned, sections = _extract_markers(text)
        assert cleaned == text
        assert sections == []


class TestParsePdfEndToEnd:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(PdfParseError):
            parse_pdf(tmp_path / "does_not_exist.pdf")

    def test_corrupt_file_raises(self, tmp_path: Path):
        bad = tmp_path / "corrupt.pdf"
        bad.write_bytes(b"not actually a pdf")
        with pytest.raises(PdfParseError):
            parse_pdf(bad)

    def test_heading_and_body_text_extracted(self, tmp_path: Path):
        pdf_path = _make_pdf(
            tmp_path,
            [
                ("A Great Title", 24),
                ("This is the body of the paper.", 11),
            ],
        )
        doc = parse_pdf(pdf_path)
        assert doc.parser_id == "pdf_fallback"
        assert doc.title == "A Great Title"
        assert "This is the body of the paper." in doc.text
        assert len(doc.sections) >= 1
        assert doc.sections[0].section_path == "1 A Great Title"
        _assert_invariants(doc)

    def test_abstract_heading_is_detected_regardless_of_numbering(self, tmp_path: Path):
        pdf_path = _make_pdf(
            tmp_path,
            [
                ("A Great Title", 24),
                ("Abstract", 24),
                ("We present a retrieval system.", 11),
            ],
        )
        doc = parse_pdf(pdf_path)
        assert "retrieval system" in doc.abstract.lower()

    def test_no_headings_produces_empty_sections_but_valid_document(self, tmp_path: Path):
        pdf_path = _make_pdf(tmp_path, [("Just some plain body text, no headings.", 11)])
        doc = parse_pdf(pdf_path)
        assert doc.sections == []
        assert "Just some plain body text" in doc.text
        _assert_invariants(doc)

    def test_bibliography_char_start_is_always_none(self, tmp_path: Path):
        pdf_path = _make_pdf(tmp_path, [("Some Heading", 22), ("Body text.", 11)])
        doc = parse_pdf(pdf_path)
        assert doc.bibliography_char_start is None

    def test_accepts_str_path_as_well_as_path_object(self, tmp_path: Path):
        pdf_path = _make_pdf(tmp_path, [("Heading", 22), ("Body.", 11)])
        doc = parse_pdf(str(pdf_path))
        assert "Body." in doc.text

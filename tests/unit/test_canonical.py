"""Tests for selfrag.ingest.canonical -- freezing parsed documents to disk.

Every test points ``SELFRAG_DATA_DIR`` at a fresh ``tmp_path`` (via the
``data_dir`` fixture) rather than the real ``data/`` directory, matching how
``selfrag.paths`` is designed to be tested (see its module docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from selfrag.ids import content_hash
from selfrag.ingest.canonical import (
    CanonicalMeta,
    CanonicalMismatchError,
    CanonicalNotFoundError,
    freeze_canonical,
    get_span,
    load_canonical,
    load_canonical_meta,
)
from selfrag.ingest.latex import ParsedDocument, Section


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SELFRAG_DATA_DIR", str(tmp_path))
    return tmp_path


def _doc(text: str = "Hello world.\nSecond line.", **kwargs) -> ParsedDocument:
    defaults = {"text": text, "parser_id": "latex"}
    defaults.update(kwargs)
    return ParsedDocument(**defaults)


class TestFreezeCanonical:
    def test_writes_text_and_meta_to_disk(self, data_dir: Path):
        doc = _doc()
        meta = freeze_canonical("2401.01234", doc)
        assert meta.doc_text_sha256 == content_hash(doc.text)
        assert meta.char_len == len(doc.text)
        assert meta.parser_id == "latex"
        assert (data_dir / "canonical" / "2401.01234.txt").exists()
        assert (data_dir / "canonical" / "2401.01234.meta.json").exists()

    def test_rejects_non_normalised_text(self, data_dir: Path):
        doc = _doc(text="trailing whitespace   \nand a\ttab")
        with pytest.raises(ValueError):
            freeze_canonical("2401.99999", doc)

    def test_idempotent_on_identical_text(self, data_dir: Path):
        doc = _doc()
        meta1 = freeze_canonical("2401.01234", doc)
        meta2 = freeze_canonical("2401.01234", doc)
        assert meta1 == meta2

    def test_raises_on_conflicting_text(self, data_dir: Path):
        freeze_canonical("2401.01234", _doc(text="Original text."))
        with pytest.raises(CanonicalMismatchError):
            freeze_canonical("2401.01234", _doc(text="Different text entirely."))

    def test_conflicting_freeze_does_not_overwrite_existing_text(self, data_dir: Path):
        freeze_canonical("2401.01234", _doc(text="Original text."))
        with pytest.raises(CanonicalMismatchError):
            freeze_canonical("2401.01234", _doc(text="Different text entirely."))
        assert load_canonical("2401.01234") == "Original text."

    def test_old_style_doc_id_with_slash_creates_nested_directory(self, data_dir: Path):
        doc = _doc(text="Old-style id body.")
        freeze_canonical("hep-th/9901001", doc)
        assert load_canonical("hep-th/9901001") == "Old-style id body."
        assert (data_dir / "canonical" / "hep-th" / "9901001.txt").exists()

    def test_path_traversal_doc_id_is_rejected(self, data_dir: Path):
        doc = _doc()
        with pytest.raises(ValueError):
            freeze_canonical("../../evil", doc)

    def test_sections_and_citations_round_trip_through_meta(self, data_dir: Path):
        doc = _doc(
            text="Abstract text.\n\nBody of the method section.",
            title="A Paper",
            abstract="Abstract text.",
            sections=[
                Section(section_path="Abstract", char_start=0, char_end=14),
                Section(section_path="1 Method", char_start=16, char_end=44),
            ],
            citations=["foo2020", "bar2021"],
            bibliography_char_start=44,
        )
        freeze_canonical("2401.01234", doc)
        meta = load_canonical_meta("2401.01234")
        assert meta.title == "A Paper"
        assert meta.abstract == "Abstract text."
        assert [s.section_path for s in meta.sections] == ["Abstract", "1 Method"]
        assert meta.sections[1].char_start == 16
        assert meta.citations == ["foo2020", "bar2021"]
        assert meta.bibliography_char_start == 44


class TestLoadCanonical:
    def test_raises_when_not_frozen(self, data_dir: Path):
        with pytest.raises(CanonicalNotFoundError):
            load_canonical("2401.01234")

    def test_meta_raises_when_not_frozen(self, data_dir: Path):
        with pytest.raises(CanonicalNotFoundError):
            load_canonical_meta("2401.01234")

    def test_returns_exact_frozen_text(self, data_dir: Path):
        doc = _doc(text="Exact text.\nWith two lines.")
        freeze_canonical("2401.01234", doc)
        assert load_canonical("2401.01234") == "Exact text.\nWith two lines."


class TestGetSpan:
    def test_exact_slice(self, data_dir: Path):
        doc = _doc(text="0123456789")
        freeze_canonical("2401.01234", doc)
        assert get_span("2401.01234", 2, 5) == "234"

    def test_full_document_span(self, data_dir: Path):
        doc = _doc(text="full text here")
        freeze_canonical("2401.01234", doc)
        assert get_span("2401.01234", 0, len(doc.text)) == doc.text

    def test_negative_start_raises(self, data_dir: Path):
        freeze_canonical("2401.01234", _doc(text="abc"))
        with pytest.raises(ValueError):
            get_span("2401.01234", -1, 2)

    def test_inverted_span_raises(self, data_dir: Path):
        freeze_canonical("2401.01234", _doc(text="abcdef"))
        with pytest.raises(ValueError):
            get_span("2401.01234", 4, 2)

    def test_empty_span_raises(self, data_dir: Path):
        freeze_canonical("2401.01234", _doc(text="abcdef"))
        with pytest.raises(ValueError):
            get_span("2401.01234", 2, 2)

    def test_out_of_bounds_end_raises(self, data_dir: Path):
        freeze_canonical("2401.01234", _doc(text="short"))
        with pytest.raises(ValueError):
            get_span("2401.01234", 0, 999)

    def test_unfrozen_doc_id_raises_not_found(self, data_dir: Path):
        with pytest.raises(CanonicalNotFoundError):
            get_span("2401.01234", 0, 1)


class TestCanonicalMetaSerialisation:
    def test_to_dict_and_from_dict_round_trip(self):
        meta = CanonicalMeta(
            doc_id="2401.01234",
            doc_text_sha256="abc123",
            parser_id="latex",
            char_len=42,
            title="T",
            abstract="A",
            sections=[Section(section_path="1 X", char_start=0, char_end=5)],
            citations=["k1"],
            bibliography_char_start=40,
        )
        restored = CanonicalMeta.from_dict(meta.to_dict())
        assert restored == meta

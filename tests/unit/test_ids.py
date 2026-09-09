"""Tests for selfrag.ids -- the identity functions everything else keys off of.

canonical_arxiv_id gets the most scrutiny here because it is the one
function in this module whose edge cases are genuinely surprising: new-style
vs. old-style ids, URL forms, version suffixes, and inputs that must raise
rather than silently produce a wrong-but-plausible id.
"""

from __future__ import annotations

import pytest

from selfrag.ids import canonical_arxiv_id, chunk_uid, config_hash, content_hash, normalize_text


class TestCanonicalArxivId:
    def test_new_style_bare(self):
        assert canonical_arxiv_id("2401.01234") == ("2401.01234", None)

    def test_new_style_with_version(self):
        assert canonical_arxiv_id("2401.01234v3") == ("2401.01234", 3)

    def test_new_style_arxiv_prefix(self):
        assert canonical_arxiv_id("arXiv:2401.01234v3") == ("2401.01234", 3)

    def test_new_style_arxiv_prefix_case_insensitive(self):
        assert canonical_arxiv_id("ARXIV:2401.01234") == ("2401.01234", None)

    def test_new_style_five_digit_sequence(self):
        assert canonical_arxiv_id("2401.12345") == ("2401.12345", None)

    def test_new_style_abs_url(self):
        assert canonical_arxiv_id("https://arxiv.org/abs/2401.01234") == ("2401.01234", None)

    def test_new_style_pdf_url(self):
        assert canonical_arxiv_id("https://arxiv.org/pdf/2401.01234v2") == ("2401.01234", 2)

    def test_new_style_pdf_url_with_extension(self):
        assert canonical_arxiv_id("https://arxiv.org/pdf/2401.01234v2.pdf") == ("2401.01234", 2)

    def test_new_style_url_no_scheme(self):
        assert canonical_arxiv_id("arxiv.org/abs/2401.01234") == ("2401.01234", None)

    def test_new_style_url_with_www(self):
        assert canonical_arxiv_id("https://www.arxiv.org/abs/2401.01234") == ("2401.01234", None)

    def test_old_style_bare(self):
        assert canonical_arxiv_id("hep-th/9901001") == ("hep-th/9901001", None)

    def test_old_style_with_version(self):
        assert canonical_arxiv_id("math.GT/0309136v2") == ("math.GT/0309136", 2)

    def test_old_style_abs_url(self):
        assert canonical_arxiv_id("https://arxiv.org/abs/hep-th/9901001") == ("hep-th/9901001", None)

    def test_strips_surrounding_whitespace(self):
        assert canonical_arxiv_id("  2401.01234  ") == ("2401.01234", None)

    def test_versions_of_same_paper_share_base_id(self):
        base_a, _ = canonical_arxiv_id("2401.01234v1")
        base_b, _ = canonical_arxiv_id("2401.01234v9")
        assert base_a == base_b

    @pytest.mark.parametrize(
        "garbage",
        [
            "",
            "not an id at all",
            "12345",
            "2401.123",  # only 3 digits after the dot -- must be 4 or 5
            "hep-th",  # old style with no sequence number
            "https://example.com/abs/2401.01234",  # wrong domain
        ],
    )
    def test_garbage_raises_value_error(self, garbage):
        with pytest.raises(ValueError):
            canonical_arxiv_id(garbage)


class TestNormalizeText:
    def test_crlf_normalised_to_lf(self):
        assert normalize_text("a\r\nb") == "a\nb"

    def test_cr_normalised_to_lf(self):
        assert normalize_text("a\rb") == "a\nb"

    def test_tabs_become_spaces(self):
        assert normalize_text("a\tb") == "a b"

    def test_trailing_whitespace_stripped_per_line(self):
        # The trailing "\n" produces a trailing empty line, which is
        # preserved (rstrip of "" is still "") -- normalize_text only ever
        # strips trailing whitespace *within* a line, never a trailing
        # newline itself.
        assert normalize_text("a  \nb\t\n") == "a\nb\n"

    def test_idempotent(self):
        text = "already\nnormalised text"
        assert normalize_text(normalize_text(text)) == normalize_text(text)


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_sensitive_to_content(self):
        assert content_hash("hello") != content_hash("hellp")

    def test_is_hex_sha256(self):
        digest = content_hash("hello")
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex


class TestChunkUid:
    def test_deterministic(self):
        a = chunk_uid("doc1", "fixed@abcd1234", 0, 10)
        b = chunk_uid("doc1", "fixed@abcd1234", 0, 10)
        assert a == b

    def test_sensitive_to_each_coordinate(self):
        base = chunk_uid("doc1", "fixed@abcd1234", 0, 10)
        assert base != chunk_uid("doc2", "fixed@abcd1234", 0, 10)
        assert base != chunk_uid("doc1", "fixed@ffffffff", 0, 10)
        assert base != chunk_uid("doc1", "fixed@abcd1234", 1, 10)
        assert base != chunk_uid("doc1", "fixed@abcd1234", 0, 11)

    def test_insensitive_to_text_content(self):
        """chunk_uid hashes coordinates, never text -- contextual retrieval
        depends on this (see selfrag.ids module docstring)."""
        # There is no text parameter at all; this test documents that fact
        # by asserting the same coordinates always give the same id
        # regardless of what the caller does with the resulting Chunk.text.
        a = chunk_uid("doc1", "fixed@abcd1234", 0, 10)
        b = chunk_uid("doc1", "fixed@abcd1234", 0, 10)
        assert a == b

    def test_rejects_negative_start(self):
        with pytest.raises(ValueError):
            chunk_uid("doc1", "fixed@abcd1234", -1, 10)

    def test_rejects_empty_span(self):
        with pytest.raises(ValueError):
            chunk_uid("doc1", "fixed@abcd1234", 5, 5)

    def test_rejects_inverted_span(self):
        with pytest.raises(ValueError):
            chunk_uid("doc1", "fixed@abcd1234", 10, 5)

    def test_rejects_separator_in_doc_id(self):
        with pytest.raises(ValueError):
            chunk_uid("doc\x1f1", "fixed@abcd1234", 0, 10)

    def test_rejects_separator_in_config_id(self):
        with pytest.raises(ValueError):
            chunk_uid("doc1", "fixed\x1f@abcd1234", 0, 10)


class TestConfigHash:
    def test_deterministic(self):
        cfg = {"a": 1, "b": 2}
        assert config_hash(cfg) == config_hash(cfg)

    def test_key_order_does_not_matter(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_int_and_equal_float_collapse_to_same_hash(self):
        assert config_hash({"a": 1}) == config_hash({"a": 1.0})

    def test_sensitive_to_value_change(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_sensitive_to_nested_value_change(self):
        assert config_hash({"a": {"b": 1}}) != config_hash({"a": {"b": 2}})

    def test_respects_length_parameter(self):
        assert len(config_hash({"a": 1}, length=8)) == 8
        assert len(config_hash({"a": 1}, length=16)) == 16

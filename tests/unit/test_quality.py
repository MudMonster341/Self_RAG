"""Tests for selfrag.ingest.quality -- deterministic parse-quality metrics.

Every test builds a ``ParsedDocument`` directly (no real parser run needed)
so each metric can be exercised in isolation against a small, hand-built
document whose expected score is obvious by inspection.
"""

from __future__ import annotations

import pytest

from selfrag.ingest.latex import ParsedDocument, Section
from selfrag.ingest.quality import ParseQualityReport, aggregate_reports, assess_quality


def _doc(text: str, sections: list[Section] | None = None, parser_id: str = "latex") -> ParsedDocument:
    return ParsedDocument(text=text, parser_id=parser_id, sections=sections or [])


class TestHeaderFooterContamination:
    def test_arxiv_stamp_line_is_contamination(self):
        text = "Real prose line one.\n\narXiv:2401.01234v3 [cs.IR] 3 Jan 2024\n\nReal prose line two."
        report = assess_quality("doc1", _doc(text))
        assert report.header_footer_contamination_rate > 0.0

    def test_bare_page_number_line_is_contamination(self):
        text = "Some prose.\n\n7\n\nMore prose."
        report = assess_quality("doc1", _doc(text))
        assert report.header_footer_contamination_rate > 0.0

    def test_repeated_short_line_is_contamination(self):
        running_head = "Proceedings of SelfRAG Workshop"
        lines = [running_head, "Body one.", running_head, "Body two.", running_head, "Body three."]
        text = "\n\n".join(lines)
        report = assess_quality("doc1", _doc(text))
        assert report.header_footer_contamination_rate > 0.0

    def test_clean_prose_has_zero_contamination(self):
        text = "This is a normal sentence.\n\nAnd here is another normal sentence entirely."
        report = assess_quality("doc1", _doc(text))
        assert report.header_footer_contamination_rate == 0.0

    def test_empty_document_has_zero_rate_not_a_crash(self):
        report = assess_quality("doc1", _doc(""))
        assert report.header_footer_contamination_rate == 0.0
        assert report.n_chars == 0
        assert report.n_spans == 0


class TestBrokenMathSpanRate:
    def test_odd_dollar_count_is_broken(self):
        text = "This span has an $unbalanced dollar sign that never closes."
        report = assess_quality("doc1", _doc(text))
        assert report.broken_math_span_rate == 1.0

    def test_balanced_inline_math_is_not_broken(self):
        text = "This span has $x + y$ which is perfectly balanced math."
        report = assess_quality("doc1", _doc(text))
        assert report.broken_math_span_rate == 0.0

    def test_unbalanced_braces_are_broken(self):
        text = "A span with a stray { brace that never closes."
        report = assess_quality("doc1", _doc(text))
        assert report.broken_math_span_rate == 1.0

    def test_lone_backslash_alone_is_not_flagged(self):
        """Escaped characters and unresolved commands are left in place by
        both parsers (deliberately, see latex.py); a bare backslash is not
        by itself a sign of broken extraction."""
        text = "A rate of 50\\% was observed, roughly."
        report = assess_quality("doc1", _doc(text))
        assert report.broken_math_span_rate == 0.0

    def test_unmatched_display_math_bracket_is_broken(self):
        text = "Some prose with an unmatched \\[ display bracket."
        report = assess_quality("doc1", _doc(text))
        assert report.broken_math_span_rate == 1.0


class TestUnexpandedMacroRate:
    def test_leaked_newcommand_definition_is_flagged(self):
        text = "Normal prose here.\n\n\\newcommand{\\docs}{\\ensuremath{z}} more text."
        report = assess_quality("doc1", _doc(text))
        assert report.unexpanded_macro_rate > 0.0

    def test_surviving_passthrough_macro_is_flagged(self):
        """`latex.py` is specifically supposed to unwrap `\\ensuremath` --
        any surviving occurrence means that unwrapping failed, which is
        exactly the dominant LaTeX residue this metric exists to catch."""
        text = "Normal prose here.\n\nWe use \\ensuremath{z} for the query."
        report = assess_quality("doc1", _doc(text))
        assert report.unexpanded_macro_rate > 0.0

    def test_stray_empty_braces_are_flagged(self):
        text = "Normal prose here.\n\nRAG-Sequence{} is a model."
        report = assess_quality("doc1", _doc(text))
        assert report.unexpanded_macro_rate > 0.0

    def test_clean_prose_has_zero_rate(self):
        text = "Normal prose sentence number one.\n\nAnother clean sentence entirely."
        report = assess_quality("doc1", _doc(text))
        assert report.unexpanded_macro_rate == 0.0

    def test_unknown_command_is_not_flagged(self):
        """Unknown commands are deliberately left in place by the parser
        (see latex.py's module docstring); flagging them would make every
        LaTeX document look broken by design rather than by defect."""
        text = "Normal prose here.\n\nA \\customcommand{x} appears in this span."
        report = assess_quality("doc1", _doc(text))
        assert report.unexpanded_macro_rate == 0.0

    def test_empty_document_has_zero_rate_not_a_crash(self):
        report = assess_quality("doc1", _doc(""))
        assert report.unexpanded_macro_rate == 0.0


class TestReferenceOnlySpanRate:
    def test_numbered_bracket_reference_list_is_reference_only(self):
        text = "[1] Foo et al. 2020.\n[2] Bar et al. 2021.\n[3] Baz et al. 2022."
        report = assess_quality("doc1", _doc(text))
        assert report.reference_only_span_rate == 1.0

    def test_ordinary_prose_is_not_reference_only(self):
        text = "This is a sentence describing the method in detail."
        report = assess_quality("doc1", _doc(text))
        assert report.reference_only_span_rate == 0.0


class TestSectionLabelCoverage:
    def test_full_coverage_when_sections_span_whole_document(self):
        text = "Method text here."
        sections = [Section(section_path="1 Method", char_start=0, char_end=len(text))]
        report = assess_quality("doc1", _doc(text, sections))
        assert report.mean_section_label_coverage == 1.0

    def test_zero_coverage_with_no_sections(self):
        report = assess_quality("doc1", _doc("Unstructured text blob."))
        assert report.mean_section_label_coverage == 0.0

    def test_partial_coverage(self):
        text = "plain preamble text. labelled method text."
        start = text.index("labelled")
        sections = [Section(section_path="1 Method", char_start=start, char_end=len(text))]
        report = assess_quality("doc1", _doc(text, sections))
        expected = (len(text) - start) / len(text)
        assert report.mean_section_label_coverage == pytest.approx(expected)

    def test_coverage_is_clamped_to_one_even_with_bad_section_bounds(self):
        text = "short"
        sections = [Section(section_path="1 X", char_start=0, char_end=10_000)]
        report = assess_quality("doc1", _doc(text, sections))
        assert report.mean_section_label_coverage == 1.0


class TestSpanLengthMetrics:
    def test_short_span_rate_counts_spans_under_threshold(self):
        text = "Short.\n\nThis paragraph is considerably longer than forty characters for sure."
        report = assess_quality("doc1", _doc(text))
        assert report.n_spans == 2
        assert report.short_span_rate == pytest.approx(0.5)

    def test_mean_span_length_matches_manual_computation(self):
        text = "abcde\n\nfghij"
        report = assess_quality("doc1", _doc(text))
        assert report.mean_span_length_chars == pytest.approx(5.0)


class TestAssessQualityBasics:
    def test_parser_id_and_doc_id_are_carried_through(self):
        report = assess_quality("2401.01234", _doc("text", parser_id="pdf_fallback"))
        assert report.doc_id == "2401.01234"
        assert report.parser_id == "pdf_fallback"

    def test_returns_parse_quality_report_instance(self):
        report = assess_quality("doc1", _doc("some text"))
        assert isinstance(report, ParseQualityReport)


class TestAggregateReports:
    def test_raises_on_empty_sequence(self):
        with pytest.raises(ValueError):
            aggregate_reports([])

    def test_averages_across_reports(self):
        r1 = assess_quality("d1", _doc("Clean prose sentence number one here today."))
        r2 = assess_quality("d2", _doc("$broken"))
        agg = aggregate_reports([r1, r2])
        assert agg["broken_math_span_rate"] == pytest.approx((r1.broken_math_span_rate + r2.broken_math_span_rate) / 2)

    def test_unexpanded_macro_rate_is_averaged(self):
        r1 = assess_quality("d1", _doc("Clean prose sentence number one here today."))
        r2 = assess_quality("d2", _doc("Residue: \\ensuremath{z} survives here."))
        agg = aggregate_reports([r1, r2])
        assert agg["unexpanded_macro_rate"] == pytest.approx(
            (r1.unexpanded_macro_rate + r2.unexpanded_macro_rate) / 2
        )

    def test_single_report_aggregates_to_itself(self):
        r = assess_quality("d1", _doc("Some text here for a single report test."))
        agg = aggregate_reports([r])
        assert agg["mean_span_length_chars"] == pytest.approx(r.mean_span_length_chars)

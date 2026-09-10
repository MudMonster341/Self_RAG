"""Tests for selfrag.ingest.latex -- the primary document parser.

Fixtures are small ``.tex`` documents written to ``tmp_path`` at test time
(never on the network, never checked in as separate files) so each test is
self-contained and the exact bytes under test are visible right next to
the assertion that checks them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from selfrag.ids import normalize_text
from selfrag.ingest.latex import (
    LatexCycleError,
    LatexSourceError,
    ParsedDocument,
    Section,
    advance_heading,
    parse_latex_source,
)


def _write_source(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


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
        reconstructed.append(doc.text[cursor : sec.char_start])  # gap, may be empty
        reconstructed.append(doc.text[sec.char_start : sec.char_end])
        cursor = sec.char_end
    reconstructed.append(doc.text[cursor:])
    assert "".join(reconstructed) == doc.text


class TestNestedSections:
    def test_numbering_and_path(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \documentclass{article}
                \begin{document}
                \section{Introduction}
                Intro text.
                \section{Method}
                Method text.
                \subsection{Retrieval}
                Retrieval text.
                \subsubsection{Dense encoder}
                Encoder text.
                \section{Conclusion}
                Conclusion text.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == [
            "1 Introduction",
            "2 Method",
            "2 Method > 2.1 Retrieval",
            "2 Method > 2.1 Retrieval > 2.1.1 Dense encoder",
            "3 Conclusion",
        ]
        _assert_invariants(doc)

    def test_new_top_level_section_resets_subsection_counter(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{A}
                \subsection{A.1}
                \section{B}
                \subsection{B.1}
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == ["1 A", "1 A > 1.1 A.1", "2 B", "2 B > 2.1 B.1"]

    def test_starred_section_is_unnumbered_and_does_not_consume_a_number(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{One}
                \section*{Acknowledgements}
                \section{Two}
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == ["1 One", "Acknowledgements", "2 Two"]

    def test_paragraph_level_is_unnumbered_and_nests_under_subsection(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Method}
                \subsection{Retrieval}
                \paragraph{Dense encoder}
                Text.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths[-1] == "1 Method > 1.1 Retrieval > Dense encoder"

    def test_consecutive_paragraphs_are_siblings_not_nested(self, tmp_path: Path):
        """Regression for the sibling-heading defect: two \\paragraph
        commands at the same nominal level must replace each other in the
        path stack, not nest -- this is exactly the RAG-Sequence/RAG-Token
        structure from the real paper (arXiv:2005.11401)."""
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Methods}
                \subsection{Models}
                \paragraph{RAG-Sequence Model}
                Text one.
                \paragraph{RAG-Token Model}
                Text two.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == [
            "1 Methods",
            "1 Methods > 1.1 Models",
            "1 Methods > 1.1 Models > RAG-Sequence Model",
            "1 Methods > 1.1 Models > RAG-Token Model",
        ]
        _assert_invariants(doc)

    def test_n_consecutive_paragraph_siblings_do_not_grow_the_path(self, tmp_path: Path):
        """Before the fix, N consecutive same-level unnumbered siblings grew
        the path unboundedly (A > A > B > A > B > C > ...). Four siblings
        here is enough to distinguish "still growing" from "fixed"."""
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Methods}
                \paragraph{A}
                a.
                \paragraph{B}
                b.
                \paragraph{C}
                c.
                \paragraph{D}
                d.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == [
            "1 Methods",
            "1 Methods > A",
            "1 Methods > B",
            "1 Methods > C",
            "1 Methods > D",
        ]

    def test_consecutive_subsections_under_the_same_section_are_siblings(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Method}
                \subsection{Retrieval}
                Text.
                \subsection{Generation}
                Text.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == [
            "1 Method",
            "1 Method > 1.1 Retrieval",
            "1 Method > 1.2 Generation",
        ]

    def test_consecutive_subsubsections_under_the_same_subsection_are_siblings(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Method}
                \subsection{Retrieval}
                \subsubsection{Dense}
                Text.
                \subsubsection{Sparse}
                Text.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == [
            "1 Method",
            "1 Method > 1.1 Retrieval",
            "1 Method > 1.1 Retrieval > 1.1.1 Dense",
            "1 Method > 1.1 Retrieval > 1.1.2 Sparse",
        ]

    def test_paragraph_following_a_deeper_heading_pops_back_correctly(self, tmp_path: Path):
        """A \\paragraph after a \\subsubsection sits one below it, and a
        following \\subsection must pop both the paragraph and the
        subsubsection back to the section level, not nest under either."""
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Method}
                \subsection{Retrieval}
                \subsubsection{Dense}
                \paragraph{Detail}
                Text.
                \subsection{Generation}
                Text.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        paths = [s.section_path for s in doc.sections]
        assert paths == [
            "1 Method",
            "1 Method > 1.1 Retrieval",
            "1 Method > 1.1 Retrieval > 1.1.1 Dense",
            "1 Method > 1.1 Retrieval > 1.1.1 Dense > Detail",
            "1 Method > 1.2 Generation",
        ]
        _assert_invariants(doc)


class TestInputInclude:
    def test_input_is_resolved_and_expanded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Introduction}
                Intro.
                \input{parts/method}
                \end{document}
                """,
                "parts/method.tex": r"""
                \section{Method}
                Method body text.
                """,
            },
        )
        doc = parse_latex_source(src)
        assert "Method body text." in doc.text
        assert [s.section_path for s in doc.sections] == ["1 Introduction", "2 Method"]

    def test_include_behaves_like_input(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \include{extra}
                \end{document}
                """,
                "extra.tex": "Extra body text.",
            },
        )
        doc = parse_latex_source(src)
        assert "Extra body text." in doc.text

    def test_input_extension_is_optional(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"\begin{document}\input{extra}\end{document}",
                "extra.tex": "Body from extra.",
            },
        )
        doc = parse_latex_source(src)
        assert "Body from extra." in doc.text

    def test_missing_input_target_raises_source_error(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}\input{nonexistent}\end{document}"},
        )
        with pytest.raises(LatexSourceError):
            parse_latex_source(src)

    def test_explicit_main_file_is_honoured(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "a.tex": r"\begin{document}A body.\end{document}",
                "b.tex": r"\begin{document}B body.\end{document}",
            },
        )
        doc = parse_latex_source(src, main_file="b.tex")
        assert "B body." in doc.text

    def test_ambiguous_main_file_raises(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "a.tex": r"\documentclass{article}\begin{document}A\end{document}",
                "b.tex": r"\documentclass{article}\begin{document}B\end{document}",
            },
        )
        with pytest.raises(LatexSourceError):
            parse_latex_source(src)

    def test_no_tex_files_raises(self, tmp_path: Path):
        with pytest.raises(LatexSourceError):
            parse_latex_source(tmp_path)


class TestCycleDetection:
    def test_direct_cycle_raises(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"\begin{document}\input{a}\end{document}",
                "a.tex": r"\input{main}",
            },
        )
        with pytest.raises(LatexCycleError):
            parse_latex_source(src)

    def test_indirect_cycle_raises(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"\begin{document}\input{a}\end{document}",
                "a.tex": r"\input{b}",
                "b.tex": r"\input{a}",
            },
        )
        with pytest.raises(LatexCycleError):
            parse_latex_source(src)

    def test_diamond_shaped_include_is_not_a_false_cycle(self, tmp_path: Path):
        """a is \\input twice from different places -- not a cycle, just reused."""
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"\begin{document}\input{a}\input{a}\end{document}",
                "a.tex": "Shared text.",
            },
        )
        doc = parse_latex_source(src)
        assert doc.text.count("Shared text.") == 2


class TestComments:
    def test_plain_comment_is_stripped(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": "\\begin{document}\nKept text. % this is a comment\nMore kept text.\n\\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "this is a comment" not in doc.text
        assert "Kept text." in doc.text
        assert "More kept text." in doc.text

    def test_escaped_percent_is_not_a_comment(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": "\\begin{document}\nA rate of 50\\% was observed.\n\\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "50\\% was observed" in doc.text

    def test_escaped_backslash_then_real_comment(self, tmp_path: Path):
        """``\\\\%`` is an escaped backslash followed by a *real* comment,
        not an escaped percent -- parity of the backslash run decides it."""
        src = _write_source(
            tmp_path,
            {"main.tex": "\\begin{document}\nKeep this \\\\% drop this\nKeep this too.\n\\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "drop this" not in doc.text
        assert "Keep this too." in doc.text


class TestFigureAndTableCaptions:
    def test_figure_caption_is_kept_and_tagged(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \begin{figure}
                \includegraphics{diagram.png}
                \caption{An overview of the retrieval pipeline.}
                \label{fig:overview}
                \end{figure}
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "Figure: An overview of the retrieval pipeline." in doc.text
        assert "includegraphics" not in doc.text
        assert "fig:overview" not in doc.text

    def test_table_caption_is_kept_and_tabular_body_is_dropped(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \begin{table}
                \begin{tabular}{cc}
                a & b \\
                c & d \\
                \end{tabular}
                \caption{Retrieval results.}
                \end{table}
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "Table: Retrieval results." in doc.text
        assert "tabular" not in doc.text

    def test_float_without_caption_is_dropped_entirely(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                Before.
                \begin{figure}
                \includegraphics{no_caption.png}
                \end{figure}
                After.
                """
            },
        )
        doc = parse_latex_source(src)
        assert "Figure:" not in doc.text
        assert "Before." in doc.text
        assert "After." in doc.text


class TestDisplayAndInlineMath:
    def test_equation_environment_becomes_placeholder(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                Before the equation.
                \begin{equation}
                x = y + z
                \end{equation}
                After the equation.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "[DISPLAY_MATH]" in doc.text
        assert "x = y + z" not in doc.text
        # placeholder must not fuse "Before" and "After" into one sentence
        before_idx = doc.text.index("Before the equation.")
        after_idx = doc.text.index("After the equation.")
        placeholder_idx = doc.text.index("[DISPLAY_MATH]")
        assert before_idx < placeholder_idx < after_idx

    def test_bracket_display_math_becomes_placeholder(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}A. \[ x^2 \] B.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "[DISPLAY_MATH]" in doc.text
        assert "x^2" not in doc.text

    def test_inline_math_dollar_delimiters_are_kept(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}We compute $k$-NN over $x$.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "$k$-NN over $x$" in doc.text

    def test_builtin_math_formatting_macros_are_unwrapped(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}Let $\mathbf{x}$ be the query vector.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "Let $x$ be the query vector." in doc.text
        assert "mathbf" not in doc.text


class TestBibliography:
    def test_bibliography_is_stripped_and_start_recorded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Conclusion}
                Final words.
                \begin{thebibliography}{9}
                \bibitem{foo} Foo et al.
                \bibitem{bar} Bar et al.
                \end{thebibliography}
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "bibitem" not in doc.text
        assert doc.bibliography_char_start is not None
        # nothing but whitespace survives after the bibliography marker --
        # the marker records *where* the reference list started, not a
        # promise that it was the literal last byte of the source.
        assert doc.text[doc.bibliography_char_start :].strip() == ""

    def test_no_bibliography_leaves_start_none(self, tmp_path: Path):
        src = _write_source(tmp_path, {"main.tex": r"\begin{document}No refs here.\end{document}"})
        doc = parse_latex_source(src)
        assert doc.bibliography_char_start is None

    def test_content_after_bibliography_still_parsed(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \begin{thebibliography}{9}
                \bibitem{foo} Foo et al.
                \end{thebibliography}
                \appendix
                \section{Extra material}
                Appendix text.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert doc.bibliography_char_start is not None
        assert [s.section_path for s in doc.sections] == ["1 Extra material"]
        assert doc.sections[0].char_start >= doc.bibliography_char_start


class TestMacros:
    def test_noarg_macro_is_expanded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \newcommand{\projname}{SelfRAG}
                \begin{document}
                Welcome to \projname, a retrieval system.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "Welcome to SelfRAG, a retrieval system." in doc.text
        assert "projname" not in doc.text
        assert "newcommand" not in doc.text

    def test_single_arg_macro_is_expanded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \newcommand{\code}[1]{`#1`}
                \begin{document}
                Run \code{pytest} to test it.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "Run `pytest` to test it." in doc.text

    def test_multi_arg_macro_definition_is_stripped_but_not_expanded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \newcommand{\pair}[2]{(#1, #2)}
                \begin{document}
                A \pair{x}{y} pair.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "newcommand" not in doc.text
        # left in place rather than mangled -- see module docstring
        assert "\\pair{x}{y}" in doc.text

    def test_macro_with_nested_brace_argument_is_left_in_place(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \newcommand{\wrap}[1]{[#1]}
                \begin{document}
                A \wrap{nested {brace} value} case.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        # the regex only substitutes single-level-brace-free arguments, so
        # this invocation must survive untouched rather than being mangled
        assert "\\wrap{nested {brace} value}" in doc.text

    def test_renewcommand_is_also_expanded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \renewcommand{\thefootnote}{X}
                \begin{document}
                Footnote marker: \thefootnote.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "Footnote marker: X." in doc.text

    def test_ensuremath_is_unwrapped_to_its_argument(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}We define \ensuremath{z} as the query.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "We define z as the query." in doc.text
        assert "ensuremath" not in doc.text

    def test_tiny_and_small_unwrap_to_their_argument(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}A \tiny{tiny} note and a \small{small} note.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "A tiny note and a small note." in doc.text
        assert "\\tiny" not in doc.text
        assert "\\small" not in doc.text

    def test_stray_empty_braces_after_zero_arg_macro_are_removed(self, tmp_path: Path):
        """``\\RAGSequence{}``-style invocations (real idiom, from
        arXiv:2005.11401's ``\\raganswer{}``) leave a stray ``{}`` once the
        bare macro token is substituted -- arity-0 expansion only consumes
        the command name, not a following unrelated brace group."""
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \newcommand{\raganswer}{RAG-Sequence}
                \begin{document}
                \paragraph{\raganswer{} Model}
                The \raganswer{} model retrieves once.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "{}" not in doc.text
        assert "RAG-Sequence Model" in doc.text
        assert "The RAG-Sequence model retrieves once." in doc.text

    def test_empty_section_title_is_still_recognised_as_a_heading(self, tmp_path: Path):
        """The stray-empty-brace cleanup runs *after* section processing,
        specifically so a genuinely empty but structurally required
        argument -- ``\\section{}``, an anonymous numbered heading -- is
        consumed by ``_HEADING_RE`` before the cleanup ever sees it,
        rather than having its braces stripped first and turning a
        recognised heading into unrecognised literal text."""
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}\section{}Body text.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "\\section" not in doc.text
        assert doc.sections and doc.sections[0].section_path == "1"

    def test_empty_group_not_adjacent_to_a_word_is_left_alone(self, tmp_path: Path):
        """Scoped to "directly follows a word character or a closing
        brace" -- a bare ``{}`` with a space before it is not macro
        residue and must survive untouched."""
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}Some text ends here. {} A new empty group.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "{}" in doc.text

    def test_in_body_newcommand_with_nested_braces_is_stripped(self, tmp_path: Path):
        """Regression for the brace-depth-matching defect: a \\newcommand
        appearing AFTER \\begin{document} whose replacement text itself
        nests a builtin macro (real shape:
        ``\\newcommand{\\docs}{\\ensuremath{\\mathbf{z}}}`` from
        arXiv:2005.11401) used to leak into the body as literal text
        because the old one-level brace regex could never find its
        closing brace."""
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Method}
                \newcommand{\docs}{\ensuremath{\mathbf{z}}}
                We retrieve \docs and rank it.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "newcommand" not in doc.text
        assert "\\docs" not in doc.text
        assert "We retrieve z and rank it." in doc.text
        _assert_invariants(doc)

    def test_newcommand_body_nested_three_levels_deep_is_fully_captured(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \newcommand{\topk}{\ensuremath{\text{top-\textbf{k}}}}
                \begin{document}
                See \topk for details.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "newcommand" not in doc.text
        assert "See top-k for details." in doc.text

    def test_declare_math_operator_is_stripped_and_expanded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \DeclareMathOperator*{\argmax}{arg\,max}
                \begin{document}
                We compute \argmax over candidates.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "DeclareMathOperator" not in doc.text
        assert "We compute arg\\,max over candidates." in doc.text

    def test_providecommand_definition_is_stripped_and_expanded(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \providecommand{\myterm}{term}
                \begin{document}
                A \myterm appears here.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "providecommand" not in doc.text
        assert "A term appears here." in doc.text


class TestTitleAndAbstract:
    def test_title_is_extracted_and_removed_from_body(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \documentclass{article}
                \title{A Great Paper}
                \begin{document}
                \maketitle
                Body text.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert doc.title == "A Great Paper"
        assert "A Great Paper" not in doc.text

    def test_title_is_cleaned_of_formatting_and_labels(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \title{A Paper About \emph{Things}\label{tit}}
                \begin{document}
                Body.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert doc.title == "A Paper About Things"

    def test_abstract_environment_becomes_its_own_section(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \begin{abstract}
                We present a system for retrieval.
                \end{abstract}
                \section{Introduction}
                Intro.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert doc.abstract == "We present a system for retrieval."
        assert doc.sections[0].section_path == "Abstract"
        assert "We present a system for retrieval." in doc.text[doc.sections[0].char_start : doc.sections[0].char_end]

    def test_abstract_command_form_is_also_recognised(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}\abstract{Short abstract text.}\end{document}"},
        )
        doc = parse_latex_source(src)
        assert doc.abstract == "Short abstract text."


class TestCitationsRefsLabels:
    def test_cite_keys_are_collected_and_bracketed_inline(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}As shown by \cite{foo2020,bar2021}.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert doc.citations == ["foo2020", "bar2021"]
        assert "As shown by [foo2020, bar2021]." in doc.text

    def test_citations_are_deduplicated_preserving_first_seen_order(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}\cite{a,b} and later \cite{b,c}.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert doc.citations == ["a", "b", "c"]

    def test_ref_becomes_bracketed_label_not_a_number(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {"main.tex": r"\begin{document}See Section~\ref{sec:method} for details.\end{document}"},
        )
        doc = parse_latex_source(src)
        assert "See Section [sec:method] for details." in doc.text

    def test_label_is_stripped_without_a_trace(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \begin{document}
                \section{Method}
                \label{sec:method}
                Body of method.
                \end{document}
                """
            },
        )
        doc = parse_latex_source(src)
        assert "label" not in doc.text
        assert "sec:method" not in doc.text


class TestAdvanceHeading:
    """Direct unit coverage of the numbering primitive shared with pdf_fallback."""

    def test_sibling_headings_increment_independently(self):
        counters = [0, 0, 0]
        stack: list[str] = []
        d1, p1 = advance_heading(1, "Intro", True, counters, stack)
        d2, p2 = advance_heading(1, "Method", True, counters, stack)
        assert (d1, p1) == ("1 Intro", "1 Intro")
        assert (d2, p2) == ("2 Method", "2 Method")

    def test_deeper_level_beyond_counters_length_is_unnumbered(self):
        counters = [0, 0, 0]
        stack: list[str] = []
        advance_heading(1, "Method", True, counters, stack)
        display, path = advance_heading(4, "Detail", True, counters, stack)
        assert display == "Detail"
        assert path == "1 Method > Detail"

    def test_consecutive_unnumbered_siblings_replace_not_nest(self):
        """Direct unit-level regression for the sibling-heading defect: two
        headings at the same level beyond ``len(counters)`` must replace
        each other in the stack, not accumulate."""
        counters = [0, 0, 0]
        stack: list[str] = []
        advance_heading(1, "Method", True, counters, stack)
        d1, p1 = advance_heading(4, "First", True, counters, stack)
        d2, p2 = advance_heading(4, "Second", True, counters, stack)
        assert (d1, p1) == ("First", "1 Method > First")
        assert (d2, p2) == ("Second", "1 Method > Second")

    def test_many_consecutive_unnumbered_siblings_stay_flat(self):
        counters = [0, 0, 0]
        stack: list[str] = []
        advance_heading(1, "Method", True, counters, stack)
        paths = [advance_heading(4, letter, True, counters, stack)[1] for letter in "ABCDE"]
        assert paths == [f"1 Method > {letter}" for letter in "ABCDE"]


class TestParsedDocumentAndSection:
    def test_section_is_frozen_dataclass(self):
        sec = Section(section_path="1 Intro", char_start=0, char_end=5)
        with pytest.raises(AttributeError):
            sec.char_start = 1  # type: ignore[misc]

    def test_parsed_document_defaults(self):
        doc = ParsedDocument(text="hello", parser_id="latex")
        assert doc.title == ""
        assert doc.abstract == ""
        assert doc.sections == []
        assert doc.citations == []
        assert doc.bibliography_char_start is None


class TestInvariantsAcrossRealisticDocument:
    def test_full_document_satisfies_all_invariants(self, tmp_path: Path):
        src = _write_source(
            tmp_path,
            {
                "main.tex": r"""
                \documentclass{article}
                \newcommand{\vecx}{\mathbf{x}}
                \title{A Paper About Retrieval}
                \begin{document}
                \maketitle
                \begin{abstract}
                We study $\vecx$ and cite \cite{foo2020} prior work.
                \end{abstract}
                \section{Introduction}
                Intro text with a ref \ref{sec:method} and a comment. % dropped
                \input{parts/method}
                \section{Conclusion}
                \begin{equation}
                x = y
                \end{equation}
                Done.
                \begin{thebibliography}{9}
                \bibitem{foo2020} Foo, 2020.
                \end{thebibliography}
                \end{document}
                """,
                "parts/method.tex": r"""
                \section{Method}
                \label{sec:method}
                \subsection{Retrieval}
                \begin{figure}
                \includegraphics{x.png}
                \caption{Pipeline overview.}
                \end{figure}
                """,
            },
        )
        doc = parse_latex_source(src)
        _assert_invariants(doc)
        assert len(doc.sections) >= 4

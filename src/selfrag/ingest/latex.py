"""LaTeX source parser -- the primary path for arXiv document ingestion.

Over 90% of arXiv submissions ship LaTeX source, and source parsing beats
PDF extraction on both quality and CPU cost, and gives section structure
almost for free. ``pdf_fallback.py`` exists only for the source-less
minority and produces the exact same :class:`ParsedDocument`/:class:`Section`
types so a caller cannot tell which parser ran except via ``parser_id``.

This is deliberately **not** a LaTeX engine. It is a comment-aware,
state-machine-and-regex-level text extractor that is honest about its
limits (see the docstrings on each processing step below). Anything it is
unsure about is left in place rather than mangled -- per CLAUDE.md, a
config that produces defensibly-graded-imperfect text is worth more than
one that silently corrupts offsets by trying to be clever.

**Deliberately not handled** (documented once, here, rather than scattered):

- Plain-TeX primitives (``\\def``, ``\\let``, catcode changes, ``\\csname``).
  ``\\newcommand``/``\\renewcommand``/``\\providecommand``/
  ``\\DeclareMathOperator`` are recognised and always stripped -- preamble
  or body, wherever they occur -- via proper brace-depth matching (escaped
  ``\\{``/``\\}`` do not count as grouping), so a replacement text with
  nested groups (``\\newcommand{\\docs}{\\ensuremath{\\mathbf{z}}}``) is
  captured whole rather than truncated at the first inner brace.
- ``\\newcommand`` with 2+ required arguments, or with an optional
  first-argument default (``\\newcommand{\\x}[2][default]{...}``). The
  *definition* is still stripped from the body regardless of how deeply
  its replacement text nests (so it never appears as literal text), but no
  expansion is attempted -- invocations are left exactly as written.
- Macro arguments containing nested braces (``\\foo{a{b}c}``) are not
  substituted; the invocation is left in place rather than guessed at.
- ``verbatim``/``lstlisting``/``minted`` environments get no special
  treatment: comment-stripping and macro expansion still run over their
  contents, which can corrupt code listings. Rare enough in practice
  (papers, not code repositories) that this is an accepted gap, not a bug.
- Cross-reference *resolution*: ``\\ref``/``\\eqref``/``\\pageref``/``\\cref``
  become a bracketed label (e.g. ``[sec:method]``), never the number or
  title the label would have resolved to at compile time.
- ``\\appendix`` does not reset section numbering to letters; appendix
  sections keep counting numerically from wherever the body left off.
- Legacy space-delimited ``\\input path`` (no braces) is not recognised,
  only ``\\input{path}``/``\\include{path}``.
- Escaped special characters (``\\&``, ``\\%``, ``\\_``, ``\\#``, ``\\$``)
  are left in their escaped form rather than unescaped, specifically to
  avoid a lone unescaped ``$`` being misread as a math delimiter by a
  later pass -- see the module-level design note in MEMORY.md/decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from selfrag.ids import normalize_text

PARSER_ID = "latex"

# Private-use-area sentinel. Vanishingly unlikely to occur in real LaTeX
# source (it is not a printable character on any standard keyboard/encoding
# path an author would use), so if it ever appears in input text, treat that
# as suspicious rather than silently colliding with our own bookkeeping.
_MARK = "\ue000"


class LatexParseError(ValueError):
    """Base class for LaTeX-source parsing failures.

    A pipeline should catch this (not a bare ``Exception``) and fall back
    to ``pdf_fallback.parse_pdf`` when a PDF rendering of the same
    submission is available -- that is the entire reason this is its own
    exception hierarchy rather than letting arbitrary errors propagate.
    """


class LatexSourceError(LatexParseError):
    """A main file or an ``\\input``/``\\include`` target could not be resolved."""


class LatexCycleError(LatexParseError):
    """``\\input``/``\\include`` formed a cycle.

    Following it would recurse forever, so this is raised the moment a
    file is about to be opened a second time while it is already open
    higher up the same inclusion chain.
    """


@dataclass(frozen=True)
class Section:
    """One heading-to-next-heading span of the parsed body text.

    ``section_path`` mirrors ``selfrag.schema.Chunk.section_path`` -- e.g.
    ``"3 Method > 3.2 Retrieval"`` -- and both offsets are indices into the
    *final*, already-normalised ``ParsedDocument.text``, never into any
    intermediate representation.

    A section's span starts at its own heading's display text and runs to
    the character right before the next heading (of any level, or the
    bibliography marker), or to the end of the document if it is last.
    Sections are therefore contiguous with *each other*, but the document
    is not required to be fully covered: text before the first heading
    (front matter, or a document with no headings at all) is a deliberate
    gap, which the module's invariant tests explicitly allow.
    """

    section_path: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ParsedDocument:
    """Output of every parser in this package -- LaTeX or PDF fallback.

    ``text`` has already been through ``normalize_text`` exactly once, and
    every offset in ``sections``/``bibliography_char_start`` indexes into
    this exact string (see ``selfrag.ingest.__init__`` for why that
    ordering is load-bearing). Callers must never re-normalise ``text`` and
    must never recompute an offset against anything else.
    """

    text: str
    parser_id: str
    title: str = ""
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    bibliography_char_start: int | None = None


def _marker(payload: str) -> str:
    if _MARK in payload:
        raise LatexParseError(
            "internal marker character leaked into parsed content; refusing "
            "to compute offsets against text that could collide with our "
            "own bookkeeping"
        )
    return f"{_MARK}{payload}{_MARK}"


def advance_heading(
    level: int,
    title: str,
    numbered: bool,
    counters: list[int],
    path_stack: list[tuple[int, str]],
) -> tuple[str, str]:
    """Compute ``(display_text, section_path)`` for one heading event.

    Shared by the LaTeX and PDF-fallback parsers so both produce
    ``section_path`` strings in the same shape regardless of source
    format -- a caller comparing two parsers' output should see comparable
    structure, not two different numbering conventions.

    Mutates ``counters`` and ``path_stack`` in place: ``counters[i]`` is the
    running count for heading level ``i + 1``, and is only ever incremented
    when ``numbered`` is true and ``level`` is within ``len(counters)``
    (real LaTeX never auto-numbers ``\\paragraph``, and neither do we; a
    caller wanting to number up to a shallower depth just passes a shorter
    ``counters`` list). Every counter *deeper* than ``level`` resets to 0,
    matching how a new ``\\section`` restarts subsection numbering. An
    unnumbered heading (starred, or deeper than ``len(counters)``) neither
    reads nor mutates ``counters``, matching how ``\\section*`` does not
    consume a number in real LaTeX.

    ``path_stack`` holds ``(level, display)`` pairs -- one per heading
    currently "open" on the path from the document root down to whatever
    heading was last seen -- rather than bare display strings. That is
    load-bearing, not decoration: a numbered heading's depth in the stack
    always equals its own ``level`` (truncating to a fixed index,
    ``level - 1``, before appending is correct and idempotent across
    repeated calls). But a heading beyond ``len(counters)`` -- ``\\paragraph``
    in LaTeX, or any markdown heading past ``###`` in ``pdf_fallback.py`` --
    has no counter to anchor its own depth, so a *fixed* truncation index
    is wrong for it: the first such heading reaches whatever depth the
    numbered prefix leaves it at (which may be shallower than
    ``level - 1``), and every consecutive sibling after it would then find
    the stack already sitting at that same depth and truncate to nothing,
    nesting under the previous sibling instead of replacing it -- growing
    the path without bound over N consecutive siblings. Storing the level
    alongside each frame instead lets every heading, numbered or not, pop
    every currently-open frame at its own level or deeper before pushing
    itself. That is the general rule; it costs the numbered branch nothing
    (popping down to a fixed index and popping "everything at this level or
    deeper" agree exactly whenever depth and level coincide), and it is
    what makes a run of sibling ``\\paragraph``\\ s replace each other
    instead of nesting.
    """
    if numbered and level <= len(counters):
        counters[level - 1] += 1
        for i in range(level, len(counters)):
            counters[i] = 0
        number = ".".join(str(c) for c in counters[:level])
        display = f"{number} {title}" if title else number
    else:
        display = title
    while path_stack and path_stack[-1][0] >= level:
        path_stack.pop()
    path_stack.append((level, display))
    return display, " > ".join(d for _, d in path_stack)


# ---------------------------------------------------------------------------
# Step 1: multi-file resolution (\input / \include) with cycle detection,
# and comment stripping (applied per file, before anything else sees it).
# ---------------------------------------------------------------------------

_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]*)\}")


def _strip_comments(text: str) -> str:
    """Strip ``%`` comments, respecting escaped ``\\%``.

    A naive ``(?<!\\\\)%`` lookbehind is wrong on ``\\\\%`` (an escaped
    backslash followed by a real comment): the character immediately
    before ``%`` is a backslash either way, so a single-character
    lookbehind cannot tell "the ``%`` is escaped" from "the *backslash* was
    escaped and the ``%`` is not". This counts the run of consecutive
    backslashes immediately preceding each ``%`` and uses its parity: an
    even count means the backslashes fully pair off and ``%`` starts a
    comment; an odd count means the last backslash escapes it.

    Does not special-case ``verbatim``/``lstlisting`` bodies -- see the
    module docstring.
    """
    out_lines = []
    for line in text.split("\n"):
        result_chars: list[str] = []
        for ch in line:
            if ch == "%":
                bs = 0
                j = len(result_chars) - 1
                while j >= 0 and result_chars[j] == "\\":
                    bs += 1
                    j -= 1
                if bs % 2 == 0:
                    break
            result_chars.append(ch)
        out_lines.append("".join(result_chars))
    return "\n".join(out_lines)


def _resolve_input_path(source_dir: Path, current_file: Path, target: str) -> Path:
    target = target.strip()
    rel = Path(target)
    if not rel.suffix:
        rel = rel.with_suffix(".tex")
    for candidate in (source_dir / rel, current_file.parent / rel):
        if candidate.is_file():
            return candidate.resolve()
    raise LatexSourceError(
        f"cannot resolve \\input/\\include target {target!r} referenced from {current_file}"
    )


def _load_and_expand(path: Path, source_dir: Path, stack: tuple[Path, ...]) -> str:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(p) for p in (*stack, resolved))
        raise LatexCycleError(f"cyclic \\input/\\include detected: {chain}")

    raw = resolved.read_text(encoding="utf-8", errors="replace")
    raw = _strip_comments(raw)
    new_stack = (*stack, resolved)

    def _repl(m: re.Match[str]) -> str:
        target_path = _resolve_input_path(source_dir, resolved, m.group(1))
        return _load_and_expand(target_path, source_dir, new_stack)

    return _INPUT_RE.sub(_repl, raw)


def _find_main_file(source_dir: Path) -> Path:
    """Best-effort choice of the main ``.tex`` file, in order of confidence:

    1. Exactly one file -- unambiguous by construction.
    2. Exactly one file literally named ``main.tex`` -- the overwhelmingly
       common arXiv convention.
    3. Exactly one file containing ``\\documentclass`` -- the structural
       signal a real submission's main file always carries.
    4. Exactly one file that is not ``\\input``/``\\include``d by any other
       file in the tree -- everything else must be a fragment included from
       somewhere, so the one un-included file is the root.

    Raises ``LatexSourceError`` (naming the candidate count) the moment more
    than one candidate survives a step, rather than guessing between them.
    """
    tex_files = sorted(source_dir.rglob("*.tex"))
    if not tex_files:
        raise LatexSourceError(f"no .tex files found under {source_dir}")
    if len(tex_files) == 1:
        return tex_files[0]

    by_name = [p for p in tex_files if p.name == "main.tex"]
    if len(by_name) == 1:
        return by_name[0]

    contents = {p: p.read_text(encoding="utf-8", errors="replace") for p in tex_files}

    with_documentclass = [p for p in tex_files if re.search(r"\\documentclass", contents[p])]
    if len(with_documentclass) == 1:
        return with_documentclass[0]

    included: set[Path] = set()
    for p in tex_files:
        for m in _INPUT_RE.finditer(contents[p]):
            try:
                included.add(_resolve_input_path(source_dir, p, m.group(1)))
            except LatexSourceError:
                continue
    roots = [p for p in tex_files if p.resolve() not in included]
    if len(roots) == 1:
        return roots[0]

    raise LatexSourceError(
        f"cannot determine the main file among {len(tex_files)} .tex files under "
        f"{source_dir}; pass main_file explicitly"
    )


# ---------------------------------------------------------------------------
# Step 1.5: cheap typography simplification -- forced line breaks and
# non-breaking spaces read fine as plain newline/space in extracted prose.
# ---------------------------------------------------------------------------

_LINEBREAK_RE = re.compile(r"\\\\")


def _simplify_typography(text: str) -> str:
    text = _LINEBREAK_RE.sub("\n", text)
    return text.replace("~", " ")


# ---------------------------------------------------------------------------
# Step 2-3: \newcommand-family extraction + best-effort expansion.
# ---------------------------------------------------------------------------

# Matches everything up to (but not including) the body's opening brace:
# the command name, its optional [nargs] and optional [default], for
# \newcommand/\renewcommand/\providecommand, or the operator name for
# \DeclareMathOperator. The body itself is never matched by regex -- see
# `_match_brace_group` below for why.
_MACRO_DEF_START_RE = re.compile(
    r"\\(?:new|renew|provide)command\*?\s*\{?\\(?P<name1>[A-Za-z]+)\}?"
    r"\s*(?:\[(?P<nargs>\d+)\])?\s*(?:\[[^\]]*\])?\s*"
    r"|\\DeclareMathOperator\*?\s*\{?\\(?P<name2>[A-Za-z]+)\}?\s*"
)

# A small, fixed allowlist of standard single-argument formatting/math
# macros that are unwrapped to their bare argument regardless of whether
# the source ever \newcommand's them -- they only ever add visual noise to
# plain text, and unwrapping them is exactly the "single required arg, no
# nested braces" substitution already needed for user macros, so it costs
# nothing extra to include a fixed set of built-ins. This is also how
# "keep inline math readable" is satisfied for math-mode formatting
# commands without needing to isolate `$...$` spans specially. ``ensuremath``
# is a pure passthrough (it only forces math mode for its argument, which
# this module already treats as plain text); ``tiny``/``small`` are real
# LaTeX declarations rather than argument-taking commands, but papers
# routinely write them as ``\tiny{...}`` to scope a font change to one
# group, and that shape is indistinguishable from a unary macro here.
_BUILTIN_UNARY_MACROS: dict[str, str] = {
    "emph": "#1",
    "textbf": "#1",
    "textit": "#1",
    "underline": "#1",
    "texttt": "#1",
    "textsc": "#1",
    "text": "#1",
    "mathbf": "#1",
    "mathrm": "#1",
    "mathcal": "#1",
    "mathbb": "#1",
    "mathsf": "#1",
    "mathtt": "#1",
    "operatorname": "#1",
    "ensuremath": "#1",
    "tiny": "#1",
    "small": "#1",
}


def _match_brace_group(text: str, open_pos: int) -> int | None:
    """Return the index just past the ``}`` that closes ``text[open_pos]``.

    ``text[open_pos]`` must be ``{``. Counts nesting depth instead of the
    single-level ``[^{}]*`` trick used elsewhere in this module for
    *invocation* arguments -- a macro *definition*'s replacement text is
    exactly where a real paper nests a builtin inside a user macro (e.g.
    ``\\newcommand{\\docs}{\\ensuremath{\\mathbf{z}}}``), and unlike an
    invocation that cannot be substituted (which is simply left in place,
    unmodified), a definition whose closing brace cannot be located has
    nowhere safe to stop -- a one-level scan would either truncate the
    body or run on past it into whatever text follows. ``\\{`` and ``\\}``
    are literal escaped braces, not grouping, and are skipped as a pair
    without changing depth. Returns ``None`` if the braces never balance
    before the end of ``text``, which tells the caller to leave the whole
    construct untouched rather than guess where it should have ended.
    """
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] in "{}":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _extract_macros(text: str) -> tuple[str, dict[str, tuple[int, str]]]:
    """Strip every macro-definition command from ``text``, wherever it occurs.

    Recognises ``\\newcommand``/``\\renewcommand``/``\\providecommand``/
    ``\\DeclareMathOperator``. Returns the stripped text and a table of
    ``name -> (arity, body)`` for the macros this module can actually
    expand (arity 0 or 1) -- ``\\DeclareMathOperator`` has no ``[nargs]`` at
    all and is registered the same way a 0-arity ``\\newcommand`` would be.
    A macro definition with 2+ required args or an optional default value
    for its first argument is still removed from the text -- a raw
    definition line is not prose either way -- but is not added to the
    table, so its invocations are left untouched (see the module
    docstring).

    Runs on the whole document, before the ``\\begin{document}`` split
    (see ``parse_latex_source``). A definition that fails to strip in the
    preamble is invisible either way -- everything before
    ``\\begin{document}`` is discarded next regardless -- but the same
    failure *after* ``\\begin{document}`` has no such safety net: the raw
    ``\\newcommand{...}{...}`` text would survive into the parsed body
    verbatim. That asymmetry, not a difference in where this function
    looks, is why a body-only definition used to leak: the old single-level
    brace regex simply failed to match (and therefore failed to strip) any
    definition whose replacement text nested two levels deep, and it could
    appear on either side of ``\\begin{document}`` by simple accident of
    where an author introduced a helper macro.
    """
    macros: dict[str, tuple[int, str]] = {}
    out: list[str] = []
    cursor = 0

    for m in _MACRO_DEF_START_RE.finditer(text):
        if m.start() < cursor:
            continue  # inside a definition body already consumed above
        name = m.group("name1") or m.group("name2")
        body_start = m.end()
        if body_start >= len(text) or text[body_start] != "{":
            continue  # no body brace where one is expected -- leave in place
        body_end = _match_brace_group(text, body_start)
        if body_end is None:
            continue  # braces never balance -- leave in place

        out.append(text[cursor : m.start()])
        cursor = body_end

        nargs_s = m.group("nargs")
        arity = int(nargs_s) if nargs_s is not None else 0
        if arity in (0, 1):
            macros[name] = (arity, text[body_start + 1 : body_end - 1])

    out.append(text[cursor:])
    return "".join(out), macros


def _expand_macro_once(text: str, name: str, arity: int, body: str) -> str:
    if arity == 0:
        pattern = re.compile(r"\\" + re.escape(name) + r"(?![A-Za-z])")
        return pattern.sub(lambda m: body, text)
    pattern = re.compile(r"\\" + re.escape(name) + r"(?![A-Za-z])\s*\{([^{}]*)\}")
    return pattern.sub(lambda m: body.replace("#1", m.group(1)), text)


def _expand_macros(text: str, macros: dict[str, tuple[int, str]], max_passes: int = 5) -> str:
    """Best-effort iterative macro expansion.

    Several small passes (rather than one) let a macro whose body invokes
    another simple macro resolve fully, and let a single-arg macro applied
    to an argument that itself contains another (now-simplified) macro
    invocation resolve on a later pass, once the inner one has already
    been flattened to something brace-free. Stops as soon as a pass makes
    no change (a self-referential definition converges immediately rather
    than looping), and is capped at ``max_passes`` regardless, so a
    mutually-recursive pair of macros cannot hang the parser -- it is
    simply left partially expanded, which is the documented trade-off.
    """
    all_macros: dict[str, tuple[int, str]] = {name: (1, body) for name, body in _BUILTIN_UNARY_MACROS.items()}
    all_macros.update(macros)
    for _ in range(max_passes):
        before = text
        for name, (arity, body) in all_macros.items():
            text = _expand_macro_once(text, name, arity, body)
        if text == before:
            break
    return text


# ---------------------------------------------------------------------------
# Step 3.5: stray empty `{}` left behind once a zero-argument macro has been
# expanded away.
# ---------------------------------------------------------------------------

_STRAY_EMPTY_BRACES_RE = re.compile(r"(?<=[\w}])\{\}")


def _strip_stray_empty_braces(text: str) -> str:
    """Remove an empty ``{}`` immediately after a word character or ``}``.

    A common LaTeX idiom writes a zero-argument macro invocation as
    ``\\name{}`` -- the empty group stops the macro from swallowing the
    space that follows it at the *source* level, and has no effect on
    typeset output. Expanding ``\\name`` does not consume that trailing
    ``{}``: arity-0 substitution (see ``_expand_macro_once`` above) only
    ever matches the bare command token, so the empty group is a separate,
    untouched piece of text either side of it. ``\\RAGSequence{}`` -- a
    real example, after name substitution -- becomes ``RAG-Sequence{}``:
    noise with no retrievable meaning.

    Deliberately blunt (it cannot tell a decorative empty group from a
    structural command's own, still-meaningful, still-empty argument), so
    ``parse_latex_source`` must call this *last* -- after every command
    that itself expects a ``{...}`` argument (headings, ``\\title``,
    captions, citations, refs, labels) has already consumed its own
    braces. Called any earlier, a genuinely empty but structurally
    required argument -- ``\\section{}`` (an anonymous heading),
    ``\\title{}`` -- would have its braces stripped *before* the command
    that expects them gets to run, turning a recognised structural command
    into unrecognised literal text. Once every such command has already
    fired, the only ``{}`` left adjacent to a word character really is
    leftover macro noise (or, at worst, a stray brace after some
    intentionally-unexpanded unknown command like ``\\LaTeX{}``, which is
    exactly the same noise).
    """
    return _STRAY_EMPTY_BRACES_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 4: \title extraction.
# ---------------------------------------------------------------------------

_TITLE_RE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_LABEL_RE = re.compile(r"\\label\{[^}]*\}")
_CITE_RE = re.compile(r"\\(?:cite[a-zA-Z]*|parencite|textcite|autocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}")
_REF_RE = re.compile(r"\\(?:eqref|pageref|autoref|[Cc]ref|ref)\{([^}]*)\}")


def _clean_fragment(s: str) -> str:
    """Local cleanup for text pulled out of the stream early (just the title).

    Everything else stays in the main stream and gets cleaned by the
    regular cite/ref/label passes below; the title is extracted before
    those run (titles live in the preamble, which is discarded before this
    module ever looks at citations), so it needs its own miniature version
    of the same cleanup.
    """
    s = _LABEL_RE.sub("", s)
    s = _CITE_RE.sub(lambda m: "[" + ", ".join(k.strip() for k in m.group(1).split(",") if k.strip()) + "]", s)
    s = _REF_RE.sub(lambda m: f"[{m.group(1)}]", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Step 5: restrict to \begin{document}...\end{document}.
# ---------------------------------------------------------------------------

_DOC_RE = re.compile(r"\\begin\{document\}(.*?)(?:\\end\{document\}|\Z)", re.DOTALL)


# ---------------------------------------------------------------------------
# Step 6: bibliography -- stripped from the body, position recorded.
# ---------------------------------------------------------------------------

_BIBLIOGRAPHY_RE = re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL)


def _process_bibliography(text: str) -> str:
    """Strip the bibliography, leaving a position marker where it started.

    The marker is prefixed with ``\\n\\n`` (rather than dropped in place of
    the matched text with nothing around it) for the same reason headings
    and the abstract marker are: whatever raw source whitespace happened to
    sit immediately before ``\\begin{thebibliography}`` must land on its
    *own* line, terminated by a real newline, so ``normalize_text``'s
    per-line ``rstrip`` can actually reach it. A marker character is not
    whitespace, so without this padding that indentation would sit
    "protected" behind the marker through the one normalisation pass and
    survive as unstripped trailing whitespace once the marker is later
    removed -- silently breaking the "normalize_text is a fixed point"
    invariant on the final text.
    """
    return _BIBLIOGRAPHY_RE.sub(lambda m: f"\n\n{_marker('__BIBLIOGRAPHY__')}", text, count=1)


# ---------------------------------------------------------------------------
# Step 7: figure/table floats -> caption text, tagged by float type.
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}", re.DOTALL)
_CAPTION_RE = re.compile(r"\\caption\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")


def _process_floats(text: str) -> str:
    """Replace each figure/table environment with just its caption, tagged.

    Graphics includes and raw tabular markup are not prose and are
    dropped; the caption is exactly the high-value retrieval target the
    task calls out, so it is kept as its own paragraph rather than left
    buried inside float markup. A float with no ``\\caption`` at all is
    dropped entirely -- a tag with no text behind it is worse than no
    mention.
    """

    def _repl(m: re.Match[str]) -> str:
        kind = "Figure" if m.group(1).startswith("figure") else "Table"
        cap = _CAPTION_RE.search(m.group(2))
        if not cap:
            return ""
        caption_text = " ".join(cap.group(1).split())
        return f"\n\n{kind}: {caption_text}\n\n"

    return _FLOAT_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# Step 8-10: citations, refs, labels.
# ---------------------------------------------------------------------------


def _process_citations(text: str) -> tuple[str, list[str]]:
    """Replace \\cite-family commands with a bracketed key list.

    Keys are kept (deduplicated, first-seen order) as
    ``ParsedDocument.citations`` -- this is the free citation graph
    referenced in the task: no resolution to actual bibliography entries
    is attempted here, just the raw key strings.
    """
    seen: dict[str, None] = {}

    def _repl(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        for k in keys:
            seen.setdefault(k, None)
        return "[" + ", ".join(keys) + "]"

    new_text = _CITE_RE.sub(_repl, text)
    return new_text, list(seen)


def _process_refs(text: str) -> str:
    return _REF_RE.sub(lambda m: f"[{m.group(1)}]", text)


def _strip_labels(text: str) -> str:
    return _LABEL_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 11: display math -> a placeholder paragraph that cannot fragment a
# sentence (it is always padded onto its own line).
# ---------------------------------------------------------------------------

_DISPLAY_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}.*?\\end\{\1\}",
    re.DOTALL,
)
_DISPLAY_BRACKET_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_DISPLAY_DOLLAR_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)


def _process_display_math(text: str) -> str:
    text = _DISPLAY_ENV_RE.sub("\n\n[DISPLAY_MATH]\n\n", text)
    text = _DISPLAY_BRACKET_RE.sub("\n\n[DISPLAY_MATH]\n\n", text)
    text = _DISPLAY_DOLLAR_RE.sub("\n\n[DISPLAY_MATH]\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Step 12: abstract -- kept as prose, but marked as its own Section.
# ---------------------------------------------------------------------------

_ABSTRACT_ENV_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)
_ABSTRACT_CMD_RE = re.compile(r"\\abstract\s*\{((?:[^{}]|\{[^{}]*\})*)\}")


def _process_abstract(text: str) -> tuple[str, str]:
    """Extract the abstract, and mark its position for the section pass.

    Runs after citations/refs/labels/display-math, so by this point the
    abstract text sitting in the stream is already clean of that noise --
    it was never removed from the stream, only observed, so it received
    every one of those passes for free.
    """
    m = _ABSTRACT_ENV_RE.search(text)
    if m is None:
        m = _ABSTRACT_CMD_RE.search(text)
        pattern = _ABSTRACT_CMD_RE
    else:
        pattern = _ABSTRACT_ENV_RE

    if m is None:
        return text, ""

    abstract = m.group(1).strip()
    new_text = pattern.sub(
        lambda mm: f"\n\n{_marker('Abstract')}Abstract\n{mm.group(1).strip()}\n\n", text, count=1
    )
    return new_text, abstract


# ---------------------------------------------------------------------------
# Step 13: section hierarchy -> heading markers with computed numbering.
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"\\(?P<cmd>section|subsection|subsubsection|paragraph)(?P<star>\*)?"
    r"\s*(?:\[[^\]]*\])?\s*\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\}"
)
_LEVEL_OF = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
_MAX_NUMBERED_LEVEL = 3


def _process_sections(text: str) -> str:
    counters = [0] * _MAX_NUMBERED_LEVEL
    path_stack: list[tuple[int, str]] = []

    def _repl(m: re.Match[str]) -> str:
        level = _LEVEL_OF[m.group("cmd")]
        starred = m.group("star") is not None
        title = " ".join(m.group("title").split())
        display, section_path = advance_heading(level, title, not starred, counters, path_stack)
        return f"\n\n{_marker(section_path)}{display}\n\n"

    return _HEADING_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# Step 15: marker extraction. Runs after normalize_text() -- this is the
# single place that converts marker positions into final char offsets, and
# it is the only correct place to do so (see selfrag.ingest.__init__).
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(_MARK + r"(.*?)" + _MARK)


def _extract_markers(text: str) -> tuple[str, list[Section], int | None]:
    sections: list[Section] = []
    bibliography_char_start: int | None = None
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

    for i, (payload, start) in enumerate(pending):
        end = pending[i + 1][1] if i + 1 < len(pending) else len(cleaned)
        if payload == "__BIBLIOGRAPHY__":
            bibliography_char_start = start
            continue
        if end > start:
            sections.append(Section(section_path=payload, char_start=start, char_end=end))

    return cleaned, sections, bibliography_char_start


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def parse_latex_source(source_dir: str | Path, main_file: str | Path | None = None) -> ParsedDocument:
    """Parse an extracted LaTeX source tree into a :class:`ParsedDocument`.

    ``source_dir`` is the root of an already-extracted arXiv e-print
    (whatever unpacked the ``.tar.gz`` is someone else's job -- this
    module only ever reads files that already exist on disk). ``main_file``
    is relative to ``source_dir``; when omitted, the module looks for
    exactly one ``.tex`` file containing ``\\documentclass`` (or the only
    ``.tex`` file, if there is just one) and raises
    :class:`LatexSourceError` if that is ambiguous.

    Raises:
        LatexSourceError: no usable main file, or an ``\\input``/``\\include``
            target does not exist.
        LatexCycleError: ``\\input``/``\\include`` forms a cycle.
        LatexParseError: the private-use marker character this module uses
            for offset bookkeeping was found in the source itself.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise LatexSourceError(f"source_dir does not exist or is not a directory: {source_dir}")

    if main_file is not None:
        main_path = (source_dir / main_file).resolve()
        if not main_path.is_file():
            raise LatexSourceError(f"main_file does not exist: {main_path}")
    else:
        main_path = _find_main_file(source_dir)

    text = _load_and_expand(main_path, source_dir, ())
    text = _simplify_typography(text)

    text, macros = _extract_macros(text)
    text = _expand_macros(text, macros)

    title_match = _TITLE_RE.search(text)
    title = _clean_fragment(title_match.group(1)) if title_match else ""
    if title_match:
        text = _TITLE_RE.sub("", text, count=1)

    doc_match = _DOC_RE.search(text)
    if doc_match:
        text = doc_match.group(1)

    text = _process_bibliography(text)
    text = _process_floats(text)
    text, citations = _process_citations(text)
    text = _process_refs(text)
    text = _strip_labels(text)
    text = _process_display_math(text)
    text, abstract = _process_abstract(text)
    text = _process_sections(text)

    # Runs last, after every structural command (headings, captions,
    # citations, refs, labels, the title) has already consumed its own
    # `{...}` argument -- see `_strip_stray_empty_braces`'s docstring for
    # why running this any earlier is unsafe: a genuinely meaningful empty
    # argument like `\section{}` (an anonymous heading) or `\title{}`
    # would otherwise be stripped of its braces *before* the command that
    # expects them ever gets to run, silently turning a recognised
    # structural command into unrecognised literal text.
    text = _strip_stray_empty_braces(text)

    normalized = normalize_text(text)
    cleaned, sections, bibliography_char_start = _extract_markers(normalized)

    return ParsedDocument(
        text=cleaned,
        parser_id=PARSER_ID,
        title=title,
        abstract=abstract,
        sections=sections,
        citations=citations,
        bibliography_char_start=bibliography_char_start,
    )

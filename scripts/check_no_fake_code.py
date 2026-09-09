#!/usr/bin/env python
"""The "no fake code" CI gate.

Fails (exit 1) if anything under ``src/`` looks like a stub rather than a
finished implementation:

* ``raise NotImplementedError`` (or any bare reference to the name).
* ``TODO`` / ``FIXME`` / ``XXX`` markers -- but only as actual *comments*,
  never as a false hit inside a string literal that happens to contain one
  of those words. Comments and strings are different token kinds, so this
  is checked with :mod:`tokenize`, and everything else below with
  :mod:`ast`, rather than by pattern-matching the raw file text: a regex
  over raw source cannot tell a comment from a docstring from a string
  literal, and this checker exists specifically to avoid the false
  positives that confusion produces.
* A bare ``pass`` as the *sole* body of a function or class, unless that
  function is an ABC/Protocol method or the class is a declared exception
  class -- both of which are legitimate, finished code, not stubs.
* A bare ``...`` (Ellipsis) as the sole body of a function, unless it is a
  ``@typing.overload`` signature or a method of a ``Protocol`` class --
  again, both real, intentional patterns.
* Any import of ``unittest.mock`` or the standalone ``mock`` package inside
  ``src/`` -- mocks belong in tests, never in shipped implementation code.

Escape hatch: a violation on a line that also carries a trailing comment of
the form ``# noqa: fake-code: <reason>`` is suppressed, provided a
non-empty reason follows -- a bare ``# noqa: fake-code`` with no reason does
not suppress anything, because the whole point of the hatch is a recorded
justification, not a silencer. The number of such escapes in use is always
reported, whether or not the run finds other violations, so escapes stay
visible instead of accumulating unnoticed.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

_BANNED_COMMENT_MARKERS = ("TODO", "FIXME", "XXX")
_BANNED_MOCK_MODULES = frozenset({"unittest.mock", "mock"})

_NOQA_PREFIX = "noqa: fake-code"


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}"


def _noqa_reason(source_line: str) -> str | None:
    """Return the justification text if ``source_line`` carries a valid escape.

    A valid escape looks like ``# noqa: fake-code: <non-empty reason>``
    (``-`` or ``,`` are also accepted as the separator before the reason).
    A bare ``# noqa: fake-code`` with nothing after it is *not* valid: the
    hatch requires a recorded reason, not just a marker.
    """
    idx = source_line.find("noqa: fake-code")
    if idx == -1:
        idx = source_line.find("noqa:fake-code")
        if idx == -1:
            return None
        idx += len("noqa:fake-code")
    else:
        idx += len("noqa: fake-code")

    rest = source_line[idx:].lstrip()
    rest = rest.lstrip(":-,").strip()
    return rest or None


def _line_text(lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ""


def _base_names(class_def: ast.ClassDef) -> set[str]:
    """Textual names of a class's bases (best-effort, no import resolution)."""
    names: set[str] = set()
    for base in class_def.bases:
        node: ast.expr = base
        if isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _is_exception_class(class_def: ast.ClassDef) -> bool:
    if class_def.name.endswith(("Error", "Exception")):
        return True
    return any(name.endswith(("Error", "Exception")) for name in _base_names(class_def))


def _is_abc_or_protocol_class(class_def: ast.ClassDef) -> bool:
    return bool(_base_names(class_def) & {"ABC", "Protocol"})


def _has_decorator(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    for dec in func.decorator_list:
        node = dec
        if isinstance(node, ast.Call):
            node = node.func
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _sole_body_is(body: list[ast.stmt], kind: type[ast.AST]) -> bool:
    if len(body) != 1:
        return False
    stmt = body[0]
    if kind is ast.Pass:
        return isinstance(stmt, ast.Pass)
    if kind is ast.Constant:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        )
    return False


class _Checker(ast.NodeVisitor):
    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.violations: list[Violation] = []
        self._class_stack: list[ast.ClassDef] = []

    def _report(self, lineno: int, reason: str) -> None:
        if _noqa_reason(_line_text(self.lines, lineno)) is not None:
            return  # valid escape hatch: suppressed, but still counted globally
        self.violations.append(Violation(self.path, lineno, reason))

    # -- NotImplementedError --------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "NotImplementedError":
            self._report(node.lineno, "reference to NotImplementedError (stub code is not allowed)")
        self.generic_visit(node)

    # -- imports of mock modules ------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in _BANNED_MOCK_MODULES:
                self._report(node.lineno, f"import of {alias.name!r} (mocks do not belong in src/)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module in _BANNED_MOCK_MODULES:
            self._report(node.lineno, f"import from {module!r} (mocks do not belong in src/)")
        elif module == "unittest" and any(alias.name == "mock" for alias in node.names):
            self._report(node.lineno, "import of 'unittest.mock' (mocks do not belong in src/)")
        self.generic_visit(node)

    # -- class-scoped pass/ellipsis checks --------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if _sole_body_is(node.body, ast.Pass) and not _is_exception_class(node):
            self._report(
                node.body[0].lineno,
                f"class {node.name!r} has a bare 'pass' body (not a declared exception class)",
            )
        self._class_stack.append(node)
        self.generic_visit(node)
        self._class_stack.pop()

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self._class_stack[-1] if self._class_stack else None
        is_method_of_abc_or_protocol = parent is not None and _is_abc_or_protocol_class(parent)
        is_overload = _has_decorator(node, "overload")
        is_abstract = _has_decorator(node, "abstractmethod")

        if _sole_body_is(node.body, ast.Pass):
            if not (is_method_of_abc_or_protocol or is_abstract):
                self._report(
                    node.body[0].lineno,
                    f"function {node.name!r} has a bare 'pass' body "
                    "(not an ABC/Protocol method)",
                )
        elif _sole_body_is(node.body, ast.Constant):
            allowed = is_overload or (
                parent is not None and _base_names(parent) & {"Protocol"}
            )
            if not allowed:
                self._report(
                    node.body[0].lineno,
                    f"function {node.name!r} has a bare '...' body "
                    "outside a Protocol or @overload definition",
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)


def _check_comments(path: Path, source: str, lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            comment = tok.string
            for marker in _BANNED_COMMENT_MARKERS:
                if marker in comment:
                    lineno = tok.start[0]
                    if _noqa_reason(_line_text(lines, lineno)) is not None:
                        continue
                    violations.append(
                        Violation(path, lineno, f"'{marker}' marker in a comment (finish or file it, do not stub)")
                    )
                    break
    except tokenize.TokenError:
        pass
    return violations


def _count_escapes(lines: list[str]) -> int:
    return sum(1 for line in lines if _noqa_reason(line) is not None)


def check_file(path: Path) -> tuple[list[Violation], int]:
    """Return (violations, escape_count) for one source file."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Violation(path, exc.lineno or 0, f"file failed to parse: {exc}")], 0

    checker = _Checker(path, lines)
    checker.visit(tree)
    violations = list(checker.violations)
    violations.extend(_check_comments(path, source, lines))
    violations.sort(key=lambda v: v.line)
    return violations, _count_escapes(lines)


def iter_source_files(src_root: Path) -> list[Path]:
    return sorted(
        p
        for p in src_root.rglob("*.py")
        if "tests" not in p.relative_to(src_root).parts and "__pycache__" not in p.parts
    )


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = Path(argv[1]) if len(argv) > 1 else repo_root / "src"

    if not src_root.is_dir():
        print(f"no such directory: {src_root}", file=sys.stderr)
        return 1

    all_violations: list[Violation] = []
    total_escapes = 0
    for path in iter_source_files(src_root):
        violations, escapes = check_file(path)
        all_violations.extend(violations)
        total_escapes += escapes

    for v in all_violations:
        print(str(v))

    print(f"\n{len(all_violations)} violation(s), {total_escapes} '# noqa: fake-code' escape(s) in use.")
    return 1 if all_violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

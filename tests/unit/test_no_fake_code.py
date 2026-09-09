"""Tests for scripts/check_no_fake_code.py -- the CI gate that checks *this*
codebase for stubs. It is not importable as a package (it lives under
scripts/, not src/), so it is loaded here by file path.

Every violation class the script claims to catch gets a positive test (it
is actually caught) and, where a false positive is plausible, a negative
test (a legitimate pattern that must NOT be flagged): ABC/Protocol methods,
@overload signatures, declared exception classes, and TODO/mock appearing
inside string literals rather than as real comments/imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_no_fake_code.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_no_fake_code", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_module()


def _write(tmp_path: Path, source: str, name: str = "sample.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _reasons(tmp_path: Path, source: str) -> list[str]:
    violations, _ = checker.check_file(_write(tmp_path, source))
    return [v.reason for v in violations]


class TestNotImplementedError:
    def test_bare_raise_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "def f():\n    raise NotImplementedError\n")
        assert any("NotImplementedError" in r for r in reasons)

    def test_raise_with_message_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "def f():\n    raise NotImplementedError('later')\n")
        assert any("NotImplementedError" in r for r in reasons)

    def test_mention_inside_string_literal_not_flagged(self, tmp_path):
        source = 'MSG = "this mentions NotImplementedError but is just a string"\n'
        reasons = _reasons(tmp_path, source)
        assert reasons == []


class TestTodoFixmeXxx:
    @pytest.mark.parametrize("marker", ["TODO", "FIXME", "XXX"])
    def test_marker_in_comment_caught(self, tmp_path, marker):
        reasons = _reasons(tmp_path, f"x = 1  # {marker}: finish this\n")
        assert any(marker in r for r in reasons)

    @pytest.mark.parametrize("marker", ["TODO", "FIXME", "XXX"])
    def test_marker_inside_string_literal_not_flagged(self, tmp_path, marker):
        source = f'MSG = "contains the word {marker} but is not a comment"\n'
        reasons = _reasons(tmp_path, source)
        assert reasons == []

    def test_marker_inside_docstring_not_flagged(self, tmp_path):
        """Docstrings are string literals in the AST/tokenizer sense, not
        comments -- the checker must not conflate the two."""
        source = '"""Module docstring mentioning TODO for illustration."""\n'
        reasons = _reasons(tmp_path, source)
        assert reasons == []


class TestBarePass:
    def test_bare_pass_function_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "def f():\n    pass\n")
        assert any("bare 'pass'" in r for r in reasons)

    def test_bare_pass_class_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "class Placeholder:\n    pass\n")
        assert any("bare 'pass'" in r for r in reasons)

    def test_abstractmethod_with_pass_not_flagged(self, tmp_path):
        source = (
            "from abc import ABC, abstractmethod\n\n"
            "class Base(ABC):\n"
            "    @abstractmethod\n"
            "    def do_it(self) -> None:\n"
            "        pass\n"
        )
        assert _reasons(tmp_path, source) == []

    def test_protocol_method_with_pass_not_flagged(self, tmp_path):
        source = (
            "from typing import Protocol\n\n"
            "class Proto(Protocol):\n"
            "    def do_it(self) -> None:\n"
            "        pass\n"
        )
        assert _reasons(tmp_path, source) == []

    def test_declared_exception_class_with_pass_not_flagged(self, tmp_path):
        source = "class MyThingError(Exception):\n    pass\n"
        assert _reasons(tmp_path, source) == []

    def test_plain_subclass_with_pass_is_still_flagged(self, tmp_path):
        """Not every empty class body is exempt -- only ABC/Protocol
        methods and exception classes are."""
        source = "class MyBase:\n    pass\n\nclass Child(MyBase):\n    pass\n"
        reasons = _reasons(tmp_path, source)
        assert len(reasons) == 2


class TestBareEllipsis:
    def test_bare_ellipsis_function_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "def f():\n    ...\n")
        assert any("'...'" in r for r in reasons)

    def test_overload_with_ellipsis_not_flagged(self, tmp_path):
        source = (
            "from typing import overload\n\n"
            "@overload\n"
            "def f(x: int) -> int: ...\n"
            "@overload\n"
            "def f(x: str) -> str: ...\n"
            "def f(x):\n"
            "    return x\n"
        )
        assert _reasons(tmp_path, source) == []

    def test_protocol_method_with_ellipsis_not_flagged(self, tmp_path):
        source = (
            "from typing import Protocol\n\n"
            "class Proto(Protocol):\n"
            "    def f(self) -> None: ...\n"
        )
        assert _reasons(tmp_path, source) == []

    def test_ellipsis_used_as_a_value_not_flagged(self, tmp_path):
        """Ellipsis as an actual expression value (e.g. a slice or a
        sentinel default), not as a whole function body, is legitimate."""
        source = "def f(x=...):\n    return x is ...\n"
        assert _reasons(tmp_path, source) == []


class TestMockImports:
    def test_import_mock_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "import mock\n")
        assert any("mock" in r for r in reasons)

    def test_import_unittest_mock_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "import unittest.mock\n")
        assert any("mock" in r for r in reasons)

    def test_from_unittest_mock_import_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "from unittest.mock import MagicMock\n")
        assert any("mock" in r for r in reasons)

    def test_from_unittest_import_mock_caught(self, tmp_path):
        reasons = _reasons(tmp_path, "from unittest import mock\n")
        assert any("mock" in r for r in reasons)

    def test_from_unittest_import_testcase_not_flagged(self, tmp_path):
        """Importing other things from unittest (not `mock`) is fine."""
        reasons = _reasons(tmp_path, "from unittest import TestCase\n")
        assert reasons == []

    def test_word_mock_in_string_not_flagged(self, tmp_path):
        reasons = _reasons(tmp_path, 'x = "please mock this in your head"\n')
        assert reasons == []


class TestNoqaEscapeHatch:
    def test_escape_with_justification_suppresses_violation(self, tmp_path):
        source = "def f():\n    pass  # noqa: fake-code: tracked in TICKET-1\n"
        violations, escapes = checker.check_file(_write(tmp_path, source))
        assert violations == []
        assert escapes == 1

    def test_escape_without_justification_does_not_suppress(self, tmp_path):
        source = "def f():\n    pass  # noqa: fake-code\n"
        violations, escapes = checker.check_file(_write(tmp_path, source))
        assert len(violations) == 1
        assert escapes == 0

    def test_escape_count_reported_even_without_violations(self, tmp_path):
        source = 'x = 1  # noqa: fake-code: not actually a violation on this line\n'
        violations, escapes = checker.check_file(_write(tmp_path, source))
        assert violations == []
        assert escapes == 1


class TestIterSourceFiles:
    def test_excludes_tests_directory(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").write_text("import mock\n", encoding="utf-8")
        (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
        files = checker.iter_source_files(tmp_path)
        assert tmp_path / "real.py" in files
        assert all("tests" not in f.relative_to(tmp_path).parts for f in files)

    def test_excludes_pycache(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "x.pyc.py").write_text("x = 1\n", encoding="utf-8")
        files = checker.iter_source_files(tmp_path)
        assert files == []


class TestMainEntryPoint:
    def test_exits_zero_on_clean_directory(self, tmp_path):
        (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
        assert checker.main([str(_SCRIPT_PATH), str(tmp_path)]) == 0

    def test_exits_nonzero_on_violation(self, tmp_path):
        (tmp_path / "dirty.py").write_text("def f():\n    pass\n", encoding="utf-8")
        assert checker.main([str(_SCRIPT_PATH), str(tmp_path)]) == 1

    def test_this_projects_own_src_is_clean(self):
        """The gate has to pass against the real codebase it guards, not
        just synthetic fixtures."""
        repo_root = _SCRIPT_PATH.parent.parent
        assert checker.main([str(_SCRIPT_PATH), str(repo_root / "src")]) == 0

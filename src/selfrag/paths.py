"""Central, typed path resolution.

Every other module that touches disk imports this instead of building paths
itself. That is the whole point: relocating the data directory (a bigger
disk, a scratch volume, a CI tmp dir) is then a single environment-variable
change instead of a grep-and-edit across the codebase, and no module can
silently drift to a hardcoded path that only works on one machine.

``SELFRAG_DATA_DIR`` is read fresh from the environment on every call rather
than cached at import time. That makes the module trivially testable
(``monkeypatch.setenv`` before a call, no import-order tricks) and lets a
single process address more than one data root if it ever needs to (e.g. a
test harness pointing at a tmp dir while the CLI process it wraps is unaware).
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "SELFRAG_DATA_DIR"
_DEFAULT_DATA_DIR = "./data"
_LEDGER_FILENAME = "ledger.duckdb"


def _ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) on demand, then return it.

    Directories are created lazily, at the point something asks to use them,
    rather than once at startup -- a fresh checkout or a fresh CI runner
    should not need a separate "init data dirs" step before anything works.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Root of all local data, from ``SELFRAG_DATA_DIR`` (default ``./data``).

    Resolved to an absolute path so downstream code (DuckDB, pyarrow) never
    has to reason about the current working directory.
    """
    raw = os.environ.get(_ENV_VAR, _DEFAULT_DATA_DIR)
    return _ensure_dir(Path(raw).expanduser().resolve())


def raw_dir() -> Path:
    """Untouched, as-downloaded source documents (PDFs, source HTML, ...)."""
    return _ensure_dir(data_dir() / "raw")


def canonical_dir() -> Path:
    """Frozen, normalised document text -- what ``doc_text_sha256`` hashes."""
    return _ensure_dir(data_dir() / "canonical")


def cache_dir() -> Path:
    """Root of all derived, safely-recomputable caches."""
    return _ensure_dir(data_dir() / "cache")


def embeddings_cache_dir() -> Path:
    """Embedding-vector cache, keyed by ``raw_text_sha256`` + embedder id.

    Kept separate from other caches because it is by far the largest and the
    most expensive to regenerate -- it is the one cache directory operators
    will want to back up or move independently of the rest.
    """
    return _ensure_dir(cache_dir() / "embeddings")


def indexes_dir() -> Path:
    """Built retrieval indexes (dense ANN, sparse, fused)."""
    return _ensure_dir(data_dir() / "indexes")


def eval_dir() -> Path:
    """Root of evaluation assets: qrels and query sets."""
    return _ensure_dir(data_dir() / "eval")


def qrels_dir() -> Path:
    """Relevance judgements, versioned by ``Qrel.qrels_version``."""
    return _ensure_dir(eval_dir() / "qrels")


def queries_dir() -> Path:
    """Query sets, partitioned by ``Split`` (dev/test)."""
    return _ensure_dir(eval_dir() / "queries")


def runs_dir() -> Path:
    """Per-run artifacts (retrieved-doc dumps, generated answers, ...).

    Distinct from the ledger: this holds bulky per-run files on disk, while
    the ledger holds the queryable numbers extracted from them.
    """
    return _ensure_dir(data_dir() / "runs")


def ledger_path() -> Path:
    """Path to the DuckDB experiment-ledger file.

    Not a directory, so it is not created here -- only its parent
    (``data_dir()``) is guaranteed to exist. ``duckdb.connect`` creates the
    file itself on first connection.
    """
    return data_dir() / _LEDGER_FILENAME

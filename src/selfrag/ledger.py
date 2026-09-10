"""The experiment ledger: the project's one machine-readable memory of runs.

Prose (``MEMORY.md``, ADRs under ``decisions/``) records *why* a decision was
made. This module records *what happened, numerically*, and is the only
place that is allowed to: every claim of the form "config A beat config B" has
to be traceable to a row here, never to a number typed into a markdown table
by hand, because hand-typed numbers cannot be re-aggregated, re-tested for
significance, or checked against the constraints the run was supposed to
respect.

Two design decisions recur throughout this module and are worth stating once:

* Per-query metric values are persisted in ``run_metrics``, never only their
  mean. A paired significance test (bootstrap, permutation, Wilcoxon --
  whatever ``selfrag.eval.stats`` uses) needs the *same query* answered by
  both runs being compared; an aggregate mean cannot be paired with anything.
  Throwing away per-query values at write time would make every later
  comparison unpaired and therefore under-powered or simply wrong.
* ``record_run`` is idempotent on ``run_id`` and raises on a conflicting
  manifest for the same ``run_id``. ``run_id`` is defined (see
  ``RunManifest.run_id``) as a hash of the *entire* manifest, so under normal
  use two different manifests cannot collide on it. The guard exists anyway,
  precisely because it should never fire: if it ever does, that is proof the
  hash failed to cover something that actually varies between runs, which is
  a bug in the manifest schema, not a bug in the ledger. The check has to be
  fast to reach, so ``record_run`` accepts an optional explicit ``run_id``
  override for callers that computed it themselves.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import orjson
import pyarrow as pa

from selfrag.schema import Chunk, Document, Namespace, RunManifest

_CURRENT_SCHEMA_VERSION = 2

# Explicit Arrow schemas for the batched upserts below. Typed exactly to the
# DuckDB column types (see `_create_schema`) rather than left to
# `pa.Table.from_pylist`'s inference, because an all-null column (e.g. every
# document in one batch has `tombstoned_at=None`) infers as Arrow's `null`
# type, which does not implicitly cast into a DuckDB TIMESTAMP column on
# INSERT -- a real failure mode this project's own dev corpus hits on every
# document's first ingest, not a hypothetical.
_DOCUMENTS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string()),
        pa.field("namespace", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("title", pa.string()),
        pa.field("doc_text_sha256", pa.string()),
        pa.field("parser_id", pa.string()),
        pa.field("char_len", pa.int32()),
        pa.field("acl_key", pa.string()),
        pa.field("ingest_run_id", pa.string()),
        pa.field("tombstoned_at", pa.timestamp("us")),
    ]
)

_CHUNKS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("chunk_uid", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("chunker_config_id", pa.string()),
        pa.field("char_start", pa.int32()),
        pa.field("char_end", pa.int32()),
        pa.field("raw_text_sha256", pa.string()),
        pa.field("dup_of", pa.string()),
    ]
)

# Columns callers are allowed to order `list_runs` by. Identifiers cannot be
# passed as bound parameters in SQL, so this whitelist -- not string
# escaping -- is what keeps `order_by` from being an injection vector.
_ORDERABLE_COLUMNS = frozenset(
    {"started_at", "finished_at", "status", "cost_usd", "p50_ms", "p95_ms", "peak_rss_mb"}
)


def _now() -> datetime:
    return datetime.now(UTC)


def _dumps(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def _loads(s: str | None) -> Any:
    if s is None:
        return None
    return orjson.loads(s)


@dataclass(frozen=True)
class RunRecord:
    """One row of ``runs``, deserialised."""

    run_id: str
    manifest: dict[str, Any]
    git_sha: str
    started_at: datetime | None
    finished_at: datetime | None
    status: str
    metrics: dict[str, Any]
    cost_usd: float | None
    p50_ms: float | None
    p95_ms: float | None
    peak_rss_mb: float | None
    notes: str

    def as_manifest(self) -> RunManifest:
        """Reconstruct the typed ``RunManifest`` this run was recorded from."""
        return RunManifest.model_validate(self.manifest)


@dataclass(frozen=True)
class ComparisonResult:
    """Per-query aligned metric values for two runs, ready for a paired test.

    Only query ids present in *both* runs' ``run_metrics`` for ``metric`` are
    kept -- a paired test is undefined for a query only one run answered.
    """

    metric: str
    run_a: str
    run_b: str
    query_ids: tuple[str, ...]
    values_a: np.ndarray
    values_b: np.ndarray

    def __len__(self) -> int:
        return len(self.query_ids)


@dataclass(frozen=True)
class ConstraintViolation:
    """One recorded metric that exceeds its manifest's pre-registered limit."""

    constraint: str
    limit: float
    actual: float


class LedgerConflictError(ValueError):
    """Raised when a ``run_id`` is re-recorded with a materially different manifest.

    See the module docstring: this should be unreachable in normal operation
    because ``run_id`` already hashes the whole manifest. It is kept as an
    explicit, named exception (rather than a generic ``ValueError``) so
    callers can catch precisely this failure mode without also catching
    ordinary validation errors.
    """


def _migrate_v1_to_v2(conn: duckdb.DuckDBPyConnection) -> None:
    """v1 -> v2: ``chunks`` gains ``tombstoned_at``, for auditable removal.

    ``Ledger.tombstone_document`` needs a way to mark a document's chunks
    removed without deleting the rows -- the same "audit trail over
    deletion" rule ``documents.tombstoned_at`` already follows (see that
    column and CLAUDE.md's ledger invariants). ``ALTER TABLE ... ADD
    COLUMN`` is safe whether ``chunks`` already holds rows (DuckDB backfills
    them with NULL, i.e. "not yet tombstoned") or is still empty.
    """
    conn.execute("ALTER TABLE chunks ADD COLUMN tombstoned_at TIMESTAMP")


class Ledger:
    """A DuckDB-backed store of run manifests, metrics, and pipeline metadata.

    Schema creation is idempotent (``CREATE TABLE IF NOT EXISTS``) so opening
    the same file twice -- or opening a brand-new file -- both just work,
    with no separate "init" step required before the ledger is usable.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn = duckdb.connect(str(self.path))
        self._lock = threading.Lock()
        self._create_schema()
        self._migrate()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- schema --------------------------------------------------------------

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       VARCHAR PRIMARY KEY,
                    manifest     JSON NOT NULL,
                    git_sha      VARCHAR,
                    started_at   TIMESTAMP,
                    finished_at  TIMESTAMP,
                    status       VARCHAR NOT NULL,
                    metrics      JSON,
                    cost_usd     DOUBLE,
                    p50_ms       DOUBLE,
                    p95_ms       DOUBLE,
                    peak_rss_mb  DOUBLE,
                    notes        VARCHAR
                )
                """
            )
            # Long/tidy per-query metrics -- see the module docstring for why
            # this table, not just the `runs.metrics` aggregate, exists.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_id    VARCHAR NOT NULL,
                    query_id  VARCHAR NOT NULL,
                    metric    VARCHAR NOT NULL,
                    value     DOUBLE NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id           VARCHAR PRIMARY KEY,
                    namespace        VARCHAR NOT NULL,
                    source_url       VARCHAR,
                    title            VARCHAR,
                    doc_text_sha256  VARCHAR,
                    parser_id        VARCHAR,
                    char_len         INTEGER,
                    acl_key          VARCHAR,
                    ingest_run_id    VARCHAR,
                    tombstoned_at    TIMESTAMP
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_uid          VARCHAR PRIMARY KEY,
                    doc_id             VARCHAR NOT NULL,
                    chunker_config_id  VARCHAR NOT NULL,
                    char_start         INTEGER NOT NULL,
                    char_end           INTEGER NOT NULL,
                    raw_text_sha256    VARCHAR,
                    dup_of             VARCHAR,
                    tombstoned_at      TIMESTAMP
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_runs (
                    ingest_run_id  VARCHAR PRIMARY KEY,
                    started_at     TIMESTAMP,
                    finished_at    TIMESTAMP,
                    source         VARCHAR,
                    n_documents    INTEGER,
                    n_errors       INTEGER,
                    notes          VARCHAR
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS errors (
                    run_id   VARCHAR,
                    stage    VARCHAR NOT NULL,
                    message  VARCHAR NOT NULL,
                    ts       TIMESTAMP NOT NULL
                )
                """
            )

    def _migrate(self) -> None:
        """Forward-only schema migration hook.

        ``migrations`` maps "version I am upgrading away from" to a function
        that mutates the schema in place. Schema version 2 added
        ``chunks.tombstoned_at`` (see ``_migrate_v1_to_v2``); a brand-new
        database never runs it, because ``_create_schema`` above already
        creates ``chunks`` with that column -- ``migrations`` only fires for
        a database that was created under an older version and needs to be
        brought forward.
        """
        migrations: dict[int, Any] = {1: _migrate_v1_to_v2}

        with self._lock:
            row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    [_CURRENT_SCHEMA_VERSION],
                )
                return

            version = row[0]
            while version < _CURRENT_SCHEMA_VERSION:
                migrate_fn = migrations.get(version)
                if migrate_fn is None:
                    raise RuntimeError(
                        f"no migration registered to move schema_version {version} forward "
                        f"(target {_CURRENT_SCHEMA_VERSION})"
                    )
                migrate_fn(self._conn)
                version += 1
                self._conn.execute("UPDATE schema_version SET version = ?", [version])

    # -- runs ------------------------------------------------------------

    def record_run(
        self,
        manifest: RunManifest,
        *,
        run_id: str | None = None,
        git_sha: str = "",
        started_at: datetime | None = None,
        status: str = "running",
        notes: str = "",
    ) -> str:
        """Insert a new run, or no-op if this exact ``(run_id, manifest)`` already exists.

        Args:
            manifest: the full, validated pipeline configuration for this run.
            run_id: override for the id under which to record this manifest.
                Defaults to ``manifest.run_id()``. Exists so a caller that
                already computed the id (e.g. to name output files before
                calling this) does not pay for hashing twice, and is what
                makes the conflict guard below reachable in tests: the
                normal path can never produce two different manifests with
                the same run_id (barring a hash collision), but an explicit
                override can be used to simulate -- and thereby test -- that
                failure mode on purpose.

        Returns:
            The run_id the manifest was (or already was) recorded under.

        Raises:
            LedgerConflictError: ``run_id`` already exists with a manifest
                that differs from the one given.
        """
        resolved_run_id = run_id or manifest.run_id()
        manifest_dict = manifest.model_dump(mode="json")
        manifest_json = _dumps(manifest_dict)

        with self._lock:
            existing = self._conn.execute(
                "SELECT manifest FROM runs WHERE run_id = ?", [resolved_run_id]
            ).fetchone()

            if existing is not None:
                existing_manifest = _loads(existing[0])
                if existing_manifest != manifest_dict:
                    raise LedgerConflictError(
                        f"run_id {resolved_run_id!r} is already recorded with a different "
                        "manifest. run_id is derived from the full manifest, so this means "
                        "either the same id was computed for two distinct configs (a hash "
                        "collision) or an explicit run_id override was reused across two "
                        "different manifests -- either way, do not overwrite the existing row."
                    )
                return resolved_run_id  # idempotent no-op

            self._conn.execute(
                """
                INSERT INTO runs (run_id, manifest, git_sha, started_at, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [resolved_run_id, manifest_json, git_sha, started_at or _now(), status, notes],
            )
        return resolved_run_id

    def finish_run(
        self,
        run_id: str,
        *,
        metrics: Mapping[str, float] | None = None,
        per_query: Mapping[str, Mapping[str, float]] | None = None,
        cost_usd: float | None = None,
        latencies_ms: Sequence[float] | None = None,
        peak_rss_mb: float | None = None,
        status: str = "finished",
        finished_at: datetime | None = None,
    ) -> None:
        """Record final outcomes for a run already created by ``record_run``.

        ``per_query`` is exploded into ``run_metrics`` rows -- one per
        ``(query_id, metric)`` pair -- which is the tidy shape
        ``compare()`` and any downstream paired statistical test consume.

        Idempotent by replacement: calling this twice for the same run_id
        (e.g. after fixing an aggregation bug) overwrites the previous
        per-query rows and summary fields rather than appending duplicates.

        Raises:
            KeyError: ``run_id`` was never recorded via ``record_run``.
        """
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", [run_id]
            ).fetchone()
            if exists is None:
                raise KeyError(f"unknown run_id {run_id!r}; call record_run() first")

            p50_ms = p95_ms = None
            if latencies_ms:
                arr = np.asarray(list(latencies_ms), dtype=float)
                p50_ms = float(np.percentile(arr, 50))
                p95_ms = float(np.percentile(arr, 95))

            self._conn.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?, metrics = ?, cost_usd = ?,
                    p50_ms = ?, p95_ms = ?, peak_rss_mb = ?
                WHERE run_id = ?
                """,
                [
                    status,
                    finished_at or _now(),
                    _dumps(dict(metrics or {})),
                    cost_usd,
                    p50_ms,
                    p95_ms,
                    peak_rss_mb,
                    run_id,
                ],
            )

            self._conn.execute("DELETE FROM run_metrics WHERE run_id = ?", [run_id])
            if per_query:
                rows = [
                    (run_id, query_id, metric, float(value))
                    for query_id, metric_values in per_query.items()
                    for metric, value in metric_values.items()
                ]
                if rows:
                    self._conn.executemany(
                        "INSERT INTO run_metrics (run_id, query_id, metric, value) "
                        "VALUES (?, ?, ?, ?)",
                        rows,
                    )

    def record_error(
        self, run_id: str | None, stage: str, message: str, ts: datetime | None = None
    ) -> None:
        """Append one error/warning row. Never overwrites -- errors accumulate."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO errors (run_id, stage, message, ts) VALUES (?, ?, ?, ?)",
                [run_id, stage, message, ts or _now()],
            )

    # -- ingest runs -----------------------------------------------------

    def record_ingest_run(
        self, ingest_run_id: str, *, source: str, started_at: datetime | None = None
    ) -> None:
        """Insert a new ``ingest_runs`` row, or no-op if this id is already recorded.

        Mirrors ``record_run``'s idempotency (see that method): retrying an
        ingest that already wrote a bookkeeping row for this
        ``ingest_run_id`` must not create a second one.
        """
        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM ingest_runs WHERE ingest_run_id = ?", [ingest_run_id]
            ).fetchone()
            if existing is not None:
                return
            self._conn.execute(
                """
                INSERT INTO ingest_runs (ingest_run_id, started_at, source, n_documents, n_errors, notes)
                VALUES (?, ?, ?, 0, 0, '')
                """,
                [ingest_run_id, started_at or _now(), source],
            )

    def finish_ingest_run(
        self,
        ingest_run_id: str,
        *,
        n_documents: int,
        n_errors: int,
        finished_at: datetime | None = None,
        notes: str = "",
    ) -> None:
        """Record final counts for an ingest run already created by ``record_ingest_run``.

        Idempotent by replacement, like ``finish_run``: calling this again
        for the same id overwrites the summary fields rather than erroring.

        Raises:
            KeyError: ``ingest_run_id`` was never recorded.
        """
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM ingest_runs WHERE ingest_run_id = ?", [ingest_run_id]
            ).fetchone()
            if exists is None:
                raise KeyError(f"unknown ingest_run_id {ingest_run_id!r}; call record_ingest_run() first")
            self._conn.execute(
                """
                UPDATE ingest_runs
                SET finished_at = ?, n_documents = ?, n_errors = ?, notes = ?
                WHERE ingest_run_id = ?
                """,
                [finished_at or _now(), n_documents, n_errors, notes, ingest_run_id],
            )

    # -- documents & chunks ------------------------------------------------
    #
    # Both upserts are single set-based `INSERT ... ON CONFLICT DO UPDATE`
    # statements over a registered Arrow table, never one statement per row
    # (see CLAUDE.md: "100k chunks row-by-row through DuckDB is unusably
    # slow"). `upsert_chunks` deliberately never writes `tombstoned_at` --
    # that column is owned exclusively by `tombstone_document`, so
    # re-upserting a chunk (an ordinary idempotent re-ingest) can never
    # accidentally clear or overwrite a previously recorded removal.

    def upsert_documents(self, docs: Sequence[Document]) -> None:
        """Idempotent, batched insert-or-update of ``Document`` rows into ``documents``."""
        if not docs:
            return
        rows = [
            {
                "doc_id": d.doc_id,
                "namespace": d.namespace.value if isinstance(d.namespace, Namespace) else d.namespace,
                "source_url": d.source_url,
                "title": d.title,
                "doc_text_sha256": d.doc_text_sha256,
                "parser_id": d.parser_id,
                "char_len": d.char_len,
                "acl_key": d.acl_key,
                "ingest_run_id": d.ingest_run_id,
                "tombstoned_at": d.tombstoned_at,
            }
            for d in docs
        ]
        table = pa.Table.from_pylist(rows, schema=_DOCUMENTS_ARROW_SCHEMA)
        with self._lock:
            self._conn.register("_selfrag_documents_batch", table)
            try:
                self._conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, namespace, source_url, title, doc_text_sha256,
                        parser_id, char_len, acl_key, ingest_run_id, tombstoned_at
                    )
                    SELECT
                        doc_id, namespace, source_url, title, doc_text_sha256,
                        parser_id, char_len, acl_key, ingest_run_id, tombstoned_at
                    FROM _selfrag_documents_batch
                    ON CONFLICT (doc_id) DO UPDATE SET
                        namespace = excluded.namespace,
                        source_url = excluded.source_url,
                        title = excluded.title,
                        doc_text_sha256 = excluded.doc_text_sha256,
                        parser_id = excluded.parser_id,
                        char_len = excluded.char_len,
                        acl_key = excluded.acl_key,
                        ingest_run_id = excluded.ingest_run_id,
                        tombstoned_at = excluded.tombstoned_at
                    """
                )
            finally:
                self._conn.unregister("_selfrag_documents_batch")

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Idempotent, batched insert-or-update of ``Chunk`` rows into ``chunks``.

        Primary key is ``chunk_uid``, which hashes coordinates only (see
        ``selfrag.ids.chunk_uid``) -- re-chunking the same frozen canonical
        text with the same chunker config always produces the same ids, so
        re-running this over the same corpus updates existing rows in place
        rather than duplicating them. That is the mechanism the Phase 1 exit
        criterion depends on.
        """
        if not chunks:
            return
        rows = [
            {
                "chunk_uid": c.chunk_uid,
                "doc_id": c.doc_id,
                "chunker_config_id": c.chunker_config_id,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "raw_text_sha256": c.raw_text_sha256,
                "dup_of": c.dup_of,
            }
            for c in chunks
        ]
        table = pa.Table.from_pylist(rows, schema=_CHUNKS_ARROW_SCHEMA)
        with self._lock:
            self._conn.register("_selfrag_chunks_batch", table)
            try:
                self._conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_uid, doc_id, chunker_config_id, char_start, char_end,
                        raw_text_sha256, dup_of
                    )
                    SELECT
                        chunk_uid, doc_id, chunker_config_id, char_start, char_end,
                        raw_text_sha256, dup_of
                    FROM _selfrag_chunks_batch
                    ON CONFLICT (chunk_uid) DO UPDATE SET
                        doc_id = excluded.doc_id,
                        chunker_config_id = excluded.chunker_config_id,
                        char_start = excluded.char_start,
                        char_end = excluded.char_end,
                        raw_text_sha256 = excluded.raw_text_sha256,
                        dup_of = excluded.dup_of
                    """
                )
            finally:
                self._conn.unregister("_selfrag_chunks_batch")

    def get_chunk_ids(self, doc_id: str | None = None) -> set[str]:
        """Every ``chunk_uid`` in the ledger, optionally restricted to one document.

        Unfiltered by tombstone status on purpose -- a tombstoned chunk's
        row still physically exists (see the module docstring's "auditable,
        not deleted" rule), and this is the primitive the Phase 1 exit
        criterion's "snapshot the id set, re-ingest, compare" check is built
        from.
        """
        if doc_id is None:
            rows = self._conn.execute("SELECT chunk_uid FROM chunks").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT chunk_uid FROM chunks WHERE doc_id = ?", [doc_id]
            ).fetchall()
        return {r[0] for r in rows}

    def count_chunks(self) -> int:
        """Total row count in ``chunks``, tombstoned or not."""
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_document(self, doc_id: str) -> Document:
        """Fetch one document by id.

        Raises:
            KeyError: no such doc_id.
        """
        row = self._conn.execute(
            """
            SELECT doc_id, namespace, source_url, title, doc_text_sha256, parser_id,
                   char_len, acl_key, ingest_run_id, tombstoned_at
            FROM documents WHERE doc_id = ?
            """,
            [doc_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown doc_id {doc_id!r}")
        return self._row_to_document(row)

    def live_documents(self) -> list[Document]:
        """Every document whose ``tombstoned_at`` is unset, ordered by ``doc_id``."""
        rows = self._conn.execute(
            """
            SELECT doc_id, namespace, source_url, title, doc_text_sha256, parser_id,
                   char_len, acl_key, ingest_run_id, tombstoned_at
            FROM documents WHERE tombstoned_at IS NULL
            ORDER BY doc_id
            """
        ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def tombstone_document(self, doc_id: str, *, tombstoned_at: datetime | None = None) -> None:
        """Mark ``doc_id`` and every one of its chunks removed, without deleting any row.

        A deliberate removal (unlike a document that was simply never seen)
        has to be distinguishable after the fact, and a re-ingest of the
        same ``doc_id`` later has to be able to tell the two apart -- that
        is only possible if the row survives with a marker on it, so this
        never issues a ``DELETE``.

        Raises:
            KeyError: ``doc_id`` is not a known document.
        """
        ts = tombstoned_at or _now()
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", [doc_id]
            ).fetchone()
            if exists is None:
                raise KeyError(f"unknown doc_id {doc_id!r}")
            self._conn.execute(
                "UPDATE documents SET tombstoned_at = ? WHERE doc_id = ?", [ts, doc_id]
            )
            self._conn.execute(
                "UPDATE chunks SET tombstoned_at = ? WHERE doc_id = ?", [ts, doc_id]
            )

    def get_run(self, run_id: str) -> RunRecord:
        """Fetch one run by id.

        Raises:
            KeyError: no such run_id.
        """
        row = self._conn.execute(
            """
            SELECT run_id, manifest, git_sha, started_at, finished_at, status,
                   metrics, cost_usd, p50_ms, p95_ms, peak_rss_mb, notes
            FROM runs WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id {run_id!r}")
        return self._row_to_record(row)

    def list_runs(self, limit: int = 20, order_by: str = "started_at", descending: bool = True) -> list[RunRecord]:
        """List recent runs.

        Args:
            order_by: a column name from ``_ORDERABLE_COLUMNS``. Rejected
                (rather than interpolated) if it is not, since column
                identifiers cannot be sent as bound SQL parameters and
                accepting an arbitrary string here would be a SQL-injection
                vector -- an allowlist is the correct substitute, not string
                sanitisation.

        Raises:
            ValueError: ``order_by`` is not an allowed column, or ``limit``
                is not positive.
        """
        if order_by not in _ORDERABLE_COLUMNS:
            raise ValueError(f"cannot order by {order_by!r}; allowed columns: {sorted(_ORDERABLE_COLUMNS)}")
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        direction = "DESC" if descending else "ASC"
        rows = self._conn.execute(
            f"""
            SELECT run_id, manifest, git_sha, started_at, finished_at, status,
                   metrics, cost_usd, p50_ms, p95_ms, peak_rss_mb, notes
            FROM runs
            ORDER BY {order_by} {direction}
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def compare(self, run_a: str, run_b: str, metric: str) -> ComparisonResult:
        """Per-query aligned metric arrays for two runs, for a paired test.

        Only queries answered by *both* runs for ``metric`` are included,
        sorted by query_id so ``values_a[i]`` and ``values_b[i]`` are always
        the same query's score in both arrays.
        """
        rows = self._conn.execute(
            """
            SELECT a.query_id, a.value, b.value
            FROM run_metrics a
            JOIN run_metrics b
              ON a.query_id = b.query_id AND a.metric = b.metric
            WHERE a.run_id = ? AND b.run_id = ? AND a.metric = ?
            ORDER BY a.query_id
            """,
            [run_a, run_b, metric],
        ).fetchall()
        query_ids = tuple(r[0] for r in rows)
        values_a = np.asarray([r[1] for r in rows], dtype=float)
        values_b = np.asarray([r[2] for r in rows], dtype=float)
        return ComparisonResult(
            metric=metric, run_a=run_a, run_b=run_b, query_ids=query_ids, values_a=values_a, values_b=values_b
        )

    def constraint_violations(self, run_id: str) -> list[ConstraintViolation]:
        """Check a run's recorded p95 latency / peak RSS / cost against its own manifest.

        A config that violates one of its pre-registered ``Constraints`` is
        disqualified regardless of quality score (see
        ``selfrag.schema.Constraints``) -- this is the machine-checkable half
        of that rule. ``cost_usd`` is recorded (via ``finish_run``) as cost
        *per query*, matching ``Constraints.max_cost_per_query_usd``.
        """
        record = self.get_run(run_id)
        constraints = record.as_manifest().constraints
        violations: list[ConstraintViolation] = []

        checks = (
            ("max_p95_latency_ms", record.p95_ms, constraints.max_p95_latency_ms),
            ("max_peak_rss_mb", record.peak_rss_mb, constraints.max_peak_rss_mb),
            ("max_cost_per_query_usd", record.cost_usd, constraints.max_cost_per_query_usd),
        )
        for name, actual, limit in checks:
            if actual is not None and actual > limit:
                violations.append(ConstraintViolation(constraint=name, limit=limit, actual=actual))
        return violations

    # -- helpers -------------------------------------------------------------

    def _row_to_record(self, row: tuple[Any, ...]) -> RunRecord:
        (
            run_id,
            manifest,
            git_sha,
            started_at,
            finished_at,
            status,
            metrics,
            cost_usd,
            p50_ms,
            p95_ms,
            peak_rss_mb,
            notes,
        ) = row
        return RunRecord(
            run_id=run_id,
            manifest=_loads(manifest) or {},
            git_sha=git_sha or "",
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            metrics=_loads(metrics) or {},
            cost_usd=cost_usd,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            peak_rss_mb=peak_rss_mb,
            notes=notes or "",
        )

    def _row_to_document(self, row: tuple[Any, ...]) -> Document:
        (
            doc_id,
            namespace,
            source_url,
            title,
            doc_text_sha256,
            parser_id,
            char_len,
            acl_key,
            ingest_run_id,
            tombstoned_at,
        ) = row
        return Document(
            doc_id=doc_id,
            namespace=Namespace(namespace),
            source_url=source_url or "",
            title=title or "",
            doc_text_sha256=doc_text_sha256 or "",
            parser_id=parser_id or "",
            char_len=char_len or 0,
            acl_key=acl_key or "public",
            ingest_run_id=ingest_run_id or "",
            tombstoned_at=tombstoned_at,
        )

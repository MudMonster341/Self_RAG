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

from selfrag.schema import RunManifest

_CURRENT_SCHEMA_VERSION = 1

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
                    dup_of             VARCHAR
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

        ``_MIGRATIONS`` maps "version I am upgrading away from" to a function
        that mutates the schema in place. There is exactly one schema
        version today, so the loop below runs zero times -- the mechanism is
        real and will run the day a second version exists, it simply has
        nothing registered yet.
        """
        migrations: dict[int, Any] = {}

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

"""Tests for selfrag.ledger.

Two behaviours get the most scrutiny because they are the ledger's whole
reason for existing as a *trustworthy* record: record_run's idempotency
(and its refusal to silently overwrite a conflicting manifest under the
same run_id), and compare()'s per-query alignment, which is what any paired
statistical test downstream depends on.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from selfrag.ledger import ConstraintViolation, Ledger, LedgerConflictError
from selfrag.schema import Constraints, RunManifest, Split


def _make_manifest(**overrides) -> RunManifest:
    defaults = dict(
        corpus_snapshot="snap1",
        dedup_config_id="none",
        parser_id="p1",
        chunker_config_id="fixed@abcd1234",
        embedder_id="e1",
        embedder_snapshot_date="2026-01-01",
        embedding_dim=8,
        qrels_version=1,
        split=Split.DEV,
    )
    defaults.update(overrides)
    return RunManifest(**defaults)


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.duckdb") as led:
        yield led


class TestSchemaCreation:
    def test_opening_twice_does_not_error(self, tmp_path):
        db_path = tmp_path / "ledger.duckdb"
        with Ledger(db_path):
            pass
        with Ledger(db_path):
            pass  # must not raise -- CREATE TABLE IF NOT EXISTS is idempotent

    def test_schema_version_row_exists(self, ledger):
        row = ledger._conn.execute("SELECT version FROM schema_version").fetchone()
        assert row == (1,)

    def test_all_required_tables_exist(self, ledger):
        tables = {
            r[0]
            for r in ledger._conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        for expected in ("runs", "run_metrics", "documents", "chunks", "ingest_runs", "errors", "schema_version"):
            assert expected in tables


class TestRecordRunIdempotency:
    def test_returns_the_manifest_run_id(self, ledger):
        manifest = _make_manifest()
        run_id = ledger.record_run(manifest)
        assert run_id == manifest.run_id()

    def test_recording_same_manifest_twice_does_not_duplicate(self, ledger):
        manifest = _make_manifest()
        ledger.record_run(manifest)
        ledger.record_run(manifest)
        count = ledger._conn.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?", [manifest.run_id()]
        ).fetchone()[0]
        assert count == 1

    def test_recording_same_manifest_twice_returns_same_run_id(self, ledger):
        manifest = _make_manifest()
        first = ledger.record_run(manifest)
        second = ledger.record_run(manifest)
        assert first == second

    def test_conflicting_manifest_under_same_run_id_raises(self, ledger):
        """run_id is derived from the whole manifest, so this can only be
        provoked by forcing an explicit run_id override -- see the
        `run_id` parameter docstring on record_run. That override is what
        makes this failure mode reachable at all, for a guard that should
        never fire in normal use."""
        manifest_a = _make_manifest()
        manifest_b = _make_manifest(notes="a materially different manifest")
        run_id = ledger.record_run(manifest_a)

        with pytest.raises(LedgerConflictError):
            ledger.record_run(manifest_b, run_id=run_id)

    def test_conflict_does_not_corrupt_the_existing_row(self, ledger):
        manifest_a = _make_manifest()
        manifest_b = _make_manifest(notes="different")
        run_id = ledger.record_run(manifest_a)

        with pytest.raises(LedgerConflictError):
            ledger.record_run(manifest_b, run_id=run_id)

        stored = ledger.get_run(run_id)
        assert stored.as_manifest().notes == manifest_a.notes

    def test_different_manifests_get_different_run_ids_naturally(self, ledger):
        run_id_a = ledger.record_run(_make_manifest())
        run_id_b = ledger.record_run(_make_manifest(chunker_config_id="fixed@ffffffff"))
        assert run_id_a != run_id_b


class TestFinishRun:
    def test_unknown_run_id_raises(self, ledger):
        with pytest.raises(KeyError):
            ledger.finish_run("does-not-exist", metrics={"m": 1.0})

    def test_persists_per_query_metrics(self, ledger):
        run_id = ledger.record_run(_make_manifest())
        ledger.finish_run(
            run_id,
            metrics={"ndcg@10": 0.5},
            per_query={"q1": {"ndcg@10": 0.4}, "q2": {"ndcg@10": 0.6}},
        )
        rows = ledger._conn.execute(
            "SELECT query_id, metric, value FROM run_metrics WHERE run_id = ? ORDER BY query_id", [run_id]
        ).fetchall()
        assert rows == [("q1", "ndcg@10", 0.4), ("q2", "ndcg@10", 0.6)]

    def test_computes_p50_and_p95_from_latencies(self, ledger):
        run_id = ledger.record_run(_make_manifest())
        latencies = list(range(1, 101))  # 1..100 ms
        ledger.finish_run(run_id, latencies_ms=latencies)
        record = ledger.get_run(run_id)
        assert record.p50_ms == pytest.approx(50.5, abs=1.0)
        assert record.p95_ms == pytest.approx(95, abs=1.0)

    def test_rerunning_finish_run_replaces_rather_than_duplicates(self, ledger):
        run_id = ledger.record_run(_make_manifest())
        ledger.finish_run(run_id, per_query={"q1": {"m": 1.0}})
        ledger.finish_run(run_id, per_query={"q1": {"m": 2.0}})
        rows = ledger._conn.execute(
            "SELECT value FROM run_metrics WHERE run_id = ? AND query_id = ?", [run_id, "q1"]
        ).fetchall()
        assert rows == [(2.0,)]

    def test_sets_status_and_finished_at(self, ledger):
        run_id = ledger.record_run(_make_manifest())
        assert ledger.get_run(run_id).status == "running"
        ledger.finish_run(run_id, metrics={})
        record = ledger.get_run(run_id)
        assert record.status == "finished"
        assert record.finished_at is not None


class TestGetRun:
    def test_unknown_run_id_raises_key_error(self, ledger):
        with pytest.raises(KeyError):
            ledger.get_run("nonexistent")

    def test_round_trips_the_manifest(self, ledger):
        manifest = _make_manifest()
        run_id = ledger.record_run(manifest)
        record = ledger.get_run(run_id)
        assert record.as_manifest().model_dump(mode="json") == manifest.model_dump(mode="json")


class TestListRuns:
    def test_orders_by_started_at_descending_by_default(self, ledger):
        first = ledger.record_run(_make_manifest(), started_at=datetime(2026, 1, 1, tzinfo=UTC))
        second = ledger.record_run(
            _make_manifest(chunker_config_id="fixed@ffffffff"),
            started_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        runs = ledger.list_runs(limit=10)
        assert [r.run_id for r in runs] == [second, first]

    def test_respects_limit(self, ledger):
        for i in range(5):
            ledger.record_run(_make_manifest(chunker_config_id=f"fixed@{i:08x}"))
        assert len(ledger.list_runs(limit=2)) == 2

    def test_rejects_disallowed_order_column(self, ledger):
        with pytest.raises(ValueError, match="allowed columns"):
            ledger.list_runs(order_by="run_id; DROP TABLE runs")

    def test_rejects_non_positive_limit(self, ledger):
        with pytest.raises(ValueError):
            ledger.list_runs(limit=0)


class TestCompare:
    def test_returns_aligned_arrays_for_shared_queries(self, ledger):
        run_a = ledger.record_run(_make_manifest())
        run_b = ledger.record_run(_make_manifest(chunker_config_id="fixed@ffffffff"))
        ledger.finish_run(run_a, per_query={"q1": {"ndcg@10": 0.4}, "q2": {"ndcg@10": 0.6}})
        ledger.finish_run(run_b, per_query={"q1": {"ndcg@10": 0.9}, "q2": {"ndcg@10": 0.5}})

        result = ledger.compare(run_a, run_b, "ndcg@10")

        assert result.query_ids == ("q1", "q2")
        assert list(result.values_a) == [0.4, 0.6]
        assert list(result.values_b) == [0.9, 0.5]
        assert len(result) == 2

    def test_excludes_queries_not_shared_by_both_runs(self, ledger):
        run_a = ledger.record_run(_make_manifest())
        run_b = ledger.record_run(_make_manifest(chunker_config_id="fixed@ffffffff"))
        ledger.finish_run(run_a, per_query={"q1": {"m": 1.0}, "q_only_in_a": {"m": 9.0}})
        ledger.finish_run(run_b, per_query={"q1": {"m": 2.0}, "q_only_in_b": {"m": 8.0}})

        result = ledger.compare(run_a, run_b, "m")
        assert result.query_ids == ("q1",)

    def test_values_stay_paired_per_query_under_reordering(self, ledger):
        """The point of a tidy run_metrics table is that pairing survives
        insertion order -- this asserts that explicitly."""
        run_a = ledger.record_run(_make_manifest())
        run_b = ledger.record_run(_make_manifest(chunker_config_id="fixed@ffffffff"))
        ledger.finish_run(run_a, per_query={"z": {"m": 1.0}, "a": {"m": 2.0}})
        ledger.finish_run(run_b, per_query={"a": {"m": 20.0}, "z": {"m": 10.0}})

        result = ledger.compare(run_a, run_b, "m")
        pairs = dict(
            zip(result.query_ids, zip(result.values_a, result.values_b, strict=True), strict=True)
        )
        assert pairs["z"] == (1.0, 10.0)
        assert pairs["a"] == (2.0, 20.0)

    def test_no_shared_metric_gives_empty_arrays(self, ledger):
        run_a = ledger.record_run(_make_manifest())
        run_b = ledger.record_run(_make_manifest(chunker_config_id="fixed@ffffffff"))
        result = ledger.compare(run_a, run_b, "nonexistent_metric")
        assert result.query_ids == ()
        assert len(result.values_a) == 0
        assert len(result.values_b) == 0


class TestConstraintViolations:
    def test_no_violations_when_within_limits(self, ledger):
        manifest = _make_manifest(constraints=Constraints(max_p95_latency_ms=1000.0, max_peak_rss_mb=4000.0, max_cost_per_query_usd=1.0))
        run_id = ledger.record_run(manifest)
        ledger.finish_run(run_id, latencies_ms=[10, 20, 30], peak_rss_mb=500.0, cost_usd=0.001)
        assert ledger.constraint_violations(run_id) == []

    def test_p95_violation_detected(self, ledger):
        manifest = _make_manifest(constraints=Constraints(max_p95_latency_ms=100.0))
        run_id = ledger.record_run(manifest)
        ledger.finish_run(run_id, latencies_ms=list(range(1, 1001)))  # p95 ~= 950ms
        violations = ledger.constraint_violations(run_id)
        assert any(v.constraint == "max_p95_latency_ms" for v in violations)

    def test_rss_violation_detected(self, ledger):
        manifest = _make_manifest(constraints=Constraints(max_peak_rss_mb=1000.0))
        run_id = ledger.record_run(manifest)
        ledger.finish_run(run_id, peak_rss_mb=5000.0)
        violations = ledger.constraint_violations(run_id)
        assert any(v.constraint == "max_peak_rss_mb" and v.actual == 5000.0 for v in violations)

    def test_cost_violation_detected(self, ledger):
        manifest = _make_manifest(constraints=Constraints(max_cost_per_query_usd=0.01))
        run_id = ledger.record_run(manifest)
        ledger.finish_run(run_id, cost_usd=0.5)
        violations = ledger.constraint_violations(run_id)
        assert any(v.constraint == "max_cost_per_query_usd" for v in violations)
        assert isinstance(violations[0], ConstraintViolation)

    def test_unfinished_run_with_no_recorded_metrics_has_no_violations(self, ledger):
        run_id = ledger.record_run(_make_manifest())
        assert ledger.constraint_violations(run_id) == []


class TestRecordError:
    def test_appends_without_overwriting(self, ledger):
        run_id = ledger.record_run(_make_manifest())
        ledger.record_error(run_id, "retrieval", "first error")
        ledger.record_error(run_id, "generation", "second error")
        rows = ledger._conn.execute(
            "SELECT stage, message FROM errors WHERE run_id = ? ORDER BY stage", [run_id]
        ).fetchall()
        assert rows == [("generation", "second error"), ("retrieval", "first error")]

    def test_accepts_null_run_id_for_ingest_time_errors(self, ledger):
        ledger.record_error(None, "ingest", "could not parse pdf")
        rows = ledger._conn.execute("SELECT stage, message FROM errors WHERE run_id IS NULL").fetchall()
        assert rows == [("ingest", "could not parse pdf")]

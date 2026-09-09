"""Tests for selfrag.eval.runfile.

Covers: streaming write/read roundtrip (including forcing multiple internal
flushes), the manifest sidecar being self-describing JSON, refusal to
overwrite an existing run file, row validation, and -- the most important
property of this module -- that ``rescore`` reproduces exactly the same
numbers as calling the metric functions inline on the same data.
"""

from __future__ import annotations

import pytest

from selfrag.eval.metrics import OverlapRule, OverlapSpec, ndcg_at_k, recall_at_k
from selfrag.eval.qrels import QrelStore
from selfrag.eval.runfile import RunFileWriter, read_run, rescore
from selfrag.schema import Qrel, RunManifest, Split


def make_manifest(**overrides) -> RunManifest:
    defaults = dict(
        corpus_snapshot="snap-2026-01-01",
        dedup_config_id="dedup-v1",
        parser_id="parser-v1",
        chunker_config_id="chunker-v1",
        embedder_id="embedder-v1",
        embedder_snapshot_date="2026-01-01",
        embedding_dim=384,
        qrels_version=1,
        split=Split.DEV,
    )
    defaults.update(overrides)
    return RunManifest(**defaults)


def make_qrel(query_id="q1", doc_id="d1", char_start=0, char_end=50, grade=1, judge="human:test"):
    return Qrel(
        qrels_version=1,
        query_id=query_id,
        doc_id=doc_id,
        char_start=char_start,
        char_end=char_end,
        grade=grade,
        judge=judge,
    )


IOU_01 = OverlapSpec(rule=OverlapRule.IOU, threshold=0.1)


# --------------------------------------------------------------------------
# Write / read roundtrip
# --------------------------------------------------------------------------


def test_write_and_read_run_roundtrip(tmp_path):
    manifest = make_manifest()
    path = tmp_path / "run.parquet"
    with RunFileWriter(path, manifest) as w:
        w.write_ranked_query("q1", [("d1", "c1", 0, 50, 0.9), ("d2", "c2", 0, 50, 0.5)])
        w.write_ranked_query("q2", [("d3", "c3", 0, 50, 0.7)])

    run = read_run(path)
    assert run.manifest == manifest
    assert run.query_ids() == ["q1", "q2"]

    grouped = run.group_by_query()
    assert grouped["q1"] == [("d1", 0, 50), ("d2", 0, 50)]
    assert grouped["q2"] == [("d3", 0, 50)]

    table = run.table.to_pylist()
    assert len(table) == 3
    assert all(row["run_id"] == manifest.run_id() for row in table)
    # rank must reflect list order, 1-indexed, per query
    q1_rows = sorted((r for r in table if r["query_id"] == "q1"), key=lambda r: r["rank"])
    assert [r["chunk_uid"] for r in q1_rows] == ["c1", "c2"]
    assert [r["rank"] for r in q1_rows] == [1, 2]


def test_streaming_writer_flushes_in_small_batches(tmp_path):
    """Force multiple internal flush() calls (batch_size=2, 7 rows) and verify
    the resulting file is still complete and correctly ordered -- proving
    the streaming/batching path doesn't drop or reorder rows."""
    manifest = make_manifest()
    path = tmp_path / "run.parquet"
    retrieved = [(f"d{i}", f"c{i}", 0, 50, 1.0 - i * 0.1) for i in range(7)]
    with RunFileWriter(path, manifest, batch_size=2) as w:
        w.write_ranked_query("q1", retrieved)
        assert w.n_rows == 7

    run = read_run(path)
    grouped = run.group_by_query()
    assert grouped["q1"] == [(f"d{i}", 0, 50) for i in range(7)]


def test_manifest_sidecar_is_self_describing_json(tmp_path):
    manifest = make_manifest(notes="a test run", seed=42)
    path = tmp_path / "run.parquet"
    with RunFileWriter(path, manifest):
        pass  # zero rows is fine; still a valid (empty) run

    manifest_path = tmp_path / "run.manifest.json"
    assert manifest_path.exists()
    reread = read_run(path).manifest
    assert reread == manifest
    assert reread.notes == "a test run"
    assert reread.seed == 42


def test_cannot_overwrite_existing_run_file(tmp_path):
    manifest = make_manifest()
    path = tmp_path / "run.parquet"
    with RunFileWriter(path, manifest) as w:
        w.write_ranked_query("q1", [("d1", "c1", 0, 50, 1.0)])

    with pytest.raises(FileExistsError):
        RunFileWriter(path, manifest)


def test_write_row_rejects_invalid_span_and_rank(tmp_path):
    manifest = make_manifest()
    path = tmp_path / "run.parquet"
    with RunFileWriter(path, manifest) as w:
        with pytest.raises(ValueError):
            w.write_row(
                query_id="q1", doc_id="d1", chunk_uid="c1", char_start=50, char_end=50, rank=1, score=1.0
            )
        with pytest.raises(ValueError):
            w.write_row(
                query_id="q1", doc_id="d1", chunk_uid="c1", char_start=0, char_end=50, rank=0, score=1.0
            )
        # a valid row afterward should still succeed
        w.write_row(query_id="q1", doc_id="d1", chunk_uid="c1", char_start=0, char_end=50, rank=1, score=1.0)


def test_read_run_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_run(tmp_path / "does_not_exist.parquet")


def test_read_run_missing_manifest_raises(tmp_path):
    # Write raw parquet bytes without ever going through RunFileWriter's close(),
    # so no manifest sidecar exists.
    manifest = make_manifest()
    path = tmp_path / "run.parquet"
    writer = RunFileWriter(path, manifest)
    writer.write_row(query_id="q1", doc_id="d1", chunk_uid="c1", char_start=0, char_end=50, rank=1, score=1.0)
    writer._flush()
    writer._writer.close()
    writer._writer = None
    writer._closed = True
    # Deliberately do NOT call close() / write the manifest.
    assert path.exists()
    assert not (tmp_path / "run.manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        read_run(path)


# --------------------------------------------------------------------------
# rescore reproduces inline scoring exactly
# --------------------------------------------------------------------------


def test_rescore_matches_inline_scoring(tmp_path):
    qrels_root = tmp_path / "qrels"
    store = QrelStore(qrels_root)
    qrels = [
        make_qrel(query_id="q1", doc_id="d1", char_start=0, char_end=50, grade=3),
        make_qrel(query_id="q1", doc_id="d2", char_start=0, char_end=50, grade=1),
        make_qrel(query_id="q2", doc_id="d3", char_start=0, char_end=50, grade=2),
    ]
    version = store.append_version(qrels, note="init")

    manifest = make_manifest(qrels_version=version)
    run_path = tmp_path / "run.parquet"
    q1_ranked = [("d2", "c1", 0, 50, 0.9), ("d1", "c2", 0, 50, 0.8), ("d4", "c3", 0, 50, 0.1)]
    q2_ranked = [("d5", "c4", 0, 50, 0.9), ("d3", "c5", 0, 50, 0.4)]
    with RunFileWriter(run_path, manifest) as w:
        w.write_ranked_query("q1", q1_ranked)
        w.write_ranked_query("q2", q2_ranked)

    result = rescore(run_path, store, version, ["ndcg@10", "recall@10"], IOU_01)
    rows = {row["query_id"]: row for row in result.to_pylist()}

    # Inline computation using the exact same qrels and ranked lists (as
    # metrics.py sees them: (doc_id, char_start, char_end) tuples in rank order).
    q1_qrels = [q for q in qrels if q.query_id == "q1"]
    q2_qrels = [q for q in qrels if q.query_id == "q2"]
    q1_spans = [(d, cs, ce) for d, _, cs, ce, _ in q1_ranked]
    q2_spans = [(d, cs, ce) for d, _, cs, ce, _ in q2_ranked]

    assert rows["q1"]["ndcg@10"] == pytest.approx(ndcg_at_k(q1_qrels, q1_spans, 10, IOU_01))
    assert rows["q1"]["recall@10"] == pytest.approx(recall_at_k(q1_qrels, q1_spans, 10, IOU_01))
    assert rows["q2"]["ndcg@10"] == pytest.approx(ndcg_at_k(q2_qrels, q2_spans, 10, IOU_01))
    assert rows["q2"]["recall@10"] == pytest.approx(recall_at_k(q2_qrels, q2_spans, 10, IOU_01))

    # And the result table records exactly what produced it.
    assert rows["q1"]["run_id"] == manifest.run_id()
    assert rows["q1"]["qrels_version"] == version
    assert rows["q1"]["overlap_rule"] == "iou"
    assert rows["q1"]["overlap_threshold"] == pytest.approx(0.1)


def test_rescore_against_a_later_qrels_version_changes_numbers(tmp_path):
    """The whole point of run files: rescoring the SAME run against a
    CORRECTED qrels version changes the numbers without touching the run."""
    qrels_root = tmp_path / "qrels"
    store = QrelStore(qrels_root)
    v1 = store.append_version(
        [make_qrel(query_id="q1", doc_id="d1", char_start=0, char_end=50, grade=1)], note="v1"
    )
    manifest = make_manifest(qrels_version=v1)
    run_path = tmp_path / "run.parquet"
    with RunFileWriter(run_path, manifest) as w:
        w.write_ranked_query("q1", [("d1", "c1", 0, 50, 0.9), ("d2", "c2", 0, 50, 0.5)])

    result_v1 = rescore(run_path, store, v1, ["ndcg@10"], IOU_01)
    ndcg_v1 = result_v1.to_pylist()[0]["ndcg@10"]
    assert ndcg_v1 == pytest.approx(1.0)  # d1 (the only relevant doc) is retrieved first

    # A correction: d1 turns out not relevant after all, d2 is the real answer.
    v2 = store.append_version(
        [make_qrel(query_id="q1", doc_id="d2", char_start=0, char_end=50, grade=2)], note="correction"
    )
    result_v2 = rescore(run_path, store, v2, ["ndcg@10"], IOU_01)
    ndcg_v2 = result_v2.to_pylist()[0]["ndcg@10"]
    assert ndcg_v2 < ndcg_v1  # d2 was retrieved second, not first -> worse nDCG under the correction


def test_rescore_requires_explicit_k_in_metric_spec(tmp_path):
    qrels_root = tmp_path / "qrels"
    store = QrelStore(qrels_root)
    version = store.append_version(
        [make_qrel(query_id="q1", doc_id="d1", grade=1)], note="init"
    )
    manifest = make_manifest(qrels_version=version)
    run_path = tmp_path / "run.parquet"
    with RunFileWriter(run_path, manifest) as w:
        w.write_ranked_query("q1", [("d1", "c1", 0, 50, 0.9)])

    with pytest.raises(ValueError):
        rescore(run_path, store, version, ["ndcg"], IOU_01)  # missing "@k"

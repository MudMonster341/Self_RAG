"""Tests for selfrag.eval.qrels.QrelStore.

Covers: version files are full snapshots that append-write correctly, an
existing version file can never be overwritten, duplicate judgements within
one version are rejected, diff correctly reports added/removed/regraded
judgements between two versions, sidecar metadata is written and readable,
and the lookup helpers work.
"""

from __future__ import annotations

import orjson
import pytest

from selfrag.eval.qrels import (
    QrelStore,
    QrelValidationError,
    QrelVersionExistsError,
)
from selfrag.schema import Qrel


def make_qrel(query_id="q1", doc_id="d1", char_start=0, char_end=50, grade=1, judge="human:test"):
    return Qrel(
        qrels_version=999,  # deliberately wrong -- append_version must overwrite this
        query_id=query_id,
        doc_id=doc_id,
        char_start=char_start,
        char_end=char_end,
        grade=grade,
        judge=judge,
    )


def test_append_version_starts_at_one_and_stamps_version(tmp_path):
    store = QrelStore(tmp_path)
    assert store.latest_version() is None

    v1 = store.append_version([make_qrel(doc_id="d1"), make_qrel(doc_id="d2")], note="first batch")

    assert v1 == 1
    assert store.latest_version() == 1
    loaded = store.load(1)
    assert len(loaded) == 2
    assert all(q.qrels_version == 1 for q in loaded)  # stamped, not the caller's stray 999


def test_append_version_increments(tmp_path):
    store = QrelStore(tmp_path)
    v1 = store.append_version([make_qrel(doc_id="d1")], note="first")
    v2 = store.append_version([make_qrel(doc_id="d1"), make_qrel(doc_id="d2")], note="second")
    assert v1 == 1
    assert v2 == 2
    assert store.latest_version() == 2
    assert len(store.load(2)) == 2


def test_load_roundtrip_preserves_fields(tmp_path):
    store = QrelStore(tmp_path)
    original = make_qrel(query_id="q7", doc_id="d42", char_start=10, char_end=99, grade=3, judge="human:mustafa")
    store.append_version([original], note="roundtrip")
    (loaded,) = store.load(1)
    assert loaded.query_id == "q7"
    assert loaded.doc_id == "d42"
    assert loaded.char_start == 10
    assert loaded.char_end == 99
    assert loaded.grade == 3
    assert loaded.judge == "human:mustafa"
    assert loaded.qrels_version == 1


def test_load_missing_version_raises(tmp_path):
    store = QrelStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load(1)


def test_append_version_rejects_empty_batch(tmp_path):
    store = QrelStore(tmp_path)
    with pytest.raises(ValueError):
        store.append_version([], note="empty")


def test_append_version_rejects_duplicate_judgement_key(tmp_path):
    store = QrelStore(tmp_path)
    dup = [
        make_qrel(query_id="q1", doc_id="d1", char_start=0, char_end=50, grade=1),
        make_qrel(query_id="q1", doc_id="d1", char_start=0, char_end=50, grade=2),
    ]
    with pytest.raises(QrelValidationError):
        store.append_version(dup, note="dup")
    # And the failed write must not have left a stray version file behind.
    assert store.latest_version() is None


# --------------------------------------------------------------------------
# The core append-only guarantee: an existing version file can never be overwritten.
# --------------------------------------------------------------------------


def test_cannot_overwrite_existing_version_file(tmp_path, monkeypatch):
    store = QrelStore(tmp_path)
    v1 = store.append_version([make_qrel(doc_id="d1")], note="first")
    assert v1 == 1

    # Simulate the target of the NEXT append already existing on disk --
    # e.g. left over from a previous crashed run that got as far as writing
    # the Parquet file before failing. Since normal use always computes the
    # next version as latest_version()+1, a stray file already sitting at
    # exactly that slot is the one way this situation can arise; we force it
    # deterministically here by pinning latest_version() rather than relying
    # on directory listing order.
    stray = tmp_path / "v2.parquet"
    stray.write_bytes(b"not a real parquet file, but it must not be touched")
    monkeypatch.setattr(store, "latest_version", lambda: 1)

    with pytest.raises(QrelVersionExistsError):
        store.append_version([make_qrel(doc_id="d2")], note="second")

    # The stray file must be untouched -- proving the refusal happened before
    # any write, not that a corrupted write occurred.
    assert stray.read_bytes() == b"not a real parquet file, but it must not be touched"


def test_cannot_overwrite_existing_metadata_sidecar(tmp_path):
    store = QrelStore(tmp_path)
    store.append_version([make_qrel(doc_id="d1")], note="first")

    # Directly collide on the metadata sidecar path instead of the parquet path.
    meta_path = tmp_path / "v2.meta.json"
    meta_path.write_text("{}")

    with pytest.raises(QrelVersionExistsError):
        store.append_version([make_qrel(doc_id="d2")], note="second")


# --------------------------------------------------------------------------
# Sidecar metadata
# --------------------------------------------------------------------------


def test_sidecar_metadata_written_and_readable(tmp_path):
    store = QrelStore(tmp_path)
    qrels = [
        make_qrel(query_id="q1", doc_id="d1", judge="human:alice"),
        make_qrel(query_id="q1", doc_id="d2", judge="human:bob"),
        make_qrel(query_id="q2", doc_id="d1", judge="human:alice"),
    ]
    store.append_version(qrels, note="initial pooling")

    meta_path = tmp_path / "v1.meta.json"
    assert meta_path.exists()
    raw = orjson.loads(meta_path.read_bytes())
    assert raw["qrels_version"] == 1
    assert raw["note"] == "initial pooling"
    assert raw["n_queries"] == 2
    assert raw["n_judgements"] == 3
    assert raw["judges"] == ["human:alice", "human:bob"]
    assert raw["parent_version"] is None
    assert "created_at" in raw and raw["created_at"]

    meta = store.load_meta(1)
    assert meta.qrels_version == 1
    assert meta.n_queries == 2
    assert meta.judges == ["human:alice", "human:bob"]


def test_sidecar_metadata_parent_version_chains(tmp_path):
    store = QrelStore(tmp_path)
    store.append_version([make_qrel(doc_id="d1")], note="v1")
    store.append_version([make_qrel(doc_id="d1"), make_qrel(doc_id="d2")], note="v2")

    assert store.load_meta(1).parent_version is None
    assert store.load_meta(2).parent_version == 1


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------


def test_diff_reports_added_removed_regraded(tmp_path):
    store = QrelStore(tmp_path)
    v1_qrels = [
        make_qrel(query_id="q1", doc_id="dA", char_start=0, char_end=50, grade=1),  # regraded later
        make_qrel(query_id="q1", doc_id="dB", char_start=0, char_end=50, grade=2),  # removed later
        make_qrel(query_id="q1", doc_id="dC", char_start=0, char_end=50, grade=3),  # unchanged
    ]
    store.append_version(v1_qrels, note="v1")

    v2_qrels = [
        make_qrel(query_id="q1", doc_id="dA", char_start=0, char_end=50, grade=3),  # regraded 1 -> 3
        make_qrel(query_id="q1", doc_id="dC", char_start=0, char_end=50, grade=3),  # unchanged
        make_qrel(query_id="q1", doc_id="dD", char_start=0, char_end=50, grade=2),  # newly added
    ]
    store.append_version(v2_qrels, note="v2 correction")

    diff = store.diff(1, 2)
    assert diff.v_a == 1
    assert diff.v_b == 2
    assert diff.added == [("q1", "dD", 0, 50)]
    assert diff.removed == [("q1", "dB", 0, 50)]
    assert diff.regraded == [(("q1", "dA", 0, 50), 1, 3)]
    assert diff.n_unchanged == 1  # dC


# --------------------------------------------------------------------------
# Lookup helpers
# --------------------------------------------------------------------------


def test_for_query_and_queries_with_judgements(tmp_path):
    store = QrelStore(tmp_path)
    qrels = [
        make_qrel(query_id="q1", doc_id="d1"),
        make_qrel(query_id="q1", doc_id="d2"),
        make_qrel(query_id="q2", doc_id="d1"),
    ]
    store.append_version(qrels, note="init")

    q1_judgements = store.for_query(1, "q1")
    assert {q.doc_id for q in q1_judgements} == {"d1", "d2"}

    q3_judgements = store.for_query(1, "q3")
    assert q3_judgements == []

    assert store.queries_with_judgements(1) == {"q1", "q2"}


def test_group_by_query(tmp_path):
    store = QrelStore(tmp_path)
    qrels = [
        make_qrel(query_id="q1", doc_id="d1"),
        make_qrel(query_id="q2", doc_id="d1"),
        make_qrel(query_id="q2", doc_id="d2"),
    ]
    store.append_version(qrels, note="init")
    grouped = store.group_by_query(1)
    assert set(grouped) == {"q1", "q2"}
    assert len(grouped["q1"]) == 1
    assert len(grouped["q2"]) == 2

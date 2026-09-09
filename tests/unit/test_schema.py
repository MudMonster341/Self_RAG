"""Tests for selfrag.schema -- the core Pydantic models.

Focused on the invariants the models enforce themselves (span validation,
grade bounds, run_id determinism) rather than re-testing Pydantic's own
type coercion.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from selfrag.schema import (
    Chunk,
    Constraints,
    Document,
    Namespace,
    Qrel,
    Query,
    QuerySource,
    QueryStratum,
    RunManifest,
    Split,
)


def _make_chunk(**overrides):
    defaults = dict(
        chunk_uid="a" * 64,
        doc_id="2401.01234",
        chunker_config_id="fixed@abcd1234",
        char_start=0,
        char_end=10,
        text="0123456789",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


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


class TestChunk:
    def test_valid_span_ok(self):
        chunk = _make_chunk()
        assert chunk.char_end - chunk.char_start == 10

    def test_empty_span_rejected(self):
        with pytest.raises(ValidationError):
            _make_chunk(char_start=5, char_end=5)

    def test_inverted_span_rejected(self):
        with pytest.raises(ValidationError):
            _make_chunk(char_start=10, char_end=5)

    def test_embedding_text_without_prefix_is_plain_text(self):
        chunk = _make_chunk(text="hello world")
        assert chunk.embedding_text() == "hello world"

    def test_embedding_text_with_prefix_prepends_it(self):
        chunk = _make_chunk(text="hello world", context_prefix="Section 3: Method")
        assert chunk.embedding_text() == "Section 3: Method\n\nhello world"


class TestDocument:
    def test_is_live_when_not_tombstoned(self):
        doc = Document(doc_id="2401.01234", namespace=Namespace.PAPERS, source_url="https://x")
        assert doc.is_live is True

    def test_is_not_live_when_tombstoned(self):
        from datetime import UTC, datetime

        doc = Document(
            doc_id="2401.01234",
            namespace=Namespace.PAPERS,
            source_url="https://x",
            tombstoned_at=datetime.now(UTC),
        )
        assert doc.is_live is False


class TestQrel:
    def _make(self, **overrides):
        defaults = dict(
            qrels_version=1,
            query_id="q1",
            doc_id="2401.01234",
            char_start=0,
            char_end=100,
            grade=2,
            judge="human:mustafa",
        )
        defaults.update(overrides)
        return Qrel(**defaults)

    def test_valid_grade_range(self):
        for grade in (0, 1, 2, 3):
            assert self._make(grade=grade).grade == grade

    def test_grade_above_max_rejected(self):
        with pytest.raises(ValidationError):
            self._make(grade=4)

    def test_grade_below_min_rejected(self):
        with pytest.raises(ValidationError):
            self._make(grade=-1)

    def test_empty_span_rejected(self):
        with pytest.raises(ValidationError):
            self._make(char_start=5, char_end=5)

    def test_inverted_span_rejected(self):
        with pytest.raises(ValidationError):
            self._make(char_start=10, char_end=5)


class TestQueryStratum:
    def test_unanswerable_strata_flagged(self):
        assert QueryStratum.UNANSWERABLE_COUNTERFACTUAL.is_unanswerable is True
        assert QueryStratum.UNANSWERABLE_FALSE_PREMISE.is_unanswerable is True
        assert QueryStratum.UNANSWERABLE_TEMPORAL.is_unanswerable is True
        assert QueryStratum.UNANSWERABLE_OFF_TOPIC.is_unanswerable is True

    def test_answerable_strata_not_flagged(self):
        assert QueryStratum.SINGLE_HOP.is_unanswerable is False
        assert QueryStratum.MULTI_HOP.is_unanswerable is False
        assert QueryStratum.COMPARATIVE.is_unanswerable is False
        assert QueryStratum.DEFINITIONAL.is_unanswerable is False
        assert QueryStratum.NEGATION.is_unanswerable is False


class TestQuery:
    def test_minimal_construction(self):
        q = Query(
            query_id="q1",
            text="what is retrieval augmented generation?",
            namespace=Namespace.PAPERS,
            stratum=QueryStratum.SINGLE_HOP,
            source=QuerySource.HUMAN,
            split=Split.DEV,
        )
        assert q.split == Split.DEV


class TestConstraints:
    def test_defaults_are_sane(self):
        c = Constraints()
        assert c.max_p95_latency_ms > 0
        assert c.max_peak_rss_mb > 0
        assert c.max_cost_per_query_usd > 0


class TestRunManifest:
    def test_run_id_is_deterministic(self):
        m1 = _make_manifest()
        m2 = _make_manifest()
        assert m1.run_id() == m2.run_id()

    def test_run_id_changes_with_any_field(self):
        base = _make_manifest().run_id()
        assert _make_manifest(chunker_config_id="fixed@ffffffff").run_id() != base
        assert _make_manifest(qrels_version=2).run_id() != base
        assert _make_manifest(split=Split.TEST).run_id() != base
        assert _make_manifest(seed=1).run_id() != base

    def test_run_id_reflects_constraints(self):
        base = _make_manifest().run_id()
        tighter = _make_manifest(constraints=Constraints(max_p95_latency_ms=100.0)).run_id()
        assert base != tighter

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            RunManifest(chunker_config_id="fixed@abcd1234")

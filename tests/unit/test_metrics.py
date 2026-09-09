"""Tests for selfrag.eval.metrics.

Covers: overlap-rule geometry, the guard rail that a query with zero
relevant gold spans returns None (not 0.0), a hand-computed nDCG value, a
known-good-vs-known-bad ranking ordering check, and the core design claim
that qrels are chunker-independent (the same qrel scores correctly against
two entirely different chunk boundary sets over the same document).
"""

from __future__ import annotations

import math

import pytest

from selfrag.eval.metrics import (
    RANKED_METRICS,
    OverlapRule,
    OverlapSpec,
    SpanRelevance,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    spans_overlap,
    success_at_k,
)
from selfrag.schema import Qrel

IOU_01 = OverlapSpec(rule=OverlapRule.IOU, threshold=0.1)
MIDPOINT = OverlapSpec(rule=OverlapRule.MIDPOINT, threshold=0.1)


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


# --------------------------------------------------------------------------
# OverlapSpec / spans_overlap geometry
# --------------------------------------------------------------------------


def test_overlap_spec_rejects_bad_iou_threshold():
    with pytest.raises(ValueError):
        OverlapSpec(rule=OverlapRule.IOU, threshold=0.0)
    with pytest.raises(ValueError):
        OverlapSpec(rule=OverlapRule.IOU, threshold=1.5)


def test_iou_overlap_exact_threshold_boundary():
    # chunk [0,100), gold [50,150) -> intersection [50,100)=50, union=150 -> iou=1/3
    chunk = (0, 100)
    gold = (50, 150)
    at_threshold = OverlapSpec(rule=OverlapRule.IOU, threshold=1 / 3)
    above_threshold = OverlapSpec(rule=OverlapRule.IOU, threshold=1 / 3 + 1e-9)
    assert spans_overlap(chunk, gold, at_threshold) is True
    assert spans_overlap(chunk, gold, above_threshold) is False


def test_iou_disjoint_spans_never_overlap():
    chunk = (0, 50)
    gold = (50, 100)  # touching but not overlapping (half-open)
    assert spans_overlap(chunk, gold, OverlapSpec(rule=OverlapRule.IOU, threshold=0.01)) is False


def test_midpoint_rule_contains_midpoint():
    gold = (100, 200)  # midpoint 150
    assert spans_overlap((140, 160), gold, MIDPOINT) is True
    assert spans_overlap((0, 100), gold, MIDPOINT) is False  # doesn't reach the midpoint at all
    assert spans_overlap((150, 300), gold, MIDPOINT) is True  # midpoint is inclusive on the left edge


def test_midpoint_rule_ignores_iou_style_partial_overlap():
    # A chunk that overlaps 99% of the gold span but not its midpoint is NOT relevant under MIDPOINT.
    gold = (0, 100)  # midpoint 50
    chunk = (50, 200)  # overlaps [50,100) of gold but starts exactly at the midpoint (inclusive -> True)
    assert spans_overlap(chunk, gold, MIDPOINT) is True
    chunk2 = (51, 200)  # now strictly past the midpoint
    assert spans_overlap(chunk2, gold, MIDPOINT) is False


# --------------------------------------------------------------------------
# SpanRelevance
# --------------------------------------------------------------------------


def test_span_relevance_rejects_mixed_query_ids():
    qrels = [make_qrel(query_id="q1"), make_qrel(query_id="q2")]
    with pytest.raises(ValueError):
        SpanRelevance(qrels, IOU_01)


def test_span_relevance_picks_max_grade_among_overlapping_gold():
    qrels = [
        make_qrel(doc_id="d1", char_start=0, char_end=100, grade=1),
        make_qrel(doc_id="d1", char_start=40, char_end=140, grade=3),
    ]
    rel = SpanRelevance(qrels, OverlapSpec(rule=OverlapRule.IOU, threshold=0.05))
    # A chunk overlapping both gold spans should report the higher grade.
    assert rel.relevance_for("d1", 50, 90) == 3


def test_span_relevance_zero_for_unjudged_doc():
    qrels = [make_qrel(doc_id="d1")]
    rel = SpanRelevance(qrels, IOU_01)
    assert rel.relevance_for("d2", 0, 50) == 0


# --------------------------------------------------------------------------
# Guard rail: zero relevant gold spans -> None, never 0.0
# --------------------------------------------------------------------------


def test_zero_gold_spans_returns_none_not_zero():
    ranked = [("d1", 0, 50), ("d2", 0, 50)]
    for fn in RANKED_METRICS.values():
        assert fn([], ranked, 10, IOU_01) is None


def test_all_negative_judgements_returns_none():
    """Qrels can exist (fully judged) yet have zero *relevant* items -- grade 0 rows
    are explicit negative judgements, not "unjudged", and must not be treated
    as a real recall/precision/ndcg denominator of zero relevant items."""
    qrels = [make_qrel(doc_id="d1", grade=0), make_qrel(doc_id="d2", grade=0)]
    ranked = [("d1", 0, 50), ("d2", 0, 50)]
    for fn in RANKED_METRICS.values():
        assert fn(qrels, ranked, 10, IOU_01) is None


def test_relevant_but_missed_ranking_is_a_real_zero_not_none():
    """Contrast with the guard rail: when relevant items DO exist but the ranking
    fails to surface them in top-k, that is a genuine 0.0 (recall/success) or
    computed low nDCG -- not a None."""
    qrels = [make_qrel(doc_id="d1", grade=2)]
    ranked = [("d2", 0, 50), ("d3", 0, 50)]  # d1 never retrieved
    assert recall_at_k(qrels, ranked, 2, IOU_01) == 0.0
    assert success_at_k(qrels, ranked, 2, IOU_01) == 0.0
    assert mrr(qrels, ranked, 2, IOU_01) == 0.0
    assert ndcg_at_k(qrels, ranked, 2, IOU_01) == 0.0


# --------------------------------------------------------------------------
# Hand-computed nDCG
# --------------------------------------------------------------------------


def test_ndcg_hand_computed():
    qrels = [
        make_qrel(doc_id="d1", char_start=0, char_end=50, grade=3),
        make_qrel(doc_id="d2", char_start=0, char_end=50, grade=2),
        make_qrel(doc_id="d3", char_start=0, char_end=50, grade=1),
    ]
    ranked = [
        ("d2", 0, 50),  # rank 1, grade 2
        ("d1", 0, 50),  # rank 2, grade 3
        ("d4", 0, 50),  # rank 3, grade 0 (not a gold doc at all)
        ("d3", 0, 50),  # rank 4, grade 1
    ]

    result = ndcg_at_k(qrels, ranked, 4, IOU_01)

    # Independent hand computation: DCG at ranks 1..4 with grades [2,3,0,1],
    # discount log2(rank+1); IDCG from grades sorted descending [3,2,1].
    dcg = 2 / math.log2(2) + 3 / math.log2(3) + 0 / math.log2(4) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    expected = dcg / idcg

    assert expected == pytest.approx(0.90794, abs=5e-5)  # sanity-pin the hand arithmetic itself
    assert result == pytest.approx(expected, rel=1e-12)


def test_ndcg_truncates_ideal_ranking_to_k():
    # 3 relevant docs, but k=1 -- IDCG should use only the single best grade.
    qrels = [
        make_qrel(doc_id="d1", grade=3),
        make_qrel(doc_id="d2", grade=2),
        make_qrel(doc_id="d3", grade=1),
    ]
    ranked = [("d1", 0, 50)]  # the best possible single result
    result = ndcg_at_k(qrels, ranked, 1, IOU_01)
    assert result == pytest.approx(1.0)  # perfect nDCG@1: retrieved exactly the top-grade doc


# --------------------------------------------------------------------------
# Known-good ranking beats known-bad ranking
# --------------------------------------------------------------------------


def test_known_good_ranking_beats_known_bad_ranking():
    qrels = [
        make_qrel(doc_id="d1", grade=3),
        make_qrel(doc_id="d2", grade=2),
    ]
    good_ranking = [("d1", 0, 50), ("d2", 0, 50), ("d3", 0, 50), ("d4", 0, 50)]
    bad_ranking = [("d3", 0, 50), ("d4", 0, 50), ("d1", 0, 50), ("d2", 0, 50)]

    for k in (2, 4):
        good_ndcg = ndcg_at_k(qrels, good_ranking, k, IOU_01)
        bad_ndcg = ndcg_at_k(qrels, bad_ranking, k, IOU_01)
        assert good_ndcg > bad_ndcg

    assert recall_at_k(qrels, good_ranking, 2, IOU_01) == 1.0
    assert recall_at_k(qrels, bad_ranking, 2, IOU_01) == 0.0

    assert precision_at_k(qrels, good_ranking, 2, IOU_01) > precision_at_k(qrels, bad_ranking, 2, IOU_01)
    assert mrr(qrels, good_ranking, 4, IOU_01) > mrr(qrels, bad_ranking, 4, IOU_01)


# --------------------------------------------------------------------------
# Core design claim: chunking changes never invalidate qrels
# --------------------------------------------------------------------------


def test_chunker_change_does_not_invalidate_qrels():
    """The core design claim of span-based qrels: a SINGLE qrel, never touched
    or re-judged, scores correctly against candidates from two completely
    different chunkers (different boundary sets) over the same document.
    This is what makes re-chunking a free ablation instead of a re-judging
    project."""
    doc_id = "2401.00001"
    gold = make_qrel(query_id="q1", doc_id=doc_id, char_start=100, char_end=200, grade=2)

    overlap = OverlapSpec(rule=OverlapRule.MIDPOINT, threshold=0.1)
    rel = SpanRelevance([gold], overlap)

    # Chunker A (e.g. chunker_config_id="v1-fixed-512"): one large chunk.
    chunker_a_chunk = (doc_id, 0, 300)
    assert rel.relevance_for(*chunker_a_chunk) == 2

    # Chunker B (a LATER, entirely different chunker version -- different
    # boundaries, same underlying document text): the gold span's midpoint
    # (150) now falls in a different, smaller chunk.
    chunker_b_chunks = [(doc_id, 0, 120), (doc_id, 120, 250), (doc_id, 250, 400)]
    assert rel.relevance_for(*chunker_b_chunks[0]) == 0
    assert rel.relevance_for(*chunker_b_chunks[1]) == 2  # contains midpoint 150
    assert rel.relevance_for(*chunker_b_chunks[2]) == 0

    # Also verify end-to-end through a real ranked-list metric, not just the
    # low-level relevance lookup: recall@k is 1.0 under EITHER chunking.
    assert recall_at_k([gold], [chunker_a_chunk], 1, overlap) == 1.0
    assert recall_at_k([gold], chunker_b_chunks, 3, overlap) == 1.0


def test_chunker_change_with_iou_rule_also_preserved():
    """Same claim, under the IOU rule instead of MIDPOINT, to show it holds
    regardless of which pre-registered overlap rule is in effect."""
    doc_id = "d1"
    gold = make_qrel(query_id="q1", doc_id=doc_id, char_start=0, char_end=100, grade=1)
    overlap = OverlapSpec(rule=OverlapRule.IOU, threshold=0.5)

    # A chunker whose boundaries align exactly with the gold span.
    aligned_chunk = (doc_id, 0, 100)
    # A different chunker version with a chunk that still overlaps enough.
    shifted_chunk = (doc_id, 0, 120)  # intersection 100, union 120 -> iou = 0.833

    assert spans_overlap((0, 100), (0, 100), overlap) is True
    assert spans_overlap((0, 120), (0, 100), overlap) is True
    assert recall_at_k([gold], [aligned_chunk], 1, overlap) == 1.0
    assert recall_at_k([gold], [shifted_chunk], 1, overlap) == 1.0


# --------------------------------------------------------------------------
# k validation
# --------------------------------------------------------------------------


def test_k_must_be_positive():
    qrels = [make_qrel()]
    ranked = [("d1", 0, 50)]
    with pytest.raises(ValueError):
        recall_at_k(qrels, ranked, 0, IOU_01)
    with pytest.raises(ValueError):
        ndcg_at_k(qrels, ranked, -1, IOU_01)


def test_precision_divides_by_k_not_by_retrieved_count():
    qrels = [make_qrel(doc_id="d1", grade=1)]
    ranked = [("d1", 0, 50)]  # only one candidate retrieved, but k asks for 4
    # 1 relevant hit out of a requested k=4 -- precision should be 1/4, not 1/1.
    assert precision_at_k(qrels, ranked, 4, IOU_01) == pytest.approx(0.25)

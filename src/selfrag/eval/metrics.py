"""Deterministic IR metrics computed against span-based relevance judgements.

Every metric here is a *per-query* value. Means, medians, and other
aggregates across queries are deliberately left to the caller (typically
``stats.py``), because averaging is a policy choice (macro vs. micro, which
queries to include) and this module should never make that choice silently.

Relevance is defined over document character spans, never chunk ids -- see
``selfrag.schema.Qrel`` and ``selfrag.ids.chunk_uid`` for why: a chunk id
encodes ``chunker_config_id``, so a qrel keyed to a chunk id would be
invalidated by the very first chunker ablation. Instead, relevance is
recomputed fresh, at scoring time, by overlapping the retrieved chunk's
``[char_start, char_end)`` span against the gold span in the same
``doc_id``, under a pre-registered overlap rule. This is what lets
``selfrag.eval.runfile.rescore`` re-score old runs against corrected qrels,
or score runs from an entirely different chunker, without re-judging
anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from selfrag.schema import Qrel

# A retrieved candidate, in rank order (index 0 == rank 1): (doc_id, char_start, char_end).
RetrievedSpan = tuple[str, int, int]


class OverlapRule(StrEnum):
    """How a retrieved chunk's span is judged to overlap a gold span.

    Exactly two options, on purpose -- a third "pick whatever looks close"
    rule would make relevance non-reproducible across scoring runs.
    """

    IOU = "iou"
    MIDPOINT = "midpoint"


# Documented default for IOU's threshold. This is a constant callers may
# reference explicitly (e.g. ``OverlapSpec(OverlapRule.IOU, DEFAULT_IOU_THRESHOLD)``);
# it is intentionally *not* a default value on any function parameter below,
# because "the overlap rule and threshold used" is exactly the kind of
# methodological choice that must be stated at every call site, not inherited
# silently from a signature default that could change under callers' feet.
DEFAULT_IOU_THRESHOLD = 0.1


@dataclass(frozen=True)
class OverlapSpec:
    """A pre-registered overlap rule, required at every call site.

    Both fields are mandatory (no defaults) so that scoring code cannot
    forget to state which rule produced its relevance judgements. Store this
    alongside any computed metrics (``rescore`` does) so a result can always
    be traced back to the rule that produced it.
    """

    rule: OverlapRule
    threshold: float

    def __post_init__(self) -> None:
        if self.rule == OverlapRule.IOU and not (0.0 < self.threshold <= 1.0):
            raise ValueError(f"IOU threshold must be in (0, 1], got {self.threshold}")

    def to_dict(self) -> dict[str, str | float]:
        return {"rule": self.rule.value, "threshold": self.threshold}


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = min(a_end, b_end) - max(a_start, b_start)
    if inter <= 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _midpoint_contained(chunk_start: int, chunk_end: int, gold_start: int, gold_end: int) -> bool:
    mid = (gold_start + gold_end) / 2.0
    return chunk_start <= mid < chunk_end


def spans_overlap(
    chunk_span: tuple[int, int], gold_span: tuple[int, int], overlap: OverlapSpec
) -> bool:
    """Whether a retrieved chunk's span counts as covering a gold span.

    Both spans must already be in the same ``doc_id`` -- this function only
    compares offsets, it does not know about document identity.
    """
    c_start, c_end = chunk_span
    g_start, g_end = gold_span
    if overlap.rule == OverlapRule.IOU:
        return _iou(c_start, c_end, g_start, g_end) >= overlap.threshold
    if overlap.rule == OverlapRule.MIDPOINT:
        return _midpoint_contained(c_start, c_end, g_start, g_end)
    raise ValueError(f"unknown overlap rule: {overlap.rule!r}")  # pragma: no cover - exhaustive enum


class SpanRelevance:
    """Graded relevance of retrieved spans against one query's qrels.

    This is the seam that makes qrels chunker-independent: it takes the gold
    judgements as document-space spans and, for each retrieved candidate
    (also a document-space span), decides on the fly whether the two
    overlap. Re-chunking a document just changes what candidates get handed
    in here; the qrels themselves never need to move.
    """

    def __init__(self, qrels: Sequence[Qrel], overlap: OverlapSpec) -> None:
        qids = {q.query_id for q in qrels}
        if len(qids) > 1:
            raise ValueError(
                f"SpanRelevance expects qrels for exactly one query, got query_ids={sorted(qids)}"
            )
        self._qrels = list(qrels)
        self._overlap = overlap
        self._by_doc: dict[str, list[Qrel]] = {}
        for q in self._qrels:
            self._by_doc.setdefault(q.doc_id, []).append(q)

    @property
    def qrels(self) -> list[Qrel]:
        return list(self._qrels)

    def relevance_for(self, doc_id: str, char_start: int, char_end: int) -> int:
        """Graded relevance of one retrieved span.

        Returns the maximum grade among gold spans (in the same ``doc_id``)
        that overlap this span under the registered rule, or 0 if none do
        (including the case where the document has no gold judgements at
        all).
        """
        best = 0
        for q in self._by_doc.get(doc_id, ()):
            if spans_overlap((char_start, char_end), (q.char_start, q.char_end), self._overlap):
                best = max(best, q.grade)
        return best

    def relevance_vector(self, ranked: Sequence[RetrievedSpan]) -> list[int]:
        """Graded relevance for a whole ranked list, in rank order."""
        return [self.relevance_for(doc_id, cs, ce) for doc_id, cs, ce in ranked]


def _relevant_qrels(qrels: Sequence[Qrel]) -> list[Qrel]:
    """Gold judgements that count as actually relevant (grade > 0).

    A qrel with grade 0 is an explicit negative judgement (something was
    pooled and judged not relevant); it must not count as "a relevant item
    to recall" or the denominators below would be wrong.
    """
    return [q for q in qrels if q.grade > 0]


def _require_positive_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(
    qrels: Sequence[Qrel], ranked: Sequence[RetrievedSpan], k: int, overlap: OverlapSpec
) -> float | None:
    """Fraction of gold-relevant spans covered by at least one retrieved span in the top-k.

    Returns ``None`` -- not 0.0 -- when the query has zero relevant gold
    spans. This is a guard rail, not an edge case to special-case away: a
    query with no known-relevant span usually means judging/pooling is
    incomplete for that query, not that the retriever failed a real target.
    Silently scoring it as 0.0 would drag the mean down for a reason that has
    nothing to do with retrieval quality; returning ``None`` lets the caller
    exclude it from the mean instead.
    """
    _require_positive_k(k)
    relevant = _relevant_qrels(qrels)
    if not relevant:
        return None
    topk = ranked[:k]
    covered = sum(
        1
        for q in relevant
        if any(
            doc_id == q.doc_id and spans_overlap((cs, ce), (q.char_start, q.char_end), overlap)
            for doc_id, cs, ce in topk
        )
    )
    return covered / len(relevant)


def precision_at_k(
    qrels: Sequence[Qrel], ranked: Sequence[RetrievedSpan], k: int, overlap: OverlapSpec
) -> float | None:
    """Fraction of the top-k retrieved spans that are relevant.

    Divides by ``k`` (the classic definition), not by the number actually
    retrieved: asking for k and receiving fewer candidates is itself a
    retrieval shortfall and should be reflected in the score, not smoothed
    away by shrinking the denominator.

    Returns ``None`` when the query has zero relevant gold spans (see
    ``recall_at_k`` for why that is a guard rail, not a 0.0).
    """
    _require_positive_k(k)
    relevant = _relevant_qrels(qrels)
    if not relevant:
        return None
    rel = SpanRelevance(qrels, overlap)
    topk = ranked[:k]
    hits = sum(1 for doc_id, cs, ce in topk if rel.relevance_for(doc_id, cs, ce) > 0)
    return hits / k


def mrr(
    qrels: Sequence[Qrel], ranked: Sequence[RetrievedSpan], k: int, overlap: OverlapSpec
) -> float | None:
    """Reciprocal rank of the first relevant span within the top-k window (``mrr@k``).

    0.0 (a real value, not a guard rail) if a relevant span exists but does
    not appear anywhere in the top-k. ``None`` only when the query has zero
    relevant gold spans at all -- there is nothing to rank for.
    """
    _require_positive_k(k)
    relevant = _relevant_qrels(qrels)
    if not relevant:
        return None
    rel = SpanRelevance(qrels, overlap)
    for idx, (doc_id, cs, ce) in enumerate(ranked[:k], start=1):
        if rel.relevance_for(doc_id, cs, ce) > 0:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(
    qrels: Sequence[Qrel], ranked: Sequence[RetrievedSpan], k: int, overlap: OverlapSpec
) -> float | None:
    """Normalised discounted cumulative gain at k, standard log2 discount.

    Gain is **linear** (``gain = grade``), the original Jarvelin-Kekalainen
    formulation. Much of the IR literature -- and several popular toolkits --
    instead use exponential gain (``2**grade - 1``), which weights grade-3
    spans far more heavily. The two conventions are not comparable.

    This matters in one specific way: internal config-vs-config comparisons are
    unaffected, because the convention is held constant across every run. But
    an nDCG@10 reported here must NOT be placed side by side with an nDCG@10
    quoted from a paper without first checking which gain that paper used.
    Record the convention whenever a number leaves this repo.

    IDCG is computed from the gold grades themselves (sorted descending,
    truncated to k) -- the ideal ranking places every relevant span as early
    as retrieval allows, regardless of what was actually retrieved.

    Returns ``None`` when the query has zero relevant gold spans (IDCG would
    be 0, making the ratio undefined -- see ``recall_at_k`` for the guard
    rail rationale).
    """
    _require_positive_k(k)
    relevant = _relevant_qrels(qrels)
    if not relevant:
        return None
    rel = SpanRelevance(qrels, overlap)
    topk = ranked[:k]
    dcg = sum(
        rel.relevance_for(doc_id, cs, ce) / math.log2(idx + 1)
        for idx, (doc_id, cs, ce) in enumerate(topk, start=1)
    )
    ideal_grades = sorted((q.grade for q in relevant), reverse=True)[:k]
    idcg = sum(grade / math.log2(idx + 1) for idx, grade in enumerate(ideal_grades, start=1))
    if idcg <= 0:
        return None  # unreachable given `relevant` is non-empty, but keep the ratio safe
    return dcg / idcg


def success_at_k(
    qrels: Sequence[Qrel], ranked: Sequence[RetrievedSpan], k: int, overlap: OverlapSpec
) -> float | None:
    """1.0 if any relevant span appears in the top-k, else 0.0 ("hit rate").

    Returns ``None`` when the query has zero relevant gold spans (see
    ``recall_at_k``).
    """
    _require_positive_k(k)
    relevant = _relevant_qrels(qrels)
    if not relevant:
        return None
    rel = SpanRelevance(qrels, overlap)
    topk = ranked[:k]
    return 1.0 if any(rel.relevance_for(doc_id, cs, ce) > 0 for doc_id, cs, ce in topk) else 0.0


RankedMetricFn = Callable[[Sequence[Qrel], Sequence[RetrievedSpan], int, OverlapSpec], "float | None"]

RANKED_METRICS: Mapping[str, RankedMetricFn] = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr,
    "ndcg": ndcg_at_k,
    "success": success_at_k,
}

"""Evaluation spine: span-based relevance, versioned qrels, persisted run files, and paired stats.

Scoring in this system is built to be a pure function of (run file, qrels):

    from selfrag.eval.qrels import QrelStore
    from selfrag.eval.runfile import rescore
    from selfrag.eval.metrics import OverlapSpec, OverlapRule

    store = QrelStore("data/eval/qrels")
    table = rescore(
        "data/eval/runs/my_run.parquet",
        store,
        qrels_version=store.latest_version(),
        metrics=["ndcg@10", "recall@20"],
        overlap=OverlapSpec(rule=OverlapRule.IOU, threshold=0.1),
    )

See ``selfrag.eval.metrics`` for why relevance is defined over document
character spans rather than chunk ids, ``selfrag.eval.qrels`` for why qrels
are versioned and append-only, ``selfrag.eval.runfile`` for why runs are
persisted at all, and ``selfrag.eval.stats`` for why comparisons are always
paired and gated by a pre-registered minimum effect size.
"""

from __future__ import annotations

"""TREC-style persisted run files, and rescoring them against qrels.

A run file is the recorded output of one retrieval configuration against one
query set: for every query, the ranked list of retrieved chunks with their
scores. Persisting it -- instead of scoring immediately and discarding it --
is what makes scoring a *pure function of (run file, qrels)*:

    metrics = rescore(run_path, qrels_store, qrels_version, metric_specs, overlap)

``rescore`` needs no index, no corpus, no embedder, no model. It only reads
two files. This matters operationally: when a qrels correction lands (a new
``QrelStore`` version), every past run -- weeks of retrieval experiments --
can be rescored against the correction in seconds, instead of re-running
retrieval for all of them. The run file is the boundary that makes that
possible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from selfrag.eval.metrics import RANKED_METRICS, OverlapSpec, RetrievedSpan
from selfrag.eval.qrels import QrelStore
from selfrag.schema import RunManifest

_RUN_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("doc_id", pa.string()),
        ("chunk_uid", pa.string()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("rank", pa.int64()),
        ("score", pa.float64()),
        ("run_id", pa.string()),
    ]
)


def _manifest_path(run_path: Path) -> Path:
    """The manifest sidecar for a run file: ``foo.parquet`` -> ``foo.manifest.json``."""
    return run_path.with_suffix("").with_suffix(".manifest.json")


def _read_manifest(run_path: Path) -> RunManifest:
    mpath = _manifest_path(run_path)
    if not mpath.exists():
        raise FileNotFoundError(
            f"run file {run_path} has no manifest sidecar at {mpath} -- "
            "a run file is not self-describing without it"
        )
    return RunManifest(**orjson.loads(mpath.read_bytes()))


class RunFileWriter:
    """Streaming Parquet writer for one run file.

    Rows are buffered in small batches and flushed to a ``pyarrow``
    ``ParquetWriter`` rather than accumulated in a single Python list for
    the whole run: a run over a large query set (thousands of queries times
    top-k retrieved chunks each) should never need to hold the entire run
    in memory just to write it out.

    Use as a context manager so the manifest sidecar is only written once
    the run completes successfully:

        with RunFileWriter(path, manifest) as w:
            for query in queries:
                w.write_ranked_query(query.query_id, retrieved)
    """

    def __init__(self, path: str | Path, manifest: RunManifest, batch_size: int = 4096) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.batch_size = batch_size
        if self.path.exists():
            raise FileExistsError(
                f"run file already exists at {self.path} -- write a new path rather than "
                "overwriting a previously recorded run"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._run_id = manifest.run_id()
        self._writer: pq.ParquetWriter | None = pq.ParquetWriter(str(self.path), _RUN_SCHEMA)
        self._buffer: list[dict[str, object]] = []
        self._closed = False
        self.n_rows = 0

    def write_row(
        self,
        *,
        query_id: str,
        doc_id: str,
        chunk_uid: str,
        char_start: int,
        char_end: int,
        rank: int,
        score: float,
    ) -> None:
        """Append one (query, retrieved chunk) row."""
        if self._closed:
            raise RuntimeError("cannot write to a closed RunFileWriter")
        if char_end <= char_start:
            raise ValueError(f"empty or inverted span [{char_start}, {char_end}) for {chunk_uid!r}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self._buffer.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "chunk_uid": chunk_uid,
                "char_start": char_start,
                "char_end": char_end,
                "rank": rank,
                "score": float(score),
                "run_id": self._run_id,
            }
        )
        self.n_rows += 1
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def write_ranked_query(
        self, query_id: str, retrieved: Sequence[tuple[str, str, int, int, float]]
    ) -> None:
        """Append a full ranked list for one query.

        ``retrieved`` is ``(doc_id, chunk_uid, char_start, char_end, score)``
        tuples, already sorted best-first; rank is assigned as 1..N from
        list order.
        """
        for i, (doc_id, chunk_uid, char_start, char_end, score) in enumerate(retrieved, start=1):
            self.write_row(
                query_id=query_id,
                doc_id=doc_id,
                chunk_uid=chunk_uid,
                char_start=char_start,
                char_end=char_end,
                rank=i,
                score=score,
            )

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=_RUN_SCHEMA)
        assert self._writer is not None
        self._writer.write_table(table)
        self._buffer = []

    def close(self) -> None:
        """Flush remaining rows, finalize the Parquet file, and write the manifest sidecar."""
        if self._closed:
            return
        self._flush()
        assert self._writer is not None
        self._writer.close()
        self._writer = None
        self._closed = True
        manifest_path = _manifest_path(self.path)
        manifest_path.write_bytes(
            orjson.dumps(self.manifest.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )

    def __enter__(self) -> RunFileWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        if exc_type is None:
            self.close()
        elif not self._closed:
            # A run that failed partway through should not end up looking
            # self-describing: close the underlying Parquet writer so the
            # file handle isn't leaked, but deliberately skip writing the
            # manifest sidecar so a partial file is never mistaken for a
            # complete, scoreable run.
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._closed = True


@dataclass(frozen=True)
class RunFile:
    """A run file loaded back from disk, paired with its manifest."""

    path: Path
    manifest: RunManifest
    table: pa.Table

    def query_ids(self) -> list[str]:
        return sorted(set(self.table.column("query_id").to_pylist()))

    def group_by_query(self) -> dict[str, list[RetrievedSpan]]:
        """Every query's ranked candidates as ``(doc_id, char_start, char_end)``, rank-ordered.

        Sorts the whole table once by ``(query_id, rank)`` rather than
        filtering per query, so scoring an entire run costs one sort instead
        of one scan per query.
        """
        sorted_table = self.table.sort_by([("query_id", "ascending"), ("rank", "ascending")])
        query_ids = sorted_table.column("query_id").to_pylist()
        doc_ids = sorted_table.column("doc_id").to_pylist()
        starts = sorted_table.column("char_start").to_pylist()
        ends = sorted_table.column("char_end").to_pylist()
        out: dict[str, list[RetrievedSpan]] = {}
        for qid, doc_id, cs, ce in zip(query_ids, doc_ids, starts, ends, strict=True):
            out.setdefault(qid, []).append((doc_id, cs, ce))
        return out


def read_run(path: str | Path) -> RunFile:
    """Load a run file and its manifest sidecar."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no run file at {path}")
    table = pq.read_table(path)
    manifest = _read_manifest(path)
    return RunFile(path=path, manifest=manifest, table=table)


def _parse_metric_spec(spec: str) -> tuple[str, int]:
    """Parse ``"ndcg@10"`` -> ``("ndcg", 10)``.

    The ``@k`` suffix is required, not defaulted, for the same reason the
    overlap rule is required everywhere: which cutoff produced a number is
    part of what the number means, and it must be visible at the call site
    rather than inherited from an implicit default.
    """
    if "@" not in spec:
        raise ValueError(
            f"metric spec {spec!r} must be of the form 'name@k' (e.g. 'ndcg@10'), "
            "the cutoff k is never implied"
        )
    name, _, k_str = spec.partition("@")
    if name not in RANKED_METRICS:
        raise ValueError(f"unknown metric {name!r}; available: {sorted(RANKED_METRICS)}")
    try:
        k = int(k_str)
    except ValueError as e:
        raise ValueError(f"metric spec {spec!r} has a non-integer cutoff {k_str!r}") from e
    return name, k


def rescore(
    run_path: str | Path,
    qrels_store: QrelStore,
    qrels_version: int,
    metrics: Sequence[str],
    overlap: OverlapSpec,
) -> pa.Table:
    """Score a persisted run file against one qrels version. Pure in (run file, qrels).

    ``metrics`` is a sequence of ``"name@k"`` specs (e.g. ``["ndcg@10",
    "recall@20"]``) resolved against ``selfrag.eval.metrics.RANKED_METRICS``.
    ``qrels_version`` need not match whatever qrels version the run's own
    manifest recorded at execution time -- rescoring an old run against a
    *newer* qrels version (after a correction) is the entire point of
    keeping run files around instead of re-running retrieval.

    Returns a table with one row per query_id in the run, one column per
    requested metric spec (containing ``None``/null for queries the metric
    guard-railed on -- see ``metrics.py``), plus ``run_id``, ``qrels_version``,
    ``overlap_rule``, and ``overlap_threshold`` columns recording exactly
    what produced the numbers.
    """
    if not metrics:
        raise ValueError("must request at least one metric spec")
    run = read_run(run_path)
    qrels_by_query = qrels_store.group_by_query(qrels_version)
    ranked_by_query = run.group_by_query()

    parsed = [_parse_metric_spec(m) for m in metrics]
    query_ids = sorted(ranked_by_query)

    columns: dict[str, list[object]] = {"query_id": list(query_ids)}
    for spec, (metric_name, k) in zip(metrics, parsed, strict=True):
        fn = RANKED_METRICS[metric_name]
        values: list[float | None] = []
        for qid in query_ids:
            q_qrels = qrels_by_query.get(qid, [])
            ranked = ranked_by_query[qid]
            values.append(fn(q_qrels, ranked, k, overlap))
        columns[spec] = values

    n = len(query_ids)
    columns["run_id"] = [run.manifest.run_id()] * n
    columns["qrels_version"] = [qrels_version] * n
    columns["overlap_rule"] = [overlap.rule.value] * n
    columns["overlap_threshold"] = [overlap.threshold] * n

    return pa.table(columns)

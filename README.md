# selfrag

A production-grade, **falsifiable** RAG system over a self-referential research corpus:
arXiv IR/RAG/ML papers plus this project's own decision log.

> **Name note.** "Self-RAG" is also a published technique (Asai et al., self-reflective
> retrieval with critique tokens). We are not claiming that paper. The name here refers to
> the corpus being self-referential — the system indexes the research it is built from, and
> the decisions taken while building it. Self-RAG-the-technique is one rung on our ablation
> ladder, nothing more.

## What this actually is

The deliverable is **not a pipeline**. It is an *experimental apparatus that produces
defensible numbers about pipelines*, plus the best configuration it finds.

That distinction is the whole project. "Try every technique" is an infinite backlog until
you can measure which ones work, on your data, with confidence intervals. So the evaluation
harness is built first and everything else is scored against it.

## Design commitments

These are load-bearing. Each exists because a specific, identified failure mode would
otherwise silently corrupt results.

| Commitment | Why |
|---|---|
| **Qrels are document character spans, never chunk ids** | Chunking is ablation axis #1. Chunk-keyed judgements are invalidated by the first experiment we run. |
| **Judges never see chunk prefixes or generated context** | Text-transform ablations contaminate the judge. Published evidence: annotator agreement collapses from κ=0.45 to κ=0.04 when a prefix is removed. Judge the transformed text and you measure your judge. |
| **Chunk ids hash coordinates, not text** | Contextual retrieval prepends LLM output. Hashing that makes ids nondeterministic across reruns, firing spurious tombstones and breaking index diffs. |
| **Comparisons are always paired, with a pre-registered minimum effect size** | At n=300 and recall@10≈0.70 the unpaired CI on a difference is ±7.3 points, while real ablation deltas are 1–3 points. Unpaired testing at this scale reports noise as discovery. |
| **Per-query metrics are persisted, not just means** | Paired statistics are impossible without them. |
| **Run files are persisted (TREC-style)** | Error analysis always finds 10–20% bad qrels. Persisted runs turn "re-run 40 configs" into "re-score 40 configs" — days into seconds. |
| **arXiv documents are deduplicated to base id** | v1…v5 of one paper are near-identical. Without this, top-k fills with one paper and recall collapses for reasons unrelated to the retriever. |
| **LanceDB chosen outright; no swappable store abstraction** | Prefilter-vs-postfilter semantics differ across engines, and postfiltering an ANN result set *changes recall@k*. "Same config, different store" is not a definable experiment. |
| **No LangChain / LlamaIndex / Haystack in the core** | The product is controlled comparison between components. Framework abstractions hide exactly the seams we need to vary and measure. |

## Constraints this is built under

CPU-only Windows laptop, 7.7 GB RAM, no GPU, no Docker. Realistic Python working set is
~2.5–3 GB. Peak RSS ≤ 2.5 GB and MCP fast-path p95 ≤ 800 ms are **pre-registered hard
constraints**: a configuration that violates one is disqualified regardless of its quality
score, because otherwise the ladder happily selects a winner that cannot be served.

Local ONNX-int8 models are the iteration tier. Paid APIs are used only for ceiling
measurements, on a dev corpus, with a persistent embedding cache — built before the first
paid call, because a crash at 80% through a paid run is a direct cash loss.

## Corpus and licensing

arXiv metadata via OAI-PMH; full text for a curated subset via LaTeX e-print source
(>90% of arXiv submissions are LaTeX, and source parsing beats PDF extraction on quality
*and* CPU cost).

Per the [arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html): requests are
rate-limited to one per three seconds on a single connection, e-prints are stored locally
for research only, and **the corpus is never redistributed**. Answers link back to arXiv.
`data/` is gitignored for this reason.

## Status

Phase 0 (foundations). See [CONTEXT.md](CONTEXT.md) for current state,
[MEMORY.md](MEMORY.md) for the running log, [decisions/](decisions/) for ADRs, and
[ERRORS.md](ERRORS.md) for failures worth remembering.

## Quickstart

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

```bash
.venv/Scripts/python.exe -m selfrag.cli doctor
```

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Layout

```
src/selfrag/
  ids.py        identity functions -- the most expensive decisions to reverse
  schema.py     core models; multimodal locator fields reserved from day one
  registry.py   components; a pipeline is YAML, a run is hash(manifest)
  ledger.py     DuckDB: machine truth. Never hand-edited.
  eval/         qrels, run files, metrics, paired statistics
configs/        pipeline definitions
decisions/      ADRs -- immutable, numbered
scripts/        CI gates and checkpoint helpers
```

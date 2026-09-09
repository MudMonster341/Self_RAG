---
status: accepted
date: 2026-09-09
authored_by: agent
derived_from: ["plan: selfrag — a production-grade, falsifiable RAG system (approved 2026-09-09)"]
supersedes: null
superseded_by: null
---

# 0002 — LanceDB is chosen outright; there is no swappable vector-store abstraction

*Recorded retroactively on 2026-09-09 from the approved project plan.*

## Context

A system whose entire purpose is controlled comparison invites a `VectorStore` protocol, so that
"which store is best" becomes another rung on the ladder. It is a tempting instinct and it is
wrong here, for three independent reasons.

**Store swaps move the headline metric with no config change.** Prefilter-versus-postfilter
semantics differ between engines, and postfiltering an ANN result set **changes recall@k**. An
abstraction that hides which one is happening would silently shift the number the entire project
is trying to measure.

**"Same config, different store" is not even definable.** HNSW's `ef_search`/`M` have no mapping
onto IVF_PQ's `nprobe`/`refine_factor`. Any comparison across stores is therefore a comparison of
two differently-tuned indexes, and attributing the delta to the store is unsupportable. Sparse and
hybrid support is not portable either, so the abstraction would leak on the first hybrid config.

**The hardware disqualifies the main alternative outright.** Qdrant's local mode is
memory-resident: 300k chunks × 1024-d fp32 is ~1.2 GB, and at 3072-d it is ~3.7 GB — against a
7.7 GB machine with a realistic Python working set of 2.5–3 GB and a pre-registered peak-RSS
ceiling of 2.5 GB. This is one of the five identified OOM sites in the design.

## Options considered

1. **A swappable abstraction over two or three stores.** Would let store choice become an ablation
   axis and would hedge against a bad pick. Costs: an interface that must paper over prefilter
   semantics and index-parameter spaces that do not correspond, producing comparisons that cannot
   be defended — and a metric that can move without a config moving.
2. **Qdrant in local mode.** Good filtering story and a familiar API. Memory-resident, so it hits
   the RSS ceiling at the target corpus size; ruled out by the hardware rather than by preference.
3. **LanceDB, chosen outright, behind a two-function seam.** Disk-backed/mmap Lance format with
   IVF_PQ; stays within the memory budget by construction.

## Decision

**LanceDB is the vector store. It is chosen, not abstracted.**

The seam is deliberately two operations, not an interface:

```
knn(vec, k, filter) -> [(chunk_id, score)]
get(chunk_ids)      -> rows
```

Prefiltering is **set explicitly and recorded in the run manifest**, so the semantics that would
otherwise move recall@k are themselves a logged experimental variable. Physical layout is **one
table per `(embedding_space, index_version)`**, keyed by `chunk_id` — a single fixed-width vector
column dies the moment a second embedding space with a different dimensionality appears, and
blue/green reindexing becomes *adding a table* rather than migrating one.

All fusion, sparse retrieval and reranking live in our own code. That is not a consequence of this
decision but a requirement independent of it: RRF-versus-convex-α is a measured axis, so it cannot
live inside a dependency.

## Consequences

- **Makes easy:** staying inside the 2.5 GB RSS constraint at 300k+ chunks, because the index is
  on disk and mmap'd rather than resident. Matryoshka truncation (1024→256/512) and int8 stack on
  top of that, with the recall cost of truncation measured rather than assumed.
- **Makes easy:** one index-parameter vocabulary in the manifest. Every number the ladder produces
  is attributable to a parameter that was actually varied.
- **Gives up:** "which vector store is best" as a question this project can answer. That is a real
  loss and it is accepted deliberately — the answer would not have been defensible anyway.
- **Gives up:** cheap migration. Moving to another engine later is real work, not a config flag,
  and would invalidate index-parameter comparability across the boundary.
- **Costs:** we inherit LanceDB's IVF_PQ tuning surface and its bugs, and we implement fusion,
  hybrid scoring and reranking ourselves.
- **Revisit if:** LanceDB cannot hold the pre-registered MCP fast path of p95 ≤ 800 ms at full
  corpus size, or the corpus grows past ~500k chunks and indexing moves to a rented box. Even then
  the replacement is a superseding ADR plus a full re-run of affected configs — never a swap
  behind an interface.

## Links

- Related: [ADR 0005 — no RAG framework in the core](0005-no-rag-framework-in-the-core.md)
  (same reasoning applied to control flow rather than to storage)
- Constraints in [CONTEXT.md](../CONTEXT.md#constraints)

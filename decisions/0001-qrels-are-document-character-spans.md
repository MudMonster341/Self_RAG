---
status: accepted
date: 2026-09-09
authored_by: agent
derived_from: ["plan: selfrag — a production-grade, falsifiable RAG system (approved 2026-09-09)"]
supersedes: null
superseded_by: null
---

# 0001 — Qrels are document character spans, never chunk ids

*Recorded retroactively on 2026-09-09 from the approved project plan; the decision was taken
during planning, before this decision log existed.*

## Context

Chunking is **ablation axis #1**: seven candidate chunkers (fixed, recursive, title-chain prefix,
semantic merge, windowed late chunking, Anthropic contextual retrieval, parent-child), each
producing a different set of chunks over the same documents.

If a relevance judgement is keyed to a `chunk_id`, then **changing the chunker invalidates 100% of
the judgements**. This would not surface until the first ablation run — realistically week 3 —
leaving two options, both bad: re-label from scratch for every chunker, or fuzzy-remap old
judgements onto new chunks and convert the entire result set into noise.

Re-labelling is not affordable. Human labelling is ~10 hours of one person's time at 3–5 min/item
and quality visibly collapses after hour 3; the plan budgets it in sessions, not sittings. A
labelling scheme that must be redone per chunker multiplies that by seven.

Two further forces:

- **Canonical text is already frozen.** One normalised UTF-8 string per document with
  `doc_text_sha256`, produced by `normalize_text` running exactly once at ingest. Character offsets
  into that string are stable by construction, which makes span-anchored judgements possible at no
  extra cost.
- **Judge contamination is measured, not hypothetical.** Inter-annotator agreement on the same
  task with the same annotators collapsed from **κ=0.45 on prefixed chunks to κ=0.04 on
  de-prefixed chunks** (arXiv 2608.00824). Since text-transform ablations are on the ladder,
  judging the transformed retrieval unit means measuring the judge.

## Options considered

1. **Qrels keyed to `chunk_id`.** Simplest to collect — the annotator judges exactly the unit the
   system retrieves, so no overlap rule is needed. Dies at the first chunker change, which is the
   first experiment we intend to run.
2. **Qrels keyed to `doc_id` only (document-level relevance).** Survives every chunker change and
   is cheap to collect. But it is far too coarse: a 10k-token paper marked "relevant" cannot
   distinguish a retriever that surfaced the right paragraph from one that surfaced the title page.
   That resolution is the entire point of the ladder, and losing it makes 1–3 point deltas
   unmeasurable.
3. **Qrels as `(query_id, doc_id, char_start, char_end, grade)` spans into the frozen canonical
   text**, with a chunk counted relevant iff its span overlaps a gold span under a pre-registered
   rule.

## Decision

**Qrels are `(query_id, doc_id, char_start, char_end, grade)` character spans into the frozen
canonical document text. Never chunk ids.**

A retrieved chunk counts as relevant iff its `[char_start, char_end)` overlaps a gold span under a
rule **pre-registered before the first labelling session and never tuned thereafter** — either
IoU ≥ 0.1, or "contains the gold span's midpoint". Tuning that rule after seeing results would be
tuning the metric, so the specific choice is fixed in the qrels schema and recorded in a follow-up
ADR at that point.

Two rules ride along with this one, because they only work if judgements are span-anchored:

- **Judges never see `context_prefix` or any generated context** — only the raw canonical span.
- **Qrels are append-only and versioned.** Error analysis reliably finds 10–20% bad qrels; every
  result carries its `qrels_version`, and corrections are new rows, never edits.

## Consequences

- **Makes easy:** running the whole chunking axis against one labelling effort. Combined with
  persisted TREC-style run files, re-qualifying 40 prior configs against a corrected qrels version
  becomes a pure function of `(run file, qrels)` — seconds instead of days. This is the highest
  leverage item in the reproducibility spine.
- **Makes easy:** honest judging, because the annotator sees canonical text that no ablation can
  transform underneath them.
- **Gives up:** the annotator no longer judges the exact object the system returns, so a chunk that
  clips a gold span at an awkward boundary is scored by fiat rather than by human judgement. The
  overlap rule is a defensible but genuinely arbitrary parameter, and it is one more thing that has
  to be right before labelling starts.
- **Costs:** the canonical text must be frozen and offset-stable *before* any labelling, which
  pushes normalisation and ingest determinism earlier in the schedule than they would otherwise sit.
- **Revisit if:** `normalize_text` ever has to change. That invalidates every stored offset and
  therefore every qrel, and would need its own migration ADR. Nothing else within this project
  justifies reopening it — reversing this decision costs the entire labelling budget.

## Links

- Related: [ADR 0003 — chunk ids hash coordinates](0003-chunk-ids-hash-coordinates-not-text.md)
  (the other half of making identity stable across the chunking axis)
- Related: [ADR 0004 — arXiv base-id deduplication](0004-arxiv-base-id-deduplication.md)
  (qrels are defined at base-paper level, which depends on that decision)
- Implemented in: `src/selfrag/schema.py`, and `src/selfrag/eval/` once it exists
- Invariants 1, 2 and 4 in [CLAUDE.md](../CLAUDE.md)

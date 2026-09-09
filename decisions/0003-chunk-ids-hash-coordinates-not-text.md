---
status: accepted
date: 2026-09-09
authored_by: agent
derived_from: ["plan: selfrag — a production-grade, falsifiable RAG system (approved 2026-09-09)"]
supersedes: null
superseded_by: null
---

# 0003 — Chunk ids hash coordinates, never chunk text

*Recorded retroactively on 2026-09-09 from the approved project plan.*

## Context

Content-addressed identity is the default instinct for a chunk: hash the text, get a stable id,
get deduplication for free. In this system it breaks, because **the chunk text is not stable**.

Anthropic contextual retrieval — which sits inside ablation axis 1, applied to the top-2 chunkers —
prepends LLM-generated context to each chunk before embedding. That generated text varies between
runs. If the id hashed the text:

- ids would be **nondeterministic across reruns** of the same configuration;
- every rerun would look like a wholesale delete-and-insert, **firing spurious tombstones**;
- **blue/green index diffing would break**, since diffing two index versions is a set operation on
  ids and the sets would never intersect.

It also collides with a Phase 1 exit criterion: re-running ingest must produce **zero new chunk
ids**. Idempotency is what makes an interrupted overnight ingest resumable on a machine that
cannot afford to redo one.

## Options considered

1. **`sha256(chunk_text)`.** Natural content addressing; identical chunks collapse automatically.
   Nondeterministic under contextual retrieval and under any text transform on the ladder, and it
   couples identity to what is really a presentation concern.
2. **Sequential ids per document (`doc_id:0001`, …).** Stable-looking and human-readable, but
   position-dependent: inserting or resizing one chunk renumbers everything after it, so a chunker
   parameter change silently reassigns ids. Ids are also not comparable across chunker configs,
   which is precisely the comparison the ladder makes.
3. **`sha256(doc_id ‖ chunker_config_id ‖ char_start ‖ char_end)`** — hash the coordinates that
   define the chunk, not the bytes it currently contains.

## Decision

**`chunk_uid = sha256(doc_id ‖ chunker_config_id ‖ char_start ‖ char_end)`.** Identity is the
chunk's coordinates in the frozen canonical text plus the configuration that produced it.

Three things fall out of that and are part of this decision:

- **`raw_text_sha256` is a separate column**, used only for deduplication and as part of the
  embedding-cache key. Content hashing still exists; it just is not identity.
- **Generated context lives in its own columns** — `context_prefix`, `context_model_id`,
  `context_prompt_sha` — never inside the identity function, and therefore regenerable
  independently of embeddings.
- **Degenerate spans are rejected, not tolerated.** `chunk_uid` raises on a negative start or an
  empty/inverted span rather than returning a valid-looking id for a chunk that cannot be resolved
  back to text.

## Consequences

- **Makes easy:** idempotent re-ingest — re-running produces byte-identical ids, so the Phase 1
  exit criterion is checkable rather than aspirational.
- **Makes easy:** blue/green reindexing and index diffing, which reduce to set operations on ids.
- **Makes easy:** regenerating contextual prefixes without disturbing identity, so the $7–12
  contextualisation budget is spent once against the winning chunker rather than per rerun.
- **Gives up:** free deduplication. Two byte-identical chunks in different documents get different
  ids, which is correct but means dedup must be done explicitly — via `raw_text_sha256` and
  chunk-level MinHash (see [ADR 0004](0004-arxiv-base-id-deduplication.md)).
- **Costs:** the id is opaque and cannot be resolved to text without the document and the chunker
  config. Debugging by eye is harder than with sequential ids.
- **By design:** changing `chunker_config_id` changes every id in that configuration. That is the
  point — chunks from two chunkers are different objects and must never be conflated.
- **Revisit if:** `normalize_text` changes, which shifts every offset and therefore every id. That
  is the same tripwire as [ADR 0001](0001-qrels-are-document-character-spans.md) and would need a
  migration ADR covering both.

## Links

- Related: [ADR 0001 — qrels are document character spans](0001-qrels-are-document-character-spans.md)
- Related: [ADR 0004 — arXiv base-id deduplication](0004-arxiv-base-id-deduplication.md)
- Implemented in: `src/selfrag/ids.py` — `chunk_uid`, `content_hash`, `normalize_text`
- Invariants 3 and 4 in [CLAUDE.md](../CLAUDE.md)

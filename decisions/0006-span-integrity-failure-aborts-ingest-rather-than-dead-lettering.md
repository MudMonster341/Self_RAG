---
status: accepted
date: 2026-09-10
authored_by: agent
derived_from: ["CLAUDE.md invariants 1 and 4", "Phase 1 task: build the ingest pipeline orchestrator"]
supersedes: null
superseded_by: null
---

# 0006 — A chunk span-integrity failure aborts the whole ingest run, not just that document

## Context

The ingest orchestrator (`src/selfrag/ingest/pipeline.py`) has a general dead-letter contract:
a document that fails at any stage — acquisition, parsing, freezing, chunking — is recorded via
`Manifest.mark_failed` with its stage and reason, and the run continues over every other document.
That contract exists so one malformed paper in a 300-document ladder run cannot take down the
whole ingest.

Immediately before persisting each chunk, the pipeline asserts
`canonical.get_span(doc_id, chunk.char_start, chunk.char_end) == chunk.text` — the chunk's stored
text must round-trip exactly through its own offsets into the frozen canonical text. This checks
CLAUDE.md invariant 1 (qrels are spans over canonical text) and invariant 4 (canonical text is
frozen once; every offset indexes into that exact string) at the one point where a violation would
otherwise become undetectable: after this, the chunk is indistinguishable from a correct one to
every downstream consumer, including a human judge reading `get_span` output for a qrel.

The question this ADR answers: when that assertion fails, does the pipeline dead-letter *that
document* and continue, matching the general contract — or does it abort the entire run?

A span mismatch cannot be caused by anything specific to one document's content. It can only mean
one of two things, and both are systemic: the chunker component (built once, shared across every
document in the run, identified by one `chunker_config_id`) has a coordinate bug that would produce
the same kind of mismatch for the next document too; or the canonical text on disk for this
`doc_id` no longer matches what the offsets were computed against — evidence that immutability
itself has been violated somewhere upstream, which calls every other document's already-persisted
chunks into question, not just this one's.

## Options considered

1. **Dead-letter it like every other stage failure.** Consistent with the rest of the pipeline's
   philosophy and the simplest mental model ("every stage failure looks the same"). But it treats
   a broken *invariant* the same as a broken *document* — the run finishes "successfully" having
   silently walked past a bug that could be corrupting every chunk it touches, and nothing forces
   a human to look at it before the ladder starts measuring retrieval quality against the result.
2. **Log loudly (`record_error`) but keep going, without dead-lettering the document.** Splits the
   difference, but produces the worst outcome of both options: a visible error buried among
   ordinary per-document failures, and a run that still finishes and gets treated as usable output.
3. **Raise a dedicated exception (`SpanIntegrityError`) that the per-document dead-letter loop does
   not catch, aborting `run_ingest` entirely.** Nothing gets persisted for the batch in progress;
   the operator sees a crash, not a log line, and has to fix the actual bug before any further
   ingest can complete.

## Decision

**A span-integrity failure raises `SpanIntegrityError` and is deliberately not caught by the
per-document dead-letter `try`/`except` blocks in `run_ingest`.** It propagates and aborts the run.

This is a narrow, explicit carve-out from the general dead-letter contract, justified by CLAUDE.md
itself: invariants "silently corrupt results" when violated, and invariant 4 specifically says a
normalization/offset problem "invalidates every offset and every qrel" — not just the one document
it was first noticed on. Continuing to ingest under a broken offset invariant would manufacture
more of exactly the corrupted state the invariant exists to prevent, and — unlike an ordinary
parse failure — there is no way to route around it by falling back to a different parser or
skipping one bad paper, because the mismatch says nothing about the *document* being bad.

## Consequences

- **Makes easy:** noticing a chunker coordinate bug or a canonical-text corruption immediately,
  at the first document it affects, rather than discovering it later as an unexplained retrieval
  quality regression with no record of which chunks were ever suspect.
- **Gives up:** the uniformity of "every stage failure looks the same to the operator." Anyone
  reading `run_ingest`'s dead-letter loop has to know this one exception type is intentionally
  excluded, which is why it is documented on `SpanIntegrityError` itself, not just here.
- **Costs availability:** a single bad chunker (or, hypothetically, a corrupted canonical file) can
  halt an entire multi-hour ingest run partway through, losing the chunk/document work for whatever
  was still in the in-memory batch when it fired. Acceptable because `run_ingest` is already
  idempotent (see the module docstring) — the aborted run's already-`upsert`ed documents and chunks
  are unaffected, and a fixed rerun resumes from exactly where the manifest says it left off.
- **Revisit if:** the ingest ladder grows large enough (hundreds of documents, unattended overnight
  runs per MEMORY.md's Phase 1 resume plan) that a single mid-run abort becomes operationally
  expensive enough to want a "quarantine the corpus snapshot and keep going" middle ground instead.
  That would need its own ADR, not a quiet change to this one's behaviour.

## Links

- Related: [ADR 0001 — qrels are document character spans](0001-qrels-are-document-character-spans.md)
- Related: [ADR 0003 — chunk ids hash coordinates, never text](0003-chunk-ids-hash-coordinates-not-text.md)
- Implemented in: `src/selfrag/ingest/pipeline.py` — `SpanIntegrityError`, `_assert_chunk_span_matches`
- Invariants 1 and 4 in [CLAUDE.md](../CLAUDE.md)

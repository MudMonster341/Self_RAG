---
status: accepted
date: 2026-09-09
authored_by: agent
derived_from: ["plan: selfrag — a production-grade, falsifiable RAG system (approved 2026-09-09)"]
supersedes: null
superseded_by: null
---

# 0004 — Documents are deduplicated to the arXiv base id

*Recorded retroactively on 2026-09-09 from the approved project plan. This concern was missing
from the first draft of the design and was added after review.*

## Context

`2401.01234v1` through `v5` are five near-identical documents. arXiv versioning is routine, and
related-work sections across papers quote each other verbatim on top of that.

Two specific failures follow if versions are treated as distinct documents:

1. **Recall@10 is corrupted for reasons unrelated to the retriever.** Top-10 fills with five
   versions of one paper. The metric then measures how many near-duplicates the corpus happens to
   contain, and a retriever change and a corpus refresh become indistinguishable.
2. **Relevance judgements become unstable.** A qrel pointing at v2 while the system returns v4
   makes "is that a hit?" a judgement call — and it is a call that will be made inconsistently
   across a 10-hour labelling effort, injecting noise directly into the ground truth.

This matters more here than in most corpora because deltas on the ladder are 1–3 points. A
duplication artefact of that size is indistinguishable from a real result.

## Options considered

1. **Treat each version as its own document.** Faithful to the source and requires no work.
   Produces both failures above.
2. **Index all versions; deduplicate at result time.** Keeps full fidelity in the store. But the
   dedup step then lives inside every retriever and every metric, differs per configuration, and
   becomes an uncontrolled variable sitting directly on the measurement path — the worst place for
   one.
3. **Canonicalise to the base id at ingest**, so the corpus contains one document per paper before
   any retrieval or labelling happens.

## Decision

**Canonicalise every arXiv identifier to its base id at ingest. Keep the latest version, and store
`version` plus `versions_seen`.**

- **Near-duplicate chunks are collapsed with MinHash** at chunk level, at a **fixed threshold**,
  keeping the earliest and recording `dup_of`. The threshold is fixed by fiat and never tuned:
  tuning it against results would be tuning the metric.
- **Qrels are defined at base-paper level**, which is what removes the "is v4 a hit for a v2 qrel?"
  judgement call entirely.
- **The dedup configuration is part of the run manifest.** Changing it invalidates prior results,
  and the manifest hash makes that visible instead of silent.
- Identifier parsing **raises rather than returning a sentinel** on an unrecognised id. A silently
  mis-parsed identifier would corrupt the document key space, and a loud failure at ingest is
  cheaper than a quiet one at evaluation.

## Consequences

- **Makes easy:** recall@k that actually measures retrieval. Removing this artefact is what lets
  1–3 point deltas be interpreted at all.
- **Makes easy:** unambiguous labelling, and idempotent re-ingest when arXiv publishes a new
  version — the base id is unchanged, so identity is stable.
- **Gives up:** version-level questions. "What did v1 claim before the revision?" is not answerable
  from the index; answering it means going back to the source. Accepted as out of scope.
- **Gives up:** superseded claims across versions, which are collapsed away. Worth noting because
  the corpus is one of competing empirical papers and the contradiction detector operates
  *between* papers, not between versions of one.
- **Costs:** dedup becomes a first-class schema concern — `dup_of`, `versions_seen`, and a MinHash
  pass at ingest — rather than a post-processing convenience.
- **Revisit if:** a query stratum genuinely needs version-level distinctions. The most likely
  candidate is the temporal / false-premise unanswerable strata, where a claim retracted between
  versions would make a good near-miss. That would be a superseding ADR, not a config change,
  because it changes the document key space.

## Links

- Related: [ADR 0001 — qrels are document character spans](0001-qrels-are-document-character-spans.md)
  (qrels are defined at base-paper level, which this decision makes possible)
- Related: [ADR 0003 — chunk ids hash coordinates](0003-chunk-ids-hash-coordinates-not-text.md)
  (`raw_text_sha256` is the exact-match half of dedup; MinHash is the near-match half)
- Implemented in: `src/selfrag/ids.py` — `canonical_arxiv_id`

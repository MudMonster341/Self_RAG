# Working memory

Append-only, **newest at the bottom**. One entry per work chunk, written at the end of it. Records
*why*, not *what changed* — the diff already says what.

Read the last few entries with `tail -n 150 MEMORY.md` before starting work. Orientation lives in
[CONTEXT.md](CONTEXT.md); decisions in [decisions/](decisions/); failures in [ERRORS.md](ERRORS.md).

**The checkpoint routine**, at the end of every work chunk:

1. Append an entry here — dated, what changed and why.
2. Add ADRs for any decisions made.
3. Append [ERRORS.md](ERRORS.md) entries for any non-obvious failures resolved.
4. Refresh [CONTEXT.md](CONTEXT.md) if the current state, commands or layout changed.
5. `.\scripts\checkpoint.ps1 "<summary>"` — stages, commits, pushes, and fails loudly if it cannot.

---

## 2026-09-09 — Project framing locked; plan approved

**Did:**
- Settled the corpus, the compute posture, the build order and the primary objective: a
  self-referential corpus of arXiv IR/RAG/ML papers plus this project's own decision log; local
  ONNX-int8 for iteration with paid APIs reserved for ceiling measurements; **eval harness first**,
  ablation ladder second; primary objective **no confident wrong answers**.
- Sized the whole design against the actual machine — 7.7 GB RAM, CPU-only — and pre-registered
  peak RSS ≤ 2.5 GB and MCP fast-path p95 ≤ 800 ms as disqualifying constraints rather than goals.
- Cut SPLADE (9–17 h per config, no CPU serving path) and ColPali (3B vision model, ~2.6 GB of
  vectors at 5k pages) on hardware grounds before either could absorb a week.

**Why:** the reframing that made the project finite is that the deliverable is **not a pipeline**
but an apparatus that produces defensible numbers about pipelines. "Try every technique" is an
infinite backlog; "measure which ones survive a paired test at a pre-registered effect size" is a
plan with an end. Everything downstream — building eval before baseline, persisting run files,
version-locking qrels — follows from that one sentence.

**Decisions:** the substantive ones were taken here and are recorded retroactively as
[ADR 0001](decisions/0001-qrels-are-document-character-spans.md) through
[ADR 0005](decisions/0005-no-rag-framework-in-the-core.md).

**Failures:** none.

**Next:** Phase 0 foundations.

---

## 2026-09-09 — Phase 0: repo, identity functions, core schema

**Did:**
- Initialised the git repository, pointed `origin` at `github.com/MudMonster341/Self_RAG`, and made
  the first commit `306f931` — *feat: project foundation — identity functions, core schema, working
  agreement*.
- `src/selfrag/ids.py`: `canonical_arxiv_id`, `normalize_text`, `content_hash`, `chunk_uid`,
  `config_hash`.
- `src/selfrag/schema.py`: core Pydantic models, including the `papers`/`project` namespace split
  and the nine query strata.
- `README.md`, `CLAUDE.md`, `.gitignore` (with `data/` excluded — the corpus is not
  redistributable), `.env.example`, `.gitattributes`.

**Why:** identity functions were written *first and alone*, in their own small module, because they
are the decisions that are most expensive to reverse. Once documents are ingested and qrels are
labelled, a change to `chunk_uid` or `normalize_text` invalidates every stored offset and every
judgement — so these get a dedicated module and heavy tests rather than being scattered through
whatever code happens to need them.

Two details that look redundant and are not: `canonical_arxiv_id` **raises** rather than returning
a sentinel, because a silently mis-parsed identifier corrupts the document key space and would only
surface at evaluation time; and `chunk_uid` rejects empty or inverted spans, which would otherwise
produce a valid-looking id for a chunk that cannot be resolved back to any text.

`data/` was gitignored at commit one rather than later, because the arXiv Terms of Use permit local
storage for research but not redistribution, and a corpus accidentally pushed to a public repo
cannot be un-pushed.

**Decisions:** [ADR 0003](decisions/0003-chunk-ids-hash-coordinates-not-text.md) and
[ADR 0004](decisions/0004-arxiv-base-id-deduplication.md) are both implemented by this module.

**Failures:** [ERR-0001](ERRORS.md#err-0001--shell-quoting-swallows-python-source-in-git-bash-2026-09-09)
— shell quoting swallowed Python source and reported it as a Python problem.

**Next:** the rest of Phase 0 — `registry.py`, `ledger.py`, `cli.py`, the run manifest, TREC run
files, versioned qrels, the embedding cache, and the CI no-fake-code gate. Phase 0 exits when
`selfrag run --config baseline.yaml` runs end to end on 10 documents and writes a ledger row.

---

## 2026-09-09 — Bootstrapped the project-memory system

**Did:**
- Created the four memory artifacts: [CONTEXT.md](CONTEXT.md), this file,
  [ERRORS.md](ERRORS.md), and [decisions/](decisions/) with `0000-template.md`.
- Wrote ADRs [0001](decisions/0001-qrels-are-document-character-spans.md)–[0005](decisions/0005-no-rag-framework-in-the-core.md)
  retroactively, from the approved plan, while the rejected options were still recoverable.
- Logged [ERR-0001](ERRORS.md) — the Git Bash quoting failure, which had already cost time once.
- Added `scripts/checkpoint.sh` and `scripts/checkpoint.ps1`.
- Installed the same discipline globally as the `project-memory` skill, so it applies to every
  project rather than only this one.

**Why:** this project runs for ~23–30 sessions across months, and the reasoning behind its schema
decisions is the part that will not survive in anyone's head. The specific failure being defended
against is *plausible reconstruction*: six weeks from now the reasons for hashing coordinates
instead of text will be re-derived, confidently and wrongly, and the re-derivation will be
indistinguishable from the record. So the options that were rejected are written down now, dated,
with their concrete costs — that is the part that cannot be recovered later.

Retroactive ADRs are dated with the date they were *written*, not the date the decision was taken,
and say so in the file. Backdating would make the log lie about its own reliability.

The checkpoint scripts refuse to commit unless `MEMORY.md` has a new entry dated today, and fail
loudly with a distinct exit code when they cannot reach the remote. Both are deliberate: a
memory system that depends on remembering to use it decays within a fortnight, and a push that
fails quietly is worse than no push at all, because it looks like a backup.

**Decisions:** none beyond the retroactive ADRs; the memory system's own conventions live in the
global `project-memory` skill rather than in this repo's decision log.

**Failures:** none new.

**Next:** first checkpoint — `.\scripts\checkpoint.ps1 "bootstrap project-memory system"` — then
back to Phase 0. `configs/`, `tests/` and `.github/` are still empty; `tests/` in particular blocks
the Phase 0 exit criterion.

---

## 2026-09-09 — Phase 0 complete: eval spine, ledger, registry, CLI, CI gate

**What:** Phase 0 is done and its exit criterion is met. `configs/`, `tests/` and `.github/` are
no longer empty. 223 tests pass; `ruff check .` clean; `check_no_fake_code.py` reports 0
violations and 0 escapes across all of `src/`.

Landed, in dependency order:

- `eval/` — span-anchored relevance with an explicit `OverlapSpec` (rule + threshold, no
  signature defaults), append-only versioned qrels as full snapshots, TREC-style persisted run
  files with a `rescore()` that needs no index/corpus/model, and paired bootstrap +
  randomization tests gated on a pre-registered minimum effect size.
- `registry.py` — `Component` ABC, `config_id`, and two real reference components
  (`chunker.fixed`, `query_transform.passthrough`).
- `ledger.py` — DuckDB; `run_metrics` stores **per-query** values, not means.
- `paths.py`, `cli.py` (`doctor`, `registry list`, `ledger`, `config validate`, `version`),
  `scripts/check_no_fake_code.py`, `.github/workflows/ci.yml`, `configs/baseline.yaml`.

**Why:** the harness was built before any retrieval code deliberately. Without it, "try every
technique" has no stopping condition and no way to tell a real improvement from noise. The single
most valuable artefact here is the persisted run file: error analysis reliably finds 10–20% bad
qrels, and when they are corrected this turns "re-run 40 configs" into "re-score 40 configs" —
days into seconds.

**Decisions:** `config_id` now covers a component's declared `version`, not just its config
values, and renders as `fixed.v1@0b9b5476`. See below; not yet a full ADR because it refines
[0003](decisions/0003-chunk-ids-hash-coordinates-not-text.md) rather than standing alone.

**Failures:**

- Caught in review, before it could do damage: `config_id` originally hashed only config *values*.
  Since `config_id` feeds `chunk_uid`, a chunker whose implementation changed while its config
  stayed byte-identical would have produced **the same chunk ids for different text** — two index
  generations silently sharing ids, stale vectors surviving a re-index, tombstones never firing,
  and blue/green diffs reporting no change. Precisely the failure class the id scheme exists to
  prevent. Fixed by adding a `version` ClassVar that participates in the hash *and* is visible in
  the label. Free to fix now; expensive after the first index is built.
- `nDCG` uses **linear** gain while much of the literature uses `2**grade - 1`. Internal
  comparisons are unaffected, but our numbers are not comparable to published ones without
  checking. Documented in the docstring rather than changed, since consistency matters more than
  matching any one paper.
- Two environment failures cost real time and are logged as
  [ERR-0001](ERRORS.md) and in [CLAUDE.md](CLAUDE.md): bash heredocs mangle Python containing
  backticks/apostrophes (write files with the Write tool), and `uv` silently resolves to the
  Windows Store Python unless given `--python ./.venv/Scripts/python.exe`.

**Observation from `selfrag doctor`:** available RAM was **1.12 GB**, against the 3 GB this
project's local-model tier assumes. With the editor and a browser open there is far less headroom
than the plan's arithmetic assumed. This does not block Phase 1, but the dev-corpus size and ORT
thread count will need measuring on a quiet machine rather than trusting the estimate.

**Next:** Phase 1 — corpus. Streaming OAI-PMH metadata harvest into Parquet (the arXiv snapshot is
a 4–5 GB JSONL and must never be materialised), LaTeX e-print parsing with a PyMuPDF fallback,
frozen canonical text, and base-id + MinHash deduplication. Exit criterion: re-running ingest
produces **zero** new chunk ids.

**Session end (2026-09-09 23:20):** session was ending on usage limits with Phase 1 not started.
Rather than stopping, a one-time scheduled task `selfrag-phase1-resume` was armed for
**03:14 local** (4 min after the stated 03:10 reset, so the reset has definitely landed). It
starts a *fresh* run with no conversation memory and orients purely from these files — which is
the first real test of whether this memory system does its job. Scope is deliberately capped: a
pilot ingest of ≤300 papers, no paid API calls. The generalised rule now lives in the global
`project-memory` skill under "Session continuity".

---

## 2026-09-10 — Overnight run produced nothing; machine slept through it

**What:** the `selfrag-phase1-resume` task fired on time at 03:14 and accomplished nothing.
Zero commits, empty `data/`, no log. Phase 1 has not been started. The repo is exactly where
it was at 23:20 last night: 6 commits, Phase 0 complete, 223 tests passing (re-verified this
morning after Windows rebooted the machine — ruff and the no-fake-code gate are clean too, so
the OS upgrade broke nothing).

**Why it failed:** the machine entered Modern Standby at 23:20:41 and did not wake until
06:38. See [ERR-0002](ERRORS.md). Two compounding factors: the app's scheduler arms an
in-process timer, not an RTC wake timer, so it cannot wake a sleeping machine; and Windows
Update rebooted the box three times around 04:30 regardless.

**The reframe this forces:** the unpushed remote is no longer just a missing backup, it is
what blocks the only reliable form of unattended work. A cloud routine does not care whether
this machine is awake, but it needs a repo to clone. Pushing is now on the critical path.

**Failures:** the risk was explicitly recorded in the plan's Known Open Risks and then not
acted on before scheduling. Writing a risk down is not mitigating it — the check ("is this
machine configured to stay awake?") was one command and was not run.

**Next:** unblock the push, then Phase 1 corpus ingest.

---

## 2026-09-10 — Phase 1: the ingest pipeline orchestrator, and its exit criterion

**What:** built `src/selfrag/ingest/pipeline.py`, wiring every existing `ingest/` module (manifest,
arxiv_client, eprint, latex, pdf_fallback, canonical, dedup, quality) into one flow: seed manifest
→ acquire → parse (LaTeX, PDF fallback) → freeze canonical → assess quality → chunk (via registry)
→ dedup → persist → record run. Extended `ledger.py` with document/chunk persistence
(`upsert_documents`/`upsert_chunks`, batched via a registered Arrow table and `ON CONFLICT DO
UPDATE` — not row-by-row) and ingest-run bookkeeping, and added `selfrag ingest run/status/quality`
to the CLI. 495 tests pass (452 → 495), `ruff` and the no-fake-code gate are clean.

**Phase 1's exit criterion is met**: `tests/integration/test_ingest_idempotency.py` proves that
re-ingesting the same fixture corpus (twice, then a third time for good measure) adds zero new
chunk ids, zero duplicate ledger rows, identical canonical-text hashes, identical document rows,
and zero re-acquisition of already-acquired documents — entirely via `httpx`-free fixtures, no
network reachable from the test.

**Design decisions:**

- **Idempotency needed no new tracking layer.** A document already `PARSED` in the manifest is
  never re-acquired or re-parsed — its canonical text is already frozen and immutable (invariant
  4), so quality assessment and chunking just re-read that frozen text every run. Recomputing the
  same deterministic thing and upserting it is indistinguishable, to the ledger, from computing it
  once. A document `ACQUIRED` but not yet `PARSED` (process died mid-pipeline on a prior run)
  resumes by re-locating its files under the same deterministic `dest_root/doc_id` layout, never by
  re-fetching — proven directly in a test that gives the fixture source zero registered fixtures
  for that doc_id and asserts `acquire()` was still never called.
- **Canonical text's write-once invariant reframes what "a changed document" can mean.**
  `freeze_canonical` refuses to re-freeze a `doc_id` under different text (by design, pre-existing).
  So the "changed source document produces new chunk ids" test models the only kind of change this
  architecture allows: a document that failed *before* any canonical text was ever frozen for it,
  fixed, and retried — never a same-doc_id content mutation.
- **A span-integrity failure aborts the whole run rather than being dead-lettered.** Every other
  per-document failure (acquire/parse/freeze/chunk) is caught, recorded via `mark_failed`, and the
  run continues — but a chunk whose stored text does not round-trip through
  `canonical.get_span(start, end)` can only mean a systemic bug (the shared chunker, or corrupted
  canonical text), not a property of one unlucky document, so `SpanIntegrityError` is deliberately
  excluded from the dead-letter `except` blocks. See
  [ADR 0006](decisions/0006-span-integrity-failure-aborts-ingest-rather-than-dead-lettering.md).
- **`chunks.tombstoned_at` is a real column, not a join against `documents.tombstoned_at`.** Bumped
  the ledger schema to v2 and exercised the forward-only migration mechanism for the first time
  (`_migrate_v1_to_v2`, tested against a hand-built v1 database). A per-chunk timestamp makes
  removal auditable at the row a future embedding-cache invalidation would actually key off, and
  `upsert_chunks` never writes that column, so an ordinary re-ingest can never clear a tombstone.

**Not done, deliberately out of scope:** no real corpus was ingested (`configs/ingest.yaml` still
names one placeholder doc_id — populating it needs a live OAI-PMH harvest, a paid-free but
rate-limited network operation this session did not run). `selfrag doctor` still reports the same
~1.2–1.3 GB available RAM as at Phase 0 exit; unchanged, not re-investigated.

**Next:** a real pilot ingest (≤300 papers per the earlier Phase 1 resume plan) against live arXiv,
then Phase 2 — retrieval.

---

## 2026-09-10 — Phase 1 complete, and proven on a real corpus rather than fixtures

**What:** the first real corpus is ingested. Twelve foundational retrieval/RAG papers
(`configs/pilot.yaml`) — RAG, REALM, FiD, Self-RAG, DPR, Contriever, E5, BGE, BEIR, Late
Chunking, ColBERT, plus one from our own live cs.IR harvest — through the full pipeline:
acquire → parse → freeze → quality → chunk → dedup → persist.

    12/12 acquired and parsed, 0 failures, 1343 chunks, 35.3s
    section-label coverage 99.68% · unexpanded_macro_rate 2.06% · broken math 0.92%

35.3s for 12 papers is essentially the arXiv throttle (12 × 3s); the pipeline itself is nearly
free. Re-running: 0 acquired, 0 parsed, 12 skipped, **0 chunks newly written**, 1.9s. Ledger
confirms 1343 distinct chunk ids against 1343 rows after three consecutive runs.

**Why it matters that this was run for real:** the exit criterion was already proven on a
fixture corpus, and that was not enough. Twice today a fully-passing test suite hid a defect
that the first contact with reality exposed in seconds — [ERR-0003](ERRORS.md) (arXiv moved its
OAI-PMH endpoint; 65 mocked transport tests could not see it) and the LaTeX parser defects
(102 fixture tests passed while real papers produced nested sibling headings and 75 unexpanded
macros). Fixtures prove the code is self-consistent. They cannot prove it meets the world.

**Failures:** one reporting defect found by running it. `chunks created` reported 1343 on a run
that parsed zero documents, because a skipped document is still re-chunked from its frozen
canonical text — correct behaviour, misleading label, and precisely the number a person reads to
decide whether a re-ingest duplicated the corpus. Split into `chunks in corpus` and `chunks
newly written`. The second now reads 0 on a re-run, which is the claim being made.

**Decisions:** [ADR 0006](decisions/0006-span-integrity-failure-aborts-ingest-rather-than-dead-lettering.md)
— a chunk that fails to round-trip through `canonical.get_span` aborts the run instead of being
dead-lettered, because it indicates a systemic bug rather than a bad document.

**Next:** Phase 2, the evaluation harness — and the corpus decision described in the reference
artifact. Swapping corpora is cheap now and expensive once human relevance judgements exist,
because judgements are corpus-specific and roughly ten hours of work. That fork is Phase 2's
first question, not a later one.

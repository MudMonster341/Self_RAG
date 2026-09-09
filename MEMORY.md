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

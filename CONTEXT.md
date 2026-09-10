# selfrag — context

**Last updated:** 2026-09-10 · **Repo:** https://github.com/MudMonster341/Self_RAG ·
**Local:** `C:\Users\Mustafa\Desktop\Mustafa\Projects\P_RAG` · **Package:** `selfrag`

Read this first. Then [MEMORY.md](MEMORY.md) for what just happened, and
[CLAUDE.md](CLAUDE.md) for the working agreement and invariants.

## What this is

A RAG system over a self-referential research corpus — arXiv IR/RAG/ML papers plus this
project's own decision log — where every technique is **tried and justified** rather than
cargo-culted.

The deliverable is not a pipeline. It is **an experimental apparatus that produces defensible
numbers about pipelines**, plus the best configuration it finds. Every component is registered
with a config schema, a pipeline is a YAML file, and a run is `hash(manifest) → metrics` in a
DuckDB ledger.

## Why it exists

"Try every RAG technique" is an infinite backlog until you can measure which ones work, on your
data, with confidence intervals. Most published RAG comparisons cannot support their own claims:
real ablation deltas are 1–3 points and the usual sample sizes cannot resolve them. So the
evaluation harness is built **first**, and everything else is scored against it. The primary
objective is **no confident wrong answers** — every claim traceable to a span, abstain when
uncovered.

## Current state

**Phase 0 — Foundations. Complete** (2026-09-09). **Phase 1 — Ingest pipeline. Complete** (2026-09-10).

**495 tests pass**, `ruff check .` is clean, and `scripts/check_no_fake_code.py` reports 0
violations and 0 escapes across `src/`.

Landed in Phase 0:

| Module | What it does |
|---|---|
| `ids.py` | Identity functions. `chunk_uid` hashes coordinates, never text. |
| `schema.py` | Core models; multimodal locator fields reserved as nullable. |
| `eval/` | Span-anchored relevance, versioned qrels, persisted run files, paired statistics. |
| `registry.py` | Components + `config_id`; two real reference components. |
| `ledger.py` | DuckDB. `run_metrics` holds **per-query** values, never means. |
| `paths.py`, `cli.py` | Path resolution; `doctor`, `registry list`, `ledger`, `config validate`. |
| `scripts/`, `.github/` | No-fake-code gate and CI. |

Landed in Phase 1 (`src/selfrag/ingest/`):

| Module | What it does |
|---|---|
| `manifest.py`, `arxiv_client.py`, `eprint.py` | Corpus manifest; rate-limited arXiv client; e-print fetch + extraction. |
| `latex.py`, `pdf_fallback.py` | LaTeX-source parser (primary); PyMuPDF PDF parser (fallback), same `ParsedDocument` shape. |
| `canonical.py` | Freezes parsed text once; `get_span` is the only read path for offsets. |
| `dedup.py` | arXiv version collapsing + MinHash/LSH chunk near-dup detection. |
| `quality.py` | Deterministic, no-LLM parse-quality metrics — the LaTeX-vs-PDF decision data. |
| `pipeline.py` | **The orchestrator.** Wires all of the above into one idempotent, resumable flow (see below). |
| `ledger.py` (extended) | `upsert_documents`/`upsert_chunks` (batched upserts), `get_chunk_ids`, `count_chunks`, `get_document`, `live_documents`, `tombstone_document`, `record_ingest_run`/`finish_ingest_run`. Schema now v2 (`chunks.tombstoned_at`). |
| `cli.py` (extended) | `selfrag ingest run/status/quality`. |

`selfrag config validate configs/baseline.yaml` resolves both stages and produces a concrete
`run_id`. **Phase 0's exit criterion is met.**

**Phase 1's exit criterion is met and proven in
[tests/integration/test_ingest_idempotency.py](tests/integration/test_ingest_idempotency.py):
running ingest twice over the same documents produces zero new chunk ids** — checked as id-set
identity, zero duplicate ledger rows, identical canonical text hashes, identical document rows,
zero new manifest rows, and zero re-acquisition, across two and then three consecutive runs.
Idempotency works by leaning on manifest status (`PARSED`/`ACQUIRED` skip re-fetching and
re-parsing) rather than any new tracking layer — see `pipeline.py`'s module docstring and
[ADR 0006](decisions/0006-span-integrity-failure-aborts-ingest-rather-than-dead-lettering.md).

There is no real corpus ingested yet (`configs/ingest.yaml` names one placeholder doc_id) and no
retrieval yet — retrieval is Phase 2. Phases 0–3 need **zero API keys**; Phase 1's `ingest run`
does call the live arXiv API when not run with `--dry-run`, subject to the 1-req/3s throttle
already enforced by `arxiv_client.py`.

⚠ **RAM is still tight.** `selfrag doctor` measures **~1.2–1.3 GB available** against the 3 GB
the local-model tier assumes — unchanged since Phase 0. Measure the dev-corpus size and ORT
thread count on a quiet machine before trusting the plan's throughput arithmetic.

## How to run it

```bash
uv venv --python 3.12
uv pip install --python ./.venv/Scripts/python.exe -e ".[dev]"
```

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
```

`uv pip install` needs the explicit `--python` target on this machine; without it uv resolves to
the Windows Store Python and fails on permissions.

End a work chunk with a checkpoint — see [MEMORY.md](MEMORY.md) for the routine:

```powershell
.\scripts\checkpoint.ps1 "what this chunk did"
```

## Where things live

| Path | What |
|---|---|
| `src/selfrag/ids.py` | Identity functions. The most expensive decisions to reverse. |
| `src/selfrag/schema.py` | Core models; multimodal locator fields reserved from day one. |
| `configs/` | Pipeline definitions (YAML). One file per configuration. |
| `decisions/` | ADRs — immutable, numbered. Start at [0001](decisions/0001-qrels-are-document-character-spans.md). |
| `scripts/` | CI gates and the checkpoint helper. |
| `data/` | Corpus, indexes, ledger. **Gitignored** — never redistributed (arXiv TOU). |

## Constraints

These shape every decision; they are not preferences.

- **Hardware:** CPU-only Windows 11 laptop, Intel Core 5 210H, **7.7 GB RAM**, no GPU, no Docker.
  Realistic Python working set ~2.5–3 GB.
- **Pre-registered hard limits:** peak RSS **≤ 2.5 GB**, MCP fast path **p95 ≤ 800 ms**. A config
  violating either is disqualified regardless of quality.
- **Budget:** $30–55 total for paid APIs, ceiling measurements only. Rate limits bind harder than
  prices.
- **Legal:** arXiv API TOU — ≤1 request/3 s, single connection, local storage for research only,
  **no redistribution**, answers link back to arXiv.
- **Statistical:** pre-registered minimum practical effect of +0.02 nDCG@10 / +4 pts recall@10.
  Below it, configs are `indistinguishable` and the cheaper one wins.

## Memory

- [MEMORY.md](MEMORY.md) — dated log of what happened and why
- [ERRORS.md](ERRORS.md) — failures hit, and how to recognise them next time
- [decisions/](decisions/) — numbered ADRs, immutable once accepted
- [CLAUDE.md](CLAUDE.md) — invariants and working agreement

# selfrag — context

**Last updated:** 2026-09-09 · **Repo:** https://github.com/MudMonster341/Self_RAG ·
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

**Phase 0 — Foundations. In progress.**

Landed: git repo initialised with `origin` pointing at GitHub; first commit `306f931`;
`src/selfrag/ids.py` (identity functions) and `src/selfrag/schema.py` (core Pydantic models);
`README.md`, `CLAUDE.md`, `.gitignore`, `.env.example`; the project-memory files you are reading.

Not yet built: `registry.py`, `ledger.py`, `cli.py`, `eval/`, the `configs/` pipeline
definitions, `tests/`, and the CI no-fake-code gate. `configs/`, `tests/` and `.github/` exist
but are empty.

**Phase 0 exits when** `selfrag run --config baseline.yaml` runs end to end on 10 documents and
writes a row to the ledger. Phases 0–3 need **zero API keys**.

Next after that: Phase 1 corpus ingest, then Phase 2 the eval harness — which is the real start
of the project, because nothing before it is falsifiable.

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

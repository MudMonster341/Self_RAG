---
status: accepted
date: 2026-09-09
authored_by: agent
derived_from: ["plan: selfrag — a production-grade, falsifiable RAG system (approved 2026-09-09)"]
supersedes: null
superseded_by: null
---

# 0005 — No LangChain, LlamaIndex or Haystack in the core

*Recorded retroactively on 2026-09-09 from the approved project plan.*

## Context

Every RAG project starts with the question of whether to build on a framework, and for most
projects the answer is yes — the connectors, loaders and chained abstractions are exactly what a
product team should not be writing.

This project is not that. **The product is controlled comparison between components.** The seams
that a framework exists to hide — chunk boundary policy, fusion method and its constants, rerank K,
prefilter-versus-postfilter, prompt assembly order, truncation — are precisely the variables being
measured.

The reproducibility spine makes this concrete. Every parameter that can affect a number must be
recorded **by value** in the run manifest, which is hashed into the `run_id`. A framework default
that changes between minor versions would silently change a measured result while the manifest
still reported the same hash — a reproducibility failure that is invisible by construction.

The hardware adds a second, more mundane force: a working set of ~2.5–3 GB and a pre-registered
peak-RSS ceiling of 2.5 GB. Framework import graphs and their eager, in-memory default stores are
not sized for that.

## Options considered

1. **Build the core on LlamaIndex or LangChain.** Fastest route to a working pipeline, and a large
   ecosystem of loaders and integrations. Costs: the measured seams live inside the dependency;
   defaults are not in our manifest; upgrades can move results; the dependency's control flow
   dictates ours.
2. **Use a framework for peripheral I/O only** (loaders, connectors) with our own core. Attractive
   compromise. But the peripheral parts here — arXiv OAI-PMH streaming and LaTeX source parsing —
   are the parts with no good framework support anyway, and pulling in the dependency for them
   drags the import graph in for a small benefit.
3. **No framework in the core.** Registered components, each with a Pydantic config schema; a
   pipeline is a YAML file; a run is `hash(manifest) → metrics` in the ledger.

## Decision

**No LangChain, LlamaIndex or Haystack in the core.** We read their implementations; we do not
inherit their control flow.

The replacement is not "write everything from scratch" but a registry: every stage is a registered
component with a declared config schema, so the manifest can enumerate every knob by value, and a
pipeline is a YAML file naming components and their parameters. That discipline is the product.

## Consequences

- **Makes easy:** a manifest that provably contains every parameter affecting a result. No hidden
  defaults, and no upgrade that changes a number without changing a hash.
- **Makes easy:** measuring fusion honestly. RRF-versus-convex-α is an ablation axis, so it has to
  be our code regardless of this decision — a framework would only have obscured it.
- **Makes easy:** staying inside the memory budget, with a small and auditable dependency surface.
- **Gives up:** the connector ecosystem, and speed to a first working demo. There is no shortcut
  to the baseline.
- **Costs:** we write our own chunkers, fusion, prompt assembly and evaluation. The real risk is
  reimplementing a subtly wrong version of a known algorithm — mitigated by reading the reference
  implementations before writing ours, and by unit tests on the pieces with known-correct outputs.
- **Revisit if:** a framework component becomes a *candidate on an axis* rather than the substrate.
  Benchmarking, say, LlamaIndex's semantic splitter as one rung of the chunking ladder is fully
  compatible with this ADR. Building the harness on it is not, and never will be — the harness is
  what makes the rung meaningful.

## Links

- Related: [ADR 0002 — LanceDB chosen outright](0002-lancedb-chosen-outright-no-vector-store-abstraction.md)
  (the same argument applied to storage: hidden semantics move the metric)
- Working agreement: [CLAUDE.md](../CLAUDE.md) — "no fake code" and the machine-truth/prose split

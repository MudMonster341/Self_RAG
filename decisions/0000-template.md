---
status: proposed            # proposed | accepted | rejected | superseded by NNNN
date: YYYY-MM-DD
authored_by: human          # human | agent
derived_from: []            # sources this was reasoned from
supersedes: null
superseded_by: null
---

# NNNN — <decision, stated as a decision>

<!--
  Copy this file to decisions/NNNN-kebab-case-title.md, next number in sequence.
  Numbers are allocated in strict order and are NEVER reused or renumbered — links point
  at numbers, so a reused number is a lie.

  Once status is `accepted`, this file is IMMUTABLE. The only edit it ever receives is
  flipping `status` to `superseded by NNNN` and filling in `superseded_by`. A changed mind
  is a new ADR, not an edit.

  Write it when the decision is made, while the rejected options are still remembered.
  Reconstructing them weeks later produces fiction.
-->

## Context

The forces at play. What made this a decision rather than a default: the constraint, the
disagreement, or the thing that would be expensive to reverse. Include the numbers that mattered —
"too slow" is not a force, "4–8 pairs/s, so top-50 costs 6–12 s/query" is.

## Options considered

1. **<Option A>** — what it is, and its concrete cost or risk.
2. **<Option B>** — same.
3. **<Option C>** — same.

Only options actually weighed belong here. Inventing plausible alternatives after the fact makes
this record worse than silence.

## Decision

One sentence, active voice, stating what was chosen. Then the argument for it, in terms of the
forces named in Context — not in terms of general good practice.

## Consequences

- What this makes easy.
- What this makes hard, or gives up. **An ADR with no downside listed has not been thought
  through.**
- **Revisit if:** the tripwire — the observation that should trigger a superseding ADR. Name a
  condition that could actually be observed, not "if requirements change".

## Links

- Related: [ADR NNNN](NNNN-....md)
- Failure that prompted it: [ERR-NNNN](../ERRORS.md)
- Implemented in: `src/selfrag/....py`

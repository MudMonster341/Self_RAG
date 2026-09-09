#!/usr/bin/env bash
# checkpoint.sh — end-of-work-chunk commit and push.
#
# Part of the `project-memory` discipline. Run this AFTER you have:
#   1. appended a dated entry to MEMORY.md
#   2. added any new ADRs under decisions/
#   3. appended any new ERRORS.md entries
#   4. refreshed CONTEXT.md if the current state changed
#
# This script does step 5: stage, commit with a conventional-commit message, push.
#
# It FAILS LOUDLY (non-zero exit + explicit message) when it cannot push — no remote, no
# upstream, detached HEAD, missing credentials, rejected push. The local commit is made first,
# so a failed push never loses work.
#
# It NEVER uses --force or --no-verify. If a hook rejects the commit, fix the hook's complaint.
#
# Usage:
#   scripts/checkpoint.sh "summary of this work chunk"
#   scripts/checkpoint.sh -t feat -s eval "add paired bootstrap"
#
# Options:
#   -t TYPE   conventional-commit type (default: chore)
#   -s SCOPE  conventional-commit scope (default: checkpoint)
#   -d        docs-only: stage only CONTEXT.md, MEMORY.md, ERRORS.md, decisions/
#   -n        dry run: show what would happen, change nothing
#   -M        skip the MEMORY.md discipline check (use only when genuinely warranted)
#   -h        this help
#
# Exit codes: 0 ok · 1 usage/not a repo · 2 MEMORY.md check failed · 3 cannot push
#             4 commit failed · 5 push failed

set -euo pipefail

TYPE="chore"
SCOPE="checkpoint"
DOCS_ONLY=0
DRY_RUN=0
SKIP_MEMORY_CHECK=0

die() { printf '\n=== CHECKPOINT FAILED ===\n%s\n\n' "$1" >&2; exit "${2:-1}"; }
say() { printf '%s\n' "$1"; }

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts ":t:s:dnMh" opt; do
  case "$opt" in
    t) TYPE="$OPTARG" ;;
    s) SCOPE="$OPTARG" ;;
    d) DOCS_ONLY=1 ;;
    n) DRY_RUN=1 ;;
    M) SKIP_MEMORY_CHECK=1 ;;
    h) usage 0 ;;
    :) die "option -$OPTARG needs a value" 1 ;;
    \?) die "unknown option -$OPTARG (try -h)" 1 ;;
  esac
done
shift $((OPTIND - 1))

SUMMARY="${*:-}"
[ -n "$SUMMARY" ] || die "No summary given.

  Usage: scripts/checkpoint.sh \"summary of this work chunk\"

  The summary becomes the commit subject. Describe the chunk, not the files." 1

# --- repo -------------------------------------------------------------------
git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "Not inside a git repository.

  Run this from within the project. If the project is not a git repo yet, initialise it and
  add a remote first — a checkpoint that cannot be pushed is not a checkpoint." 1

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HAS_HEAD=0
git rev-parse --verify --quiet HEAD >/dev/null && HAS_HEAD=1

# --- MEMORY.md discipline check --------------------------------------------
if [ "$SKIP_MEMORY_CHECK" -eq 0 ]; then
  TODAY="$(date +%F)"
  [ -f MEMORY.md ] || die "MEMORY.md not found at $ROOT.

  This project has not been bootstrapped for project-memory. Create CONTEXT.md, MEMORY.md,
  ERRORS.md and decisions/ first, or re-run with -M to skip this check." 2

  grep -q "^## ${TODAY}" MEMORY.md || die "MEMORY.md has no entry dated ${TODAY}.

  A checkpoint without a memory entry is just a commit. Append an entry beginning:

      ## ${TODAY} — <short title of this work chunk>

  Then re-run. To checkpoint anyway, re-run with -M." 2

  # --porcelain, not `git diff HEAD`: an untracked MEMORY.md does not appear in a diff.
  if [ "$HAS_HEAD" -eq 1 ] && [ -z "$(git status --porcelain -- MEMORY.md)" ]; then
    die "MEMORY.md is unchanged since the last commit.

  Today's entry was already committed; this work chunk has no entry of its own. Append a new
  dated entry for it, or re-run with -M." 2
  fi
fi

# --- what to stage ----------------------------------------------------------
if [ "$DOCS_ONLY" -eq 1 ]; then
  PATHSPEC=(-- CONTEXT.md MEMORY.md ERRORS.md decisions)
else
  PATHSPEC=(-- .)
fi

# --- push target, checked BEFORE committing so the failure is predictable ----
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
PUSH_BLOCKER=""
REMOTE=""
UPSTREAM=""

if [ "$BRANCH" = "HEAD" ]; then
  PUSH_BLOCKER="HEAD is detached — there is no branch to push.
  Check out a branch (git switch -c <name>) and re-run."
elif [ -z "$(git remote)" ]; then
  PUSH_BLOCKER="This repository has no git remote.
  Add one, then re-run:

      git remote add origin https://github.com/<user>/<repo>.git"
else
  UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name "@{upstream}" 2>/dev/null || true)"
  if [ -n "$UPSTREAM" ]; then
    REMOTE="${UPSTREAM%%/*}"
  elif git remote | grep -qx origin; then
    REMOTE="origin"
  else
    REMOTE="$(git remote | head -n 1)"
  fi
fi

# --- stage ------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  say "[dry run] would stage:"
  git status --short "${PATHSPEC[@]}"
  say "[dry run] would commit: ${TYPE}(${SCOPE}): ${SUMMARY}"
  say "[dry run] would push to: ${REMOTE:-<none>} ${BRANCH}"
  [ -n "$PUSH_BLOCKER" ] && say "[dry run] push would FAIL: ${PUSH_BLOCKER}"
  exit 0
fi

git add -A "${PATHSPEC[@]}"

# --- commit -----------------------------------------------------------------
SUBJECT="${TYPE}(${SCOPE}): ${SUMMARY}"
[ "${#SUBJECT}" -le 72 ] || say "warning: commit subject is ${#SUBJECT} chars (>72). Consider a shorter summary."

if git diff --cached --quiet; then
  if [ "$HAS_HEAD" -eq 0 ]; then
    die "Nothing staged and no commits yet — there is nothing to checkpoint." 1
  fi
  say "Nothing staged; skipping commit."
else
  # No --no-verify: hooks are part of the project's quality gate.
  git commit -m "$SUBJECT" || die "git commit failed.

  Nothing has been pushed. If a pre-commit hook rejected this, fix what it reported and re-run.
  Do not bypass hooks with --no-verify." 4
  say "Committed: $SUBJECT"
fi

# --- push -------------------------------------------------------------------
[ -z "$PUSH_BLOCKER" ] || die "Committed locally, but CANNOT PUSH.

  ${PUSH_BLOCKER}

  Your work is safe in the local commit above. It is not backed up until this is fixed." 3

if [ -n "$UPSTREAM" ]; then
  if git rev-list --count "@{upstream}..HEAD" 2>/dev/null | grep -qx 0; then
    say "Already up to date with ${UPSTREAM}; nothing to push."
    exit 0
  fi
  git push "$REMOTE" "$BRANCH" || die "git push to ${REMOTE}/${BRANCH} failed.

  Your work is safe in the local commit; it is NOT on the remote yet.

  Common causes:
    - No credentials. Set up a credential helper or an SSH key, then re-run.
    - Non-fast-forward: the remote has commits you do not.
      Fix with:  git pull --rebase ${REMOTE} ${BRANCH}   then re-run this script.
      Never with --force: that destroys the history this system exists to preserve." 5
else
  say "Branch '${BRANCH}' has no upstream; setting it to ${REMOTE}/${BRANCH}."
  git push --set-upstream "$REMOTE" "$BRANCH" || die "git push --set-upstream ${REMOTE} ${BRANCH} failed.

  Your work is safe in the local commit; it is NOT on the remote yet.

  Check that the remote exists and that you have credentials for it:
      git remote -v
      git ls-remote ${REMOTE}" 5
fi

say "Pushed ${BRANCH} to ${REMOTE}. Checkpoint complete."

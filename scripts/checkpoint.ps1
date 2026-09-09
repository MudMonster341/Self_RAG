<#
.SYNOPSIS
  End-of-work-chunk commit and push. Windows PowerShell 5.1 compatible.

.DESCRIPTION
  Part of the `project-memory` discipline. Run this AFTER you have:
    1. appended a dated entry to MEMORY.md
    2. added any new ADRs under decisions/
    3. appended any new ERRORS.md entries
    4. refreshed CONTEXT.md if the current state changed

  This script does step 5: stage, commit with a conventional-commit message, push.

  It FAILS LOUDLY (non-zero exit + explicit message) when it cannot push — no remote, no
  upstream, detached HEAD, missing credentials, rejected push. The local commit is made first,
  so a failed push never loses work.

  It NEVER uses --force or --no-verify. If a hook rejects the commit, fix the hook's complaint.

  Exit codes: 0 ok · 1 usage/not a repo · 2 MEMORY.md check failed · 3 cannot push
              4 commit failed · 5 push failed

.PARAMETER Summary
  The commit subject. Describe the work chunk, not the files.

.PARAMETER Type
  Conventional-commit type. Default: chore.

.PARAMETER Scope
  Conventional-commit scope. Default: checkpoint.

.PARAMETER DocsOnly
  Stage only CONTEXT.md, MEMORY.md, ERRORS.md and decisions/.

.PARAMETER DryRun
  Show what would happen; change nothing.

.PARAMETER SkipMemoryCheck
  Skip the MEMORY.md discipline check. Use only when genuinely warranted.

.EXAMPLE
  .\scripts\checkpoint.ps1 "wire up the DuckDB ledger"

.EXAMPLE
  .\scripts\checkpoint.ps1 -Type feat -Scope eval "add paired bootstrap CI"
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $Summary,

    [string] $Type = 'chore',
    [string] $Scope = 'checkpoint',
    [switch] $DocsOnly,
    [switch] $DryRun,
    [switch] $SkipMemoryCheck
)

function Stop-Checkpoint {
    param([string] $Message, [int] $Code = 1)
    Write-Host ''
    Write-Host '=== CHECKPOINT FAILED ===' -ForegroundColor Red
    Write-Host $Message -ForegroundColor Red
    Write-Host ''
    exit $Code
}

$summaryText = ''
if ($Summary) { $summaryText = ($Summary -join ' ').Trim() }
if (-not $summaryText) {
    Stop-Checkpoint @'
No summary given.

  Usage: .\scripts\checkpoint.ps1 "summary of this work chunk"

  The summary becomes the commit subject. Describe the chunk, not the files.
'@ 1
}

# --- repo -------------------------------------------------------------------
$null = git rev-parse --is-inside-work-tree
if ($LASTEXITCODE -ne 0) {
    Stop-Checkpoint @'
Not inside a git repository.

  Run this from within the project. If the project is not a git repo yet, initialise it and
  add a remote first - a checkpoint that cannot be pushed is not a checkpoint.
'@ 1
}

$root = (git rev-parse --show-toplevel).Trim()
Set-Location -LiteralPath $root

$null = git rev-parse --verify --quiet HEAD
$hasHead = ($LASTEXITCODE -eq 0)

# --- MEMORY.md discipline check --------------------------------------------
if (-not $SkipMemoryCheck) {
    $today = Get-Date -Format 'yyyy-MM-dd'
    $memoryPath = Join-Path $root 'MEMORY.md'

    if (-not (Test-Path -LiteralPath $memoryPath)) {
        Stop-Checkpoint "MEMORY.md not found at $root.

  This project has not been bootstrapped for project-memory. Create CONTEXT.md, MEMORY.md,
  ERRORS.md and decisions/ first, or re-run with -SkipMemoryCheck." 2
    }

    $hasTodayEntry = Select-String -LiteralPath $memoryPath -Pattern "^## $today" -Quiet
    if (-not $hasTodayEntry) {
        Stop-Checkpoint "MEMORY.md has no entry dated $today.

  A checkpoint without a memory entry is just a commit. Append an entry beginning:

      ## $today - <short title of this work chunk>

  Then re-run. To checkpoint anyway, re-run with -SkipMemoryCheck." 2
    }

    if ($hasHead) {
        # --porcelain, not `git diff HEAD`: an untracked MEMORY.md does not appear in a diff.
        $memoryStatus = @(git status --porcelain -- MEMORY.md | Where-Object { $_ })
        if ($memoryStatus.Count -eq 0) {
            Stop-Checkpoint @'
MEMORY.md is unchanged since the last commit.

  Today's entry was already committed; this work chunk has no entry of its own. Append a new
  dated entry for it, or re-run with -SkipMemoryCheck.
'@ 2
        }
    }
}

# --- what to stage ----------------------------------------------------------
if ($DocsOnly) {
    $pathspec = @('--', 'CONTEXT.md', 'MEMORY.md', 'ERRORS.md', 'decisions')
} else {
    $pathspec = @('--', '.')
}

# --- push target, checked BEFORE committing so the failure is predictable ----
$branch = (git symbolic-ref --quiet --short HEAD)
$detached = ($LASTEXITCODE -ne 0)
if ($branch) { $branch = $branch.Trim() }

$pushBlocker = ''
$remote = ''
$upstream = ''

$remotes = @(git remote | Where-Object { $_ })

if ($detached -or -not $branch) {
    $pushBlocker = "HEAD is detached - there is no branch to push.`n  Check out a branch (git switch -c <name>) and re-run."
    $branch = 'HEAD'
} elseif ($remotes.Count -eq 0) {
    $pushBlocker = @'
This repository has no git remote.
  Add one, then re-run:

      git remote add origin https://github.com/<user>/<repo>.git
'@
} else {
    # for-each-ref never writes to stderr and is empty when there is no upstream
    $upstream = (git for-each-ref --format='%(upstream:short)' "refs/heads/$branch")
    if ($upstream) { $upstream = $upstream.Trim() }

    if ($upstream) {
        $remote = $upstream.Split('/')[0]
    } elseif ($remotes -contains 'origin') {
        $remote = 'origin'
    } else {
        $remote = $remotes[0]
    }
}

# --- dry run ----------------------------------------------------------------
$subject = "$Type($Scope): $summaryText"

if ($DryRun) {
    Write-Host '[dry run] would stage:'
    git status --short @pathspec
    Write-Host "[dry run] would commit: $subject"
    $remoteLabel = '<none>'
    if ($remote) { $remoteLabel = $remote }
    Write-Host "[dry run] would push to: $remoteLabel $branch"
    if ($pushBlocker) { Write-Host "[dry run] push would FAIL: $pushBlocker" -ForegroundColor Yellow }
    exit 0
}

# --- stage ------------------------------------------------------------------
git add -A @pathspec
if ($LASTEXITCODE -ne 0) { Stop-Checkpoint 'git add failed; nothing was committed.' 1 }

# --- commit -----------------------------------------------------------------
if ($subject.Length -gt 72) {
    Write-Host "warning: commit subject is $($subject.Length) chars (>72). Consider a shorter summary." -ForegroundColor Yellow
}

$null = git diff --cached --quiet
$nothingStaged = ($LASTEXITCODE -eq 0)

if ($nothingStaged) {
    if (-not $hasHead) { Stop-Checkpoint 'Nothing staged and no commits yet - there is nothing to checkpoint.' 1 }
    Write-Host 'Nothing staged; skipping commit.'
} else {
    # No --no-verify: hooks are part of the project's quality gate.
    git commit -m $subject
    if ($LASTEXITCODE -ne 0) {
        Stop-Checkpoint @'
git commit failed.

  Nothing has been pushed. If a pre-commit hook rejected this, fix what it reported and re-run.
  Do not bypass hooks with --no-verify.
'@ 4
    }
    Write-Host "Committed: $subject" -ForegroundColor Green
}

# --- push -------------------------------------------------------------------
if ($pushBlocker) {
    Stop-Checkpoint "Committed locally, but CANNOT PUSH.

  $pushBlocker

  Your work is safe in the local commit above. It is not backed up until this is fixed." 3
}

if ($upstream) {
    $ahead = (git rev-list --count "$upstream..HEAD").Trim()
    if ($ahead -eq '0') {
        Write-Host "Already up to date with $upstream; nothing to push."
        exit 0
    }
    git push $remote $branch
    if ($LASTEXITCODE -ne 0) {
        Stop-Checkpoint "git push to $remote/$branch failed.

  Your work is safe in the local commit; it is NOT on the remote yet.

  Common causes:
    - No credentials. Set up Git Credential Manager or an SSH key, then re-run.
    - Non-fast-forward: the remote has commits you do not.
      Fix with:  git pull --rebase $remote $branch   then re-run this script.
      Never with --force: that destroys the history this system exists to preserve." 5
    }
} else {
    Write-Host "Branch '$branch' has no upstream; setting it to $remote/$branch."
    git push --set-upstream $remote $branch
    if ($LASTEXITCODE -ne 0) {
        Stop-Checkpoint "git push --set-upstream $remote $branch failed.

  Your work is safe in the local commit; it is NOT on the remote yet.

  Check that the remote exists and that you have credentials for it:
      git remote -v
      git ls-remote $remote" 5
    }
}

Write-Host "Pushed $branch to $remote. Checkpoint complete." -ForegroundColor Green
exit 0

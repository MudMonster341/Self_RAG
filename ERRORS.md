# Errors worth remembering

Append-only. Newest at the bottom. **Only non-obvious failures** — ones where the symptom did not
point at the cause, or where the fix is something that would not be rediscovered quickly. A log of
typos is noise, and noise is what stops a file like this being read.

Format: `ERR-NNNN` allocated in order, never reused. See [CONTEXT.md](CONTEXT.md) for orientation
and [MEMORY.md](MEMORY.md) for the running log.

---

## ERR-0001 — Shell quoting swallows Python source in Git Bash (2026-09-09)

**Symptom:**

```
bash: unexpected EOF while looking for matching quote
```

The reported line number pointed at a line of Python that was perfectly valid. Nothing was written
to disk, and nothing ran — but the failure looked like a Python error, so the first ten minutes
were spent debugging the wrong language.

**Context:** Windows 11, Git Bash, while scaffolding Phase 0. A single command both wrote a Python
module with a bash heredoc *and* ran a smoke test through an inline `python -c "..."` string. The
Python contained apostrophes in docstrings and nested quotes in string literals — ordinary,
correct Python.

**Root cause:** the shell, not Python. Python source is being parsed twice: once by bash and once
by the interpreter. Apostrophes, nested quotes and backticks inside an inline `python -c "..."`
argument are consumed by bash's own quoting rules before Python ever sees them, and an unbalanced
quote after bash's pass terminates parsing of the whole command — including the heredoc that was
supposed to follow. Any code containing a contraction, a possessive, or a nested quote is a
tripwire, and both Windows shells here (Git Bash and PowerShell) have their own incompatible
escaping rules, so a working incantation on one is not portable to the other.

**Fix:** **never inline Python source into a shell string.** Write the file first — with the Write
tool, or a heredoc using a *quoted* delimiter (`<<'PY'`, which disables all shell expansion inside
the body) — then run it as a file:

```bash
python path/to/smoke_test.py       # correct
python -c "import selfrag; ..."    # never
```

Running from a file also means the failing code is on disk and can be re-run, edited and diffed,
which an inline string never can be.

[CLAUDE.md](CLAUDE.md) states this as a standing rule for this machine, in the stronger form:
prefer the Write tool over heredocs entirely for Python files.

**Recognise it next time:** a *shell* error — `unexpected EOF while looking for matching quote`,
`unexpected token`, or a heredoc that silently never terminates — reported against code that is
valid in its own language. The tell is that the quoted line number makes no sense. When that
happens, stop reading the inner language and count the quotes in the outer one.

---

## ERR-0002 — Scheduled task "ran" at 03:14 and did nothing; machine was in Modern Standby (2026-09-10)

**Symptom:**

```
taskId:    selfrag-phase1-resume
lastRunAt: 2026-09-09T23:14:11.937Z   <- stamped, on time, to the second
enabled:   false                      <- auto-disabled, i.e. "already fired"
```

Every indicator said the task ran. Nothing existed to show for it: no commits, no
`data/` artifacts, no `MEMORY.md` entry, no log file anywhere under
`~/.claude/scheduled-tasks/`. The task had spent its single one-shot.

**Root cause:** the machine was asleep. Three independent failures compounded:

1. **Modern Standby.** The system entered Modern Standby at **23:20:41** local — six minutes
   after the task was scheduled — and did not exit until **06:38:00**. The 03:14 fire time
   fell squarely inside that window. Note that `powercfg SUB_SLEEP STANDBYIDLE` reads `0`
   ("never") on AC: **Modern Standby is not governed by the classic idle-sleep timeout**, so
   the setting that looks like it prevents sleep does not.
2. **The app-level scheduler cannot wake the machine.** It arms an in-process timer, not an
   RTC wake timer. Only Windows Task Scheduler with "wake the computer to run this task" arms
   a real hardware wake. Wake timers being *enabled* (`RTCWAKE = 1`) is necessary but not
   sufficient — something has to actually arm one.
3. **Windows Update rebooted the box three times** between 04:30 and 04:32
   (`TrustedInstaller.exe`, "Operating System: Upgrade (Planned)"). Even a never-sleep
   configuration would have lost the session.

**Fix:** do not rely on the desktop app's scheduler for unattended overnight work on this
machine. Either (a) keep the machine explicitly awake *and* set Windows Update active hours
for the run window, or (b) push the repo and use a cloud routine, which is immune to the
local machine's power state. (b) is the durable answer — which makes the unpushed remote a
correctness problem for scheduling, not merely a missing backup.

**How to recognise this next time:** a scheduled task that reports a punctual `lastRunAt`,
shows as disabled/completed, and leaves **no log at all** did not fail — it never executed.
A run that genuinely started and then errored leaves a trace. Check
`Get-WinEvent -ProviderName Microsoft-Windows-Kernel-Power` for events 506/507 around the
fire time before assuming the task's own logic was at fault.

**Note:** this exact risk was written down in the plan's Known Open Risks ("the
unattended-overnight assumption depends on the laptop not sleeping; power settings need
checking before the first long run") and then not acted on before scheduling. Identifying a
risk is not mitigating it.

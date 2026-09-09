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

---
name: modify-code
description: Action for changing the behaviour of one existing function, class, or script, after the tests have been updated to specify the change. Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# modify-code

Change the behaviour of one existing function, class, or script.

**Kind:** behaviour-changing.

## Arguments (the plan must give these; if missing, ask)
- The file to change (exactly one)
- A description of the change
- Whether any public signature changes, and if so the callers to update in later `modify-code` steps

## Precondition
For a function or class, a preceding `write-test` step has already updated the tests to specify the new behaviour, so the human has reviewed the change as a spec. The file exists. If not, stop and report.

## May touch
Only the named file.

## Must not
- Edit any test file.
- Touch any other file.
- Break the rules of the action that created the code: a function stays pure with one function per file (see `write-function`), a class keeps its family rules (see `write-class`), a script keeps its script rules (see `write-script`).
- Change a public signature unless the plan says so.

## Checks (run them and report the results)
- The target's tests pass. If a public signature changed, the full suite passes once the last listed caller has been updated.
- The rules of the original action still hold (e.g. the `ast` checks for scripts and functions).
- A script is checked by running it on example input (see `run-code`).

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`modify-code: <file path>`

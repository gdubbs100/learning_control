---
name: refactor
description: Action for changing the structure of existing code without changing its behaviour, one operation at a time (rename, move, extract-function, inline, delete-dead-code). Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# refactor

Change the structure of existing code without changing its behaviour.

**Kind:** behaviour-preserving.

## Arguments (the plan must give these; if missing, ask)
- `operation`: one of `rename`, `move`, `extract-function`, `inline`, `delete-dead-code`
- The target (file and symbol)
- The new name or location where relevant

One step does one operation.

## Precondition
The full test suite passes before the step. Run it first. If it does not pass, stop and report.

## May touch
The source files that define or reference the target, and nothing else. The only permitted change to a test file is the mechanical update of an imported name or path caused by a `rename` or `move`, or the removal of tests for code removed by `delete-dead-code`.

## Must not
- Change behaviour, add features, or change logic.
- Edit test logic or expected values.
- Combine more than one operation in one step.
- Break the rules of the action that created the code. In particular, an extracted function follows `write-function`, and a separate `write-test(function)` step covers it afterwards.

## Checks (run them and report the results)
- The full test suite passes after the step, with no change to what the tests assert.
- `delete-dead-code`: a search finds no remaining references to the deleted code.
- `rename` and `move`: a search finds no remaining references to the old name or path.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`refactor: <operation> <target>`

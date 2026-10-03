---
name: run-code
description: Read-only action for running the test suite or a script and reporting the result, without fixing anything. Use as a step of an approved plan built from the project's actions (see actions.md), or to check a script or the tests.
---

# run-code

Run the tests or a script and report what happened.

**Kind:** read-only.

## Arguments (the plan must give these; if missing, ask)
- `target`: `tests` (all, a file, or a pattern) or `script` (path and command-line arguments)

## May touch
Nothing in the source tree. A script may write the output it is specified to produce, and nothing else. Outputs are not committed unless the plan says so.

## Must not
Change any source or test file, or attempt fixes. A failure is reported back to the human.

## Report
- Tests: passes, failures, and errors with their messages.
- Script: the exit status, the output it produced, and any error text.

## Commit message
None (no commit).

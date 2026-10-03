---
name: write-function
description: Action for creating one new pure Python function in its own standalone file, for scripts and other functions to import. Use only as a step of an approved plan built from the project's actions (see actions.md), after its paired write-test step.
---

# write-function

Create a single pure function in its own file.

**Kind:** behaviour-changing.

## Arguments (the plan must give these; if missing, ask)
- File path (may sit in a themed folder, e.g. `utils/plotting/`)
- Function name and purpose
- Input names and types; output type

## Precondition
Any types the function takes or returns, and any functions it calls, already exist and are importable. The paired `write-test(function)` step has been done. If something is missing, stop and report.

## May touch
Only the new function file (plus package `__init__.py` files if a new folder is needed).

## Must not
- Define more than one function in the file, or any class.
- Have side effects: no file or network I/O, printing, global state, or mutating its arguments.
- Read from the environment (command line, environment variables, current time, unseeded randomness).
- Modify any other file.

## Style
- Inputs and outputs are in-code objects (floats, ints, arrays, custom classes), never paths or command-line values.
- A docstring states the purpose, the inputs, and the output.
- All parameters and the return value have type hints.
- Names are descriptive.
- Do not write tests here. The paired `write-test(function)` action does that.

## Checks (run them and report the results)
- Parsing the file with `ast` finds exactly one top-level function definition and no class definitions.
- The function has a docstring.
- All parameters and the return value have type hints.
- Every imported name resolves.
- The paired `write-test(function)` passes (this covers determinism and no mutation of inputs).

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`write-function: <file path>`

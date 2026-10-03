---
name: write-test
description: Action for writing tests under tests/ for a function, a class family contract, or a single class, before the code exists, so the expected behaviour can be reviewed as a spec. Also used to change or remove listed test cases before a modify-code step. Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# write-test

Create tests for code we write, before the code exists.

**Kind:** test-only.

## Arguments (the plan must give these; if missing, ask)
- `target`: one of `function`, `class-contract`, `class`
- The code under test: a function (file path and name), a class family (its file), or a class (family file and name)
- `function` and `class`: a list of cases with the expected results (inputs and expected output, or construction arguments, method calls, and expected results), plus tolerances for floating-point comparisons
- `class-contract`: no cases, since the checks are generic
- Before a `modify-code` step: the specific cases to add, change, or remove

## Precondition
The target's file path, name, and signature are fixed by the plan. For `class-contract` and `class`, the family file and base class already exist. The code under test need not exist yet. Until it does, the tests fail on import, which is expected.

## May touch
Only files under `tests/`, mirroring the source path (e.g. `tests/utils/plotting/test_<name>.py`). For `class`, this is the family's test file, where tests are added for the new class and existing tests are left unchanged. The exception is a change to existing behaviour: when the plan lists specific cases to change or remove, those cases may be edited and nothing else.

## Must not
- Modify any source file.
- Take expected values from running the implementation. The plan supplies them.
- Use I/O, mocks, or unseeded randomness.
- In `class-contract`, contain behaviour specific to one child class (that belongs in `class`).

## What each target covers
- `function`: the plan's cases, plus generic checks in every file: calling the function twice with the same input gives the same output, and the input is unchanged afterwards.
- `class-contract`: one parametrised test that runs against every class in the family file, so new children are covered automatically. For each class it checks that it instantiates, every public method of the base class is present with a compatible signature, and `__init__` follows the extension rule (accepts all base parameters, calls `super().__init__()`, added parameters are keyword-only).
- `class`: the plan's cases for the specific behaviour of one child class.

## Checks (run them and report the results)
Every case in the plan appears as a test. Once the code exists, the tests pass under `pytest`. Before it exists, report that they fail on import and nothing else.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`write-test: <target> <path or name>`

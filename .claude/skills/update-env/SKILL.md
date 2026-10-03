---
name: update-env
description: Action for changing the existing uv environment and its dependencies - add, remove or upgrade packages, or change the Python version. The only action that changes dependencies. Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# update-env

Change the existing `uv` environment and its dependencies. This is the only action that changes dependencies.

**Kind:** behaviour-changing

## Arguments (the plan must give these; if missing, ask)
- `operation`: one of `add`, `remove`, `upgrade`, `change-python`, `sync`
- For `add`, `remove`, `upgrade`: the packages (with version constraints and dependency group, e.g. `dev`)
- For `change-python`: the new version

## Precondition
The environment exists (`.venv/` and `uv.lock` are present). If not, stop and use `setup-env`.

## May touch
`pyproject.toml`, `uv.lock`, and `.venv/`. Changes go through `uv add`, `uv remove`, `uv lock --upgrade-package`, and `uv sync`, not hand edits.

## Must not
- Combine more than one operation in one step.
- Edit any `.py` file. Code that needs to change after a removal or upgrade is handled by its own action.
- Recreate the environment from scratch, unless the operation is `change-python`.

## Checks (run them and report the results)
- `uv sync` succeeds and `uv lock --check` reports the lockfile is up to date.
- The full test suite still runs. Failures are reported, not fixed.
- `remove`: a search finds no remaining imports of the package. Any found are reported to the human.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`update-env: <operation> <packages>`

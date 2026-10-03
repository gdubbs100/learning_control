---
name: setup-env
description: Action for creating the project's Python environment with uv. Refuses if an environment already exists (use update-env). Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# setup-env

Create the project's Python environment with `uv`, from the project's dependency files.

**Kind:** behaviour-changing

## Arguments (the plan must give these; if missing, ask)
- Python version (default: the version in `pyproject.toml`, or the latest stable)
- Dependency groups to install (default: all, including dev/test)
- Environment location (default `.venv`)

## Precondition
`uv` is installed. No environment exists yet (no `.venv/`). If one does, stop and use `update-env`.

## May touch
`.venv/`, `uv.lock`, and `pyproject.toml` only if it does not exist yet (created with `uv init`).

## Must not
- Edit an existing `pyproject.toml`, or any source or test file.
- Install packages by any route outside `uv` (e.g. `pip`).
- Overwrite an existing environment.

## Checks (run them and report the results)
- `uv sync` succeeds.
- `uv run python --version` matches the requested version.
- `uv run pytest --collect-only` runs without import errors.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`setup-env: <python version>` (commits `uv.lock` and `pyproject.toml` if created; `.venv/` is gitignored).

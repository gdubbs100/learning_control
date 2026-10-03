---
name: update-project-files
description: Action for changing non-Python files: documentation or configuration. Never touches .py files or dependencies (use update-env). Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# update-project-files

Change files that are not Python code. Dependencies and the environment are handled by `update-env`.

**Kind:** depends on `target`: `docs` is behaviour-preserving; `config` is behaviour-changing.

## Arguments (the plan must give these; if missing, ask)
- `target`: one of `docs`, `config`
- The file(s)
- The change

## Precondition
The named files exist or are named for creation.

## May touch
Only the named non-Python files (e.g. `README.md`, `actions.md`).

## Must not
- Touch any `.py` file. Docstrings belong to code and are changed by the code's own actions.
- Touch `pyproject.toml`'s dependencies, `uv.lock`, or `.venv/`. Use `update-env`.

## Checks (run them and report the results)
- `docs`: links and referenced paths resolve.
- `config`: the full test suite still passes.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`update-project-files: <target> <file path>`

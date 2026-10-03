---
name: explore-code
description: Read-only action for answering a question about the codebase by reading and searching it, reporting findings with file_path:line_number references. Use as the first step of a plan on existing code, or whenever the answer is needed before choosing the next action (see actions.md).
---

# explore-code

Answer a question about the codebase by reading and searching it. Change nothing.

**Kind:** read-only.

## Arguments (the plan must give these; if missing, ask)
- The question
- Optionally where to look

## May touch
Nothing.

## Must not
- Edit, create, or delete any file.
- Run scripts or tests (that is `run-code`).

## Allowed tools
Any read-only means: file reading, glob and content search, `git log` and `git blame`, and `ast`-based inspection of structure and imports.

## Report
Answer the question directly and cite `file_path:line_number` for each claim. Say what you could not find.

## Commit message
None (no commit).

---
name: write-script
description: Action for creating a new Python script (a file that takes input such as command-line arguments and produces an output such as an image, CSV or model). Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# write-script

Create one Python script. A script defines nothing: it imports everything and only runs control flow.

**Kind:** behaviour-changing.

## Arguments (the plan must give these; if missing, ask)
- Script path (chosen per task, no fixed location)
- Inputs (names, types, source)
- Outputs (what is produced and where)
- The existing functions it will import and call

## Precondition
Every function or class the script needs already exists and is importable. If one is missing, stop and report it. A separate action must create it first.

## May touch
Only the new script file.

## Must not
- Define any function, class, or lambda. Everything is imported.
- Contain logic beyond control flow (sequencing, loops, conditionals) and calls to imported functions.
- Use complex comprehensions or conditional expressions. Simple ones are allowed.
- Parse arguments itself (e.g. calling `argparse` directly). Argument parsing lives in an imported helper.
- Modify any other file.

## Structure and style
- The script body sits under an `if __name__ == "__main__":` guard.
- The control flow must read clearly to a human from the names alone. Use descriptive function and variable names so the script reads like a short description of what it does. Avoid cryptic abbreviations and inline computation.

## Checks (run them and report the results)
- Parsing the file with `ast` finds no `FunctionDef`, `AsyncFunctionDef`, `ClassDef`, or `Lambda` nodes.
- The `if __name__ == "__main__":` guard is present.
- Every imported name resolves.
- Running the script on example input produces the stated output.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`write-script: <script path>`

---
name: structured-plan
description: Plan and carry out any task that will change code in this project, using only the project's atomic actions (explore-code, write-test, write-function, initialise-new-class, write-class, write-script, modify-code, refactor, run-code, setup-env, update-env, update-project-files) instead of free-form editing. Use whenever the user asks for a code change, new feature, bug fix, or refactor, before touching any file.
---

# structured-plan

All code changes in this project are made by a small set of named, atomic actions, each defined by a skill and summarised in `actions.md`. The point of the restricted action space is that a human can follow and comment on every edit. A plan is an ordered list of these actions. Do not write code outside an action.

## When to use
Any task that creates or changes files in this project. Pure questions need no plan: use `explore-code` or just answer.

## The actions
| Action | Kind | Use for |
|---|---|---|
| `explore-code` | read-only | Answer a question about the codebase |
| `run-code` | read-only | Run the tests or a script and report |
| `write-test` | test-only | Tests for a function, a class family contract, or a class (before the code) |
| `write-function` | behaviour-changing | One new pure function in its own file |
| `initialise-new-class` | behaviour-changing | New class family: abstract base class in a new file |
| `write-class` | behaviour-changing | Concrete child class, added to its base class's file |
| `write-script` | behaviour-changing | New script: only imports and control flow |
| `modify-code` | behaviour-changing | Change the behaviour of one existing function, class, or script |
| `refactor` | behaviour-preserving | Rename, move, extract-function, inline, or delete-dead-code, one per step |
| `setup-env` | behaviour-changing | Create the `uv` environment (only if none exists) |
| `update-env` | behaviour-changing | Add, remove, or upgrade dependencies, or change the Python version, one operation per step |
| `update-project-files` | depends on target | Docs or config (never `.py` files or dependencies) |

Each action's skill states its arguments, preconditions, what it may touch, what it must not do, and its checks. Read the skill for each action before using it.

## Procedure

### 0. Load all the actions
Before drafting, read `actions.md` so every action in it is available to the plan. The table above must list every action in `actions.md`; if they differ, tell the user and treat `actions.md` as the source of truth.

### 1. Understand the task
If the task touches existing code, start with `explore-code` to find the relevant files and functions. Ask the user only about things you cannot find out or sensibly default.

### 2. Draft the plan
Write a numbered list, one action per line, with its arguments and a short reason:

```
1. explore-code("where are trajectories plotted?")
2. write-test(function, utils/plotting/plot_trajectory, cases=...): spec the new plotting function
3. write-function(utils/plotting/plot_trajectory, ...): implement it
4. write-script(scripts/plot_run.py, ...): script that plots a run
5. run-code(script, scripts/plot_run.py, ...): check the script works
```

Rules for the plan:
- **Every step is exactly one action** from the table, with its required arguments filled in. Unknown arguments are asked about, not guessed.
- **One change per step.** Each step touches only what its action may touch.
- **Tests come first and are paired:**
  - Each `write-function` is preceded by its `write-test(function)`.
  - Each `initialise-new-class` is followed by a `write-test(class-contract)` for that family.
  - Each `write-class` is preceded by its `write-test(class)`.
  - Each `modify-code` on a function or class is preceded by a `write-test` that lists the cases to add, change, or remove.
  - Scripts have no paired test. Check them with `run-code`.
  - A `refactor` is not paired with new tests.
- **Dependencies come first.** Anything a step imports must exist by then, created by an earlier step. Scripts come last, since they import everything else.
- **Class families:** a class with several parents needs a new base class from `initialise-new-class` that combines them, then `write-class` from that base.
- **Plans on existing code** start with `explore-code`.
- **Restrictions apply to code we write.** Third-party code (e.g. `gymnasium` environments) may be used in scripts and imported by our code even if it does not follow the restrictions.

### 3. Check the plan before showing it
Go through every step and confirm:
- The action name is in the table.
- Its arguments are complete.
- Its preconditions will hold given the steps before it.
- It touches nothing outside its allowed scope.
- The pairing and ordering rules above hold.

Fix the plan until it passes.

### 4. Get approval
Show the plan to the user and wait for approval or comments. Do not edit any file before the plan is approved. Revise the plan as they comment.

### 5. Carry out the plan one action at a time
For each step, in order:
1. Invoke the action's skill and follow it exactly.
2. Run the action's checks and report the results.
3. Stop and report if any check fails, a precondition does not hold, or an argument is missing. Do not work around it.
4. Report briefly what the step did and its commit message (`<action>: <target>`), then move to the next step. Commit only when the user has asked for commits.

### 6. When no action fits
If the work needs something no action allows (or a step needs to go outside its scope), stop. Do not improvise. Say what is missing and propose a new action or a change to an existing one in `actions.md`, and let the user decide.

### 7. When the plan changes
If what you learn mid-way means the plan is wrong, stop, explain, and propose the revised plan for approval. Do not silently reorder, skip, or add steps.

## Principles to hold to
- A restricted action space makes your edits easier for a human to follow and comment on. Staying inside it matters more than finishing quickly.
- Each action is either behaviour-preserving or behaviour-changing, never both.
- Each action states what it must not do, so a reviewer knows what to look for. Check against that list.
- Keep the plan as short as the task allows. Do not add steps for the sake of completeness.

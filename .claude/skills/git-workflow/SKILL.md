---
name: git-workflow
description: Manages git for a structured plan - creates a branch when a plan is approved, commits after each action, pushes when the plan is finished, and opens a pull request once the user confirms the work is final. Use alongside structured-plan whenever a plan changes files in this project.
---

# git-workflow

One plan = one branch = one pull request. This skill wraps a `structured-plan`; it never changes code itself, and it never replaces an action.

## When to use
- At the start of carrying out an approved plan (stage 1).
- After each action passes its checks (stage 2).
- When the last step of the plan is done (stage 3).
- When the user says the work is final (stage 4).

Plans with no file changes (only `explore-code` / `run-code`) need no branch.

## Stage 1: branch (once the plan is approved, before step 1)
1. `git status`. If there are uncommitted changes that are not part of this plan, stop and ask the user what to do with them. Never stash, discard or commit them silently.
2. If on `main` or another plan's branch, `git fetch origin` and create the new branch from the up-to-date `origin/main`: `git switch -c plan/<short-kebab-slug> origin/main`. If the user says to continue on the current branch (e.g. it is already this plan's branch), do that instead.
3. The slug is 2 to 5 words describing the task (e.g. `plan/record-last-episode-video`). Tell the user the branch name.

## Stage 2: commit (after each action passes its checks)
- Commit only when the action's checks have passed. If a check failed and the plan is stopped, do not commit.
- Stage only the files that action may touch (by name, never `git add -A` or `git add .`). Do not stage outputs, `.venv/`, or unrelated changes.
- Use the action's commit message from `actions.md` (`<action>: <target>`), followed by the attribution line given in the session's system-reminder, if any.
- One commit per action step. Never amend, and never skip hooks.
- Read-only actions (`explore-code`, `run-code`) make no commit.
- Commits stay local until stage 3.

## Stage 3: push (when the last step of the plan is done)
1. Run the full test suite one last time (`run-code(tests)`) and report the result. If it fails, stop and report; do not push.
2. `git status` should show a clean tree apart from known ignored outputs. Report anything left over.
3. `git push -u origin <branch>`. Never force-push, and never push to `main`.
4. Report the branch, the list of commits, and the test result, then ask the user to review. Do not open a PR yet.

## Stage 4: pull request (only after the user says the work is final)
- If the user asks for changes, make them through a new plan step or a new plan (using the actions), commit, and push to the same branch. Then ask again.
- When the user confirms, open the PR against `main`:
  - Use `gh pr create` if `gh` is installed and authenticated.
  - Otherwise give the user the compare URL (`https://github.com/<owner>/<repo>/compare/main...<branch>`, derived from `git remote get-url origin`) and the title and body to paste.
- Title: short, in the imperative, describing the task.
- Body: what the plan did (the numbered actions, one line each), how it was checked (tests, script run), and anything the reviewer should look at. End with the PR attribution line given in the session's system-reminder, if any.
- Report the PR URL. Do not merge it.

## Must not
- Commit, push or open a PR on `main`.
- Force-push, rewrite history, delete branches, or skip hooks.
- Push or open a PR without the user having asked for this workflow in the session. Opening the PR always waits for the user's confirmation (stage 4).
- Stage files outside the action's allowed scope, or commit secrets or generated outputs.
- Change any code, test, or doc. Fixes go through the plan's actions.

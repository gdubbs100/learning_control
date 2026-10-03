# Actions

This document defines the allowed set of atomic actions for building code in this project. It is a planning document: once stable, each action will be implemented as a skill.

## Principles

- Every code change is made by one named action. No free-form editing.
- A plan for a task is an ordered list of actions with their arguments.
- If no action fits, stop and propose a new action here. Do not improvise.
- Each action is either behaviour-preserving or behaviour-changing, never both.
- Test actions are `test-only`: they touch only files under `tests/`, never source files.
- Tests are written as we go. Every function and class is paired with a test action in the plan (see `Planning rule`).
- Each action states what it must not do, so a reviewer knows what to look for.
- The restrictions apply to code we write. Third-party code (e.g. `gymnasium` environments) can be used in scripts and imported by our code even if it does not follow these restrictions.

## Action template

```
### action-name
**Status:** draft | agreed
**Kind:** read-only | behaviour-preserving | behaviour-changing | test-only
**Purpose:** one sentence.
**Arguments:** what the plan must specify.
**Precondition:** what must be true before.
**May touch:** files/scope allowed to change.
**Must not:** explicit prohibitions.
**Postcondition / check:** how the reviewer (or a command) verifies it.
**Commit message:** format.
```

## Actions

### write-script
**Status:** draft
**Kind:** behaviour-changing
**Purpose:** Create a Python script: a file that takes input (e.g. command line arguments) and produces an output (e.g. an image, CSV, model).
**Arguments:** script path (chosen per task, no fixed location); inputs (names, types, source); outputs (what is produced and where); the existing functions it will import and call.
**Precondition:** Every function or class the script needs already exists and is importable. If one is missing, a separate action must create it first.
**May touch:** only the new script file.
**Must not:**
- Define any function, class, or lambda. Everything is imported.
- Contain logic beyond control flow (sequencing, loops, conditionals) and calls to imported functions.
- Use complex comprehensions or conditional expressions. Simple ones are allowed.
- Parse arguments itself (e.g. calling `argparse` directly). Argument parsing lives in an imported helper.
- Modify any other file.

**Structure:** The script body sits under an `if __name__ == "__main__":` guard (currently required, may be relaxed later).
**Style:** The control flow must read clearly to a human from the names alone. Use descriptive function and variable names, so the script reads like a short description of what it does. Avoid cryptic abbreviations and inline computation.
**Postcondition / check:**
- Parsing the file with `ast` finds no `FunctionDef`, `AsyncFunctionDef`, `ClassDef`, or `Lambda` nodes.
- The `if __name__ == "__main__":` guard is present.
- Every imported name resolves.
- Running the script on example input produces the stated output.
**Commit message:** `write-script: <script path>`

### write-function
**Status:** draft
**Kind:** behaviour-changing
**Purpose:** Create a single pure function in its own standalone Python file, for scripts and other functions to import.
**Arguments:** file path (may sit in a themed folder, e.g. `utils/plotting/`); function name; purpose; input names and types; output type.
**Precondition:** Any types the function takes or returns, and any functions it calls, already exist and are importable.
**May touch:** only the new function file (plus package `__init__.py` files if a new folder is needed).
**Must not:**
- Define more than one function in the file, or any class.
- Have side effects: no file or network I/O, printing, global state, or mutating its arguments.
- Read from the environment (command line, environment variables, current time, unseeded randomness).
- Modify any other file.

**Style:** Inputs and outputs are in-code objects (floats, ints, arrays, custom classes), never paths or command-line values. The function has a docstring stating its purpose, its inputs, and its output. All parameters and the return value have type hints. Names are descriptive. Tests are written by the paired `write-test(function)` action, not by this one.
**Postcondition / check:**
- Parsing the file with `ast` finds exactly one top-level function definition and no class definitions.
- The function has a docstring.
- All parameters and the return value have type hints.
- Every imported name resolves.
- The paired `write-test(function)` passes (this covers determinism and no mutation of inputs).
**Commit message:** `write-function: <file path>`

### initialise-new-class
**Status:** draft
**Kind:** behaviour-changing
**Purpose:** Start a new class family: create the base class for a type of object (e.g. an agent, an environment, a model) in a new file. It specifies the main methods that every class of that type must have. Concrete classes are added to the same file by `write-class`.
**Arguments:** file path; class name; purpose of the type; the main methods (name, parameter types, return type, purpose); the properties (attributes) all members share; existing base classes to combine (only when the new family must inherit from more than one, see below).
**Precondition:** Any types used in the method signatures, and any base classes to combine, already exist and are importable.
**May touch:** only the new class file (plus package `__init__.py` files if a new folder is needed).
**Must not:**
- Implement the main methods. They are abstract and only specify the interface. The `__init__` method may be concrete, to set the properties all members share.
- Inherit from anything except `abc.ABC`, or the existing base classes listed in the arguments.
- Modify any other file, including the files of the base classes it combines.

**Multiple inheritance:** This is the only action that creates a class with more than one parent. When a class would need to inherit from several base classes, first use this action to create a new base class that inherits from all of them and handles how they combine. Then use `write-class` to create the class from that new base class.

**Style:** The class and each method have a docstring stating purpose, inputs, and outputs. All parameters, return values, and attributes have type hints.
**Postcondition / check:**
- The file contains one class. It inherits from `ABC` or from the listed base classes, and every main method is marked `@abstractmethod`.
- The class cannot be instantiated.
- Docstrings and type hints are present.
- Every imported name resolves.
**Commit message:** `initialise-new-class: <file path>`

### write-class
**Status:** draft
**Kind:** behaviour-changing
**Purpose:** Create a class that inherits from an existing base class. It is a concrete member of that type of object. The class is added to the same file as its base class, so each file holds one class family.
**Arguments:** base class (this also fixes the file); class name; purpose; any new properties; methods to override; new methods to add.
**Precondition:** The base class exists, is importable, and has its own family file created by `initialise-new-class`.
**May touch:** only the base class's file, by adding the new class. Existing classes in the file are left unchanged.
**Must not:**
- Inherit from more than one class. If that is needed, first use `initialise-new-class` to create a base class that combines them.
- Remove any method from the base class. This includes replacing one with `None`, `del`, or a body that only raises `NotImplementedError`.
- Leave any abstract method of the base class unimplemented.
- Change `__init__` except by extension: it must accept every base `__init__` parameter unchanged and pass them to `super().__init__()`. It may add new parameters, which must be keyword-only.
- Change the signature of an overridden method in a way that breaks callers of the base class.
- Modify any other file, or change any existing class in the file.

**Style:** The class and each method have a docstring. All parameters, return values, and attributes have type hints. Overridden methods say how they differ from the base class. Names are descriptive.
**Postcondition / check:**
- The class has exactly one parent, the stated base class, and sits in the base class's file.
- The diff only adds the new class; existing code in the file is unchanged.
- The class can be instantiated (all abstract methods are implemented).
- Every public method of the base class is still present and callable with the base class signature.
- If `__init__` is defined, it accepts all base `__init__` parameters, calls `super().__init__()`, and any added parameters are keyword-only.
- Docstrings and type hints are present.
- Every imported name resolves.
**Commit message:** `write-class: <file path>`

### write-test
**Status:** draft
**Kind:** test-only
**Purpose:** Create tests for code we write, before the code exists, so the expected behaviour can be reviewed as a specification.
**Arguments:**
- `target`: one of `function`, `class-contract`, `class`.
- The code under test: a function (file path and name), a class family (its file), or a class (family file and name).
- `function` and `class`: a list of cases with the expected results (inputs and expected output, or construction arguments, method calls, and expected results), plus tolerances for floating-point comparisons.
- `class-contract`: no cases, since the checks are generic.

**Precondition:** The target's file path, name, and signature are fixed by the plan. For `class-contract` and `class`, the family file and base class already exist (created by `initialise-new-class`). The code under test need not exist yet; until it does, the tests fail on import.
**May touch:** only files under `tests/`, mirroring the source path (e.g. `tests/utils/plotting/test_<name>.py`). For `class`, this is the family's test file, where tests are added for the new class and existing tests are left unchanged. The exception is a change to existing behaviour: when the plan lists specific cases to change or remove (used before `modify-code`), those cases may be edited and nothing else.
**Must not:**
- Modify any source file.
- Take expected values from running the implementation. The plan supplies them.
- Use I/O, mocks, or unseeded randomness.
- In `class-contract`, contain behaviour specific to one child class (that belongs in `class`).

**What each target covers:**
- `function`: the plan's cases, plus generic checks in every file: calling the function twice with the same input gives the same output, and the input is unchanged afterwards.
- `class-contract`: one parametrised test that runs against every class in the family file, so new children are covered automatically. For each class it checks that it instantiates, every public method of the base class is present with a compatible signature, and `__init__` follows the extension rule.
- `class`: the plan's cases for the specific behaviour of one child class.

**Postcondition / check:** Every case in the plan appears as a test, and once the code exists the tests pass under `pytest`.
**Commit message:** `write-test: <target> <path or name>`

### modify-code
**Status:** draft
**Kind:** behaviour-changing
**Purpose:** Change the behaviour of one existing function, class, or script.
**Arguments:** the file to change (exactly one); a description of the change; whether any public signature changes (and, if so, the callers to update in later `modify-code` steps).
**Precondition:** For a function or class, a preceding `write-test` step has already updated the tests to specify the new behaviour, so the human has reviewed the change as a spec. The file exists.
**May touch:** only the named file.
**Must not:**
- Edit any test file.
- Touch any other file.
- Break the rules of the action that created the code (a function stays pure with one function per file, a class keeps its family rules, a script keeps its script rules).
- Change a public signature unless the plan says so.

**Postcondition / check:**
- The target's tests pass. If a public signature changed, the full suite passes once the last listed caller has been updated.
- The rules of the original action still hold (e.g. the AST checks for scripts and functions).
- A script is checked by running it on example input (see `run-code`).
**Commit message:** `modify-code: <file path>`

### refactor
**Status:** draft
**Kind:** behaviour-preserving
**Purpose:** Change the structure of existing code without changing its behaviour.
**Arguments:** `operation`: one of `rename`, `move`, `extract-function`, `inline`, `delete-dead-code`; the target (file and symbol); the new name or location where relevant. One step does one operation.
**Precondition:** The full test suite passes before the step.
**May touch:** the source files that define or reference the target, and nothing else. The only permitted change to a test file is the mechanical update of an imported name or path caused by a `rename` or `move`, or the removal of tests for code removed by `delete-dead-code`.
**Must not:**
- Change behaviour, add features, or change logic.
- Edit test logic or expected values.
- Combine more than one operation in one step.
- Break the rules of the action that created the code. In particular, an extracted function follows `write-function`, and a separate `write-test(function)` step covers it afterwards.

**Postcondition / check:**
- The full test suite passes after the step, with no change to what the tests assert.
- `delete-dead-code`: a search finds no remaining references to the deleted code.
- `rename` and `move`: a search finds no remaining references to the old name or path.
**Commit message:** `refactor: <operation> <target>`

### explore-code
**Status:** draft
**Kind:** read-only
**Purpose:** Answer a question about the codebase by reading and searching it.
**Arguments:** the question; optionally where to look.
**Precondition:** none.
**May touch:** nothing.
**Must not:** edit, create, or delete any file; run scripts or tests (that is `run-code`).
**Allowed tools:** any read-only means: file reading, glob and content search, `git log` and `git blame`, `ast`-based inspection of structure and imports.
**Postcondition / check:** The report answers the question and cites `file_path:line_number` for each claim.
**Commit message:** none (no commit).

### run-code
**Status:** draft
**Kind:** read-only
**Purpose:** Run the tests or a script and report the result.
**Arguments:** `target`: `tests` (all, a file, or a pattern) or `script` (path and command-line arguments).
**Precondition:** none.
**May touch:** nothing in the source tree. A script may write the output it is specified to produce, and nothing else. Outputs are not committed unless the plan says so.
**Must not:** change any source or test file, or attempt fixes. A failure is reported back to the human.
**Postcondition / check:** The report lists, for tests, passes, failures, and errors with their messages; for a script, the exit status, the output it produced, and any error text.
**Commit message:** none (no commit).

### update-project-files
**Status:** draft
**Kind:** depends on `target`: `docs` is behaviour-preserving; `dependencies` and `config` are behaviour-changing.
**Purpose:** Change files that are not Python code: documentation, dependencies, and configuration.
**Arguments:** `target`: one of `docs`, `dependencies`, `config`; the file(s); the change.
**Precondition:** The named files exist or are named for creation.
**May touch:** only the named non-Python files (e.g. `README.md`, `actions.md`, `pyproject.toml`).
**Must not:** touch any `.py` file. Docstrings belong to code and are changed by the code's own actions.
**Postcondition / check:**
- `docs`: links and referenced paths resolve.
- `dependencies`: the environment installs and the full test suite still runs.
- `config`: the full test suite still passes.
**Commit message:** `update-project-files: <target> <file path>`

## Open questions

- `write-script`: what counts as a "simple" comprehension or conditional expression (e.g. one line, no nesting)?
- `write-script`: revisit whether the `__main__` guard should stay required.
- `write-script`: revisit whether argument parsing must always live in an imported helper.
- `write-class`: may classes hold mutable state (an agent or environment probably must), or should they be immutable by default? Does "pure" apply to methods?
- `write-class`: do added `__init__` parameters need defaults? Only matters if objects are built generically (e.g. from a config). Decide when we first need that.
- `write-class`: do data-only classes (e.g. a config or a state) also need a base class, or can they be plain dataclasses?
- Tests: is `tests/` mirroring the source tree, with `pytest` as the runner, right? (Assumed in the test actions.)
- Tests: is property-based testing (`hypothesis`) allowed for functions, or example-based only?
- Tests: do functions need a shared contract test like class families have?
- Scripts have no test action. They are checked by running them on example input (see `write-script`). Revisit if that proves too weak.
- Impure code (writing files, reading the command line): the purity rule in `write-function` may need bending for these. Monitor this as we work, then decide whether to add an exception or a separate action. This includes the argument-parsing helper that `write-script` imports.
- Plotting: a pure plotting function returns a figure and does not save it. Saving is a separate concern. Revisit with the impure-code question.
- `refactor`: tests need mechanical import updates for `rename` and `move`, so I relaxed "no test edits" to that narrow case. Acceptable?
- `refactor`: `extract-function` creates a new function that needs its own test. Should the extracted function's test be a required follow-up step in every plan?
- `modify-code`: when a signature changes, the suite may be red until callers are updated. Allow that, or require the signature change and all callers in one step?
- `update-project-files`: version policy for dependencies (pinned, ranges)? Is `actions.md` itself covered by `docs`?
- `run-code`: where should script outputs go, and should they be ignored by git?
- Granularity: how fine should actions be?
- One commit per action?
- Review after every action, or in batches?
- Escape hatch process for when no action fits.
- Whether to enforce any actions with hooks.

## Planning rule (draft)

A plan is a numbered list, one action per line, e.g.:

1. `action-name(arg1, arg2)`: why this step exists

Tests are paired with the code they cover, test first:

- Each `write-function` is preceded by its `write-test(function)`.
- Each `initialise-new-class` is followed by a `write-test(class-contract)` for that family.
- Each `write-class` is preceded by its `write-test(class)`.
- Each `modify-code` on a function or class is preceded by a `write-test` that lists the cases to add, change, or remove.
- Scripts have no paired test. They are checked with `run-code`.
- A plan on existing code starts with `explore-code`, and a `refactor` is not paired with new tests.

Example:

1. `explore-code("where are trajectories plotted?")`
2. `write-test(function, utils/plotting/plot_trajectory, cases=...)`
3. `write-function(utils/plotting/plot_trajectory, ...)`
4. `write-script(scripts/plot_run.py, ...)`
5. `run-code(script, scripts/plot_run.py, ...)`

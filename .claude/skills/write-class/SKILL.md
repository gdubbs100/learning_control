---
name: write-class
description: Action for adding a concrete class that inherits from an existing base class, placed in the base class's file (one class family per file). Use only as a step of an approved plan built from the project's actions (see actions.md), after its paired write-test step.
---

# write-class

Create a class that inherits from an existing base class. It is a concrete member of that type of object and is added to the same file as its base class.

**Kind:** behaviour-changing.

## Arguments (the plan must give these; if missing, ask)
- Base class (this also fixes the file)
- Class name and purpose
- Any new properties
- Methods to override, and new methods to add

## Precondition
The base class exists, is importable, and has its own family file created by `initialise-new-class`. The paired `write-test(class)` step has been done. If not, stop and report.

## May touch
Only the base class's file, by adding the new class. Existing classes in the file are left unchanged.

## Must not
- Inherit from more than one class. If that is needed, first use `initialise-new-class` to create a base class that combines them.
- Remove any method from the base class. This includes replacing one with `None`, `del`, or a body that only raises `NotImplementedError`.
- Leave any abstract method of the base class unimplemented.
- Change `__init__` except by extension: it must accept every base `__init__` parameter unchanged and pass them to `super().__init__()`. It may add new parameters, which must be keyword-only.
- Change the signature of an overridden method in a way that breaks callers of the base class.
- Modify any other file, or change any existing class in the file.

## Style
The class and each method have a docstring. All parameters, return values, and attributes have type hints. Overridden methods say how they differ from the base class. Names are descriptive.

## Checks (run them and report the results)
- The class has exactly one parent, the stated base class, and sits in the base class's file.
- The diff only adds the new class. Existing code in the file is unchanged.
- The class can be instantiated (all abstract methods are implemented).
- Every public method of the base class is still present and callable with the base class signature.
- If `__init__` is defined, it accepts all base `__init__` parameters, calls `super().__init__()`, and any added parameters are keyword-only.
- Docstrings and type hints are present.
- Every imported name resolves.
- The paired `write-test(class)` and the family's `write-test(class-contract)` pass.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`write-class: <file path>`

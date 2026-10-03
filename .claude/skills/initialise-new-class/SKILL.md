---
name: initialise-new-class
description: Action for starting a new class family by creating its abstract base class in a new file (e.g. an agent, environment or model type). Also used to create a base class that combines several existing base classes. Use only as a step of an approved plan built from the project's actions (see actions.md).
---

# initialise-new-class

Create the base class for a type of object in a new file. It specifies the main methods that every class of that type must have. Concrete classes are added to the same file later by `write-class`.

**Kind:** behaviour-changing.

## Arguments (the plan must give these; if missing, ask)
- File path and class name
- Purpose of the type
- The main methods (name, parameter types, return type, purpose)
- The properties (attributes) all members share
- Existing base classes to combine (only when the new family must inherit from more than one)

## Precondition
Any types used in the method signatures, and any base classes to combine, already exist and are importable. If not, stop and report.

## May touch
Only the new class file (plus package `__init__.py` files if a new folder is needed).

## Must not
- Implement the main methods. They are abstract and only specify the interface. The `__init__` method may be concrete, to set the properties all members share.
- Inherit from anything except `abc.ABC`, or the existing base classes listed in the arguments.
- Modify any other file, including the files of the base classes it combines.

## Multiple inheritance
This is the only action that creates a class with more than one parent. When a class would need to inherit from several base classes, first use this action to create a new base class that inherits from all of them and handles how they combine. Then use `write-class` to create the class from that new base class.

## Style
The class and each method have a docstring stating purpose, inputs, and outputs. All parameters, return values, and attributes have type hints.

## Checks (run them and report the results)
- The file contains one class. It inherits from `ABC` or from the listed base classes, and every main method is marked `@abstractmethod`.
- The class cannot be instantiated.
- Docstrings and type hints are present.
- Every imported name resolves.

The plan should follow this with `write-test(class-contract)` for the family.

If a check fails, fix it within the scope above. If you cannot, stop and report. If the task seems to need something outside this action, stop and propose a new action rather than improvising.

## Commit message
`initialise-new-class: <file path>`

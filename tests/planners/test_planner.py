import inspect

import numpy as np
import pytest

import planners.planner as family_module
from planners.planner import Planner

BASE_CLASS = Planner
# Arguments for the base class __init__ parameters, used to construct every class in the family.
BASE_INIT_ARGUMENTS = {"horizon": 3, "action_space_size": 2, "target_state": np.zeros(4)}
# Values for keyword-only parameters that child classes add, chosen by type annotation.
EXTRA_ARGUMENT_VALUES_BY_TYPE = {int: 0, float: 0.0, str: "", bool: False}


def concrete_family_classes() -> list[type]:
    """Every concrete class in the family file that inherits from the base class."""
    return [
        member
        for _, member in inspect.getmembers(family_module, inspect.isclass)
        if issubclass(member, BASE_CLASS)
        and member is not BASE_CLASS
        and member.__module__ == family_module.__name__
        and not inspect.isabstract(member)
    ]


def construction_arguments(cls: type) -> dict:
    """Base arguments plus a typed value for each required added keyword-only parameter."""
    arguments = dict(BASE_INIT_ARGUMENTS)
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        if name in arguments or name == "self":
            continue
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        if parameter.default is inspect.Parameter.empty:
            arguments[name] = EXTRA_ARGUMENT_VALUES_BY_TYPE[parameter.annotation]
    return arguments


def public_methods(cls: type) -> dict[str, object]:
    """Public methods defined on a class, including inherited ones."""
    return {
        name: member
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def test_base_class_is_abstract() -> None:
    assert inspect.isabstract(BASE_CLASS)


@pytest.mark.parametrize("cls", concrete_family_classes(), ids=lambda cls: cls.__name__)
def test_family_class_follows_contract(cls: type) -> None:
    # Exactly one parent: the base class.
    assert cls.__bases__ == (BASE_CLASS,)

    # It instantiates and the base class attributes are set from the base arguments.
    instance = cls(**construction_arguments(cls))
    for name, value in BASE_INIT_ARGUMENTS.items():
        assert getattr(instance, name) == value

    # Every public method of the base class is present with a compatible signature.
    for name, base_method in public_methods(BASE_CLASS).items():
        child_method = getattr(cls, name, None)
        assert callable(child_method), f"{cls.__name__} is missing {name}"
        base_parameters = list(inspect.signature(base_method).parameters.values())
        child_parameters = list(inspect.signature(child_method).parameters.values())
        for base_parameter, child_parameter in zip(base_parameters, child_parameters):
            assert child_parameter.name == base_parameter.name
            assert child_parameter.kind == base_parameter.kind
        assert len(child_parameters) >= len(base_parameters)
        for extra_parameter in child_parameters[len(base_parameters):]:
            assert extra_parameter.default is not inspect.Parameter.empty, (
                f"{cls.__name__}.{name} adds required parameter {extra_parameter.name}"
            )

    # __init__ follows the extension rule.
    if "__init__" in vars(cls):
        base_init_parameters = inspect.signature(BASE_CLASS.__init__).parameters
        child_init_parameters = inspect.signature(cls.__init__).parameters
        for name, base_parameter in base_init_parameters.items():
            assert name in child_init_parameters, f"__init__ drops base parameter {name}"
            assert child_init_parameters[name].kind == base_parameter.kind
            assert child_init_parameters[name].default == base_parameter.default
        for name, child_parameter in child_init_parameters.items():
            if name not in base_init_parameters:
                assert child_parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
                    f"added __init__ parameter {name} must be keyword-only"
                )
        assert "super().__init__(" in inspect.getsource(cls.__init__)

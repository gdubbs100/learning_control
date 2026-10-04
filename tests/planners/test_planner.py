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
        np.testing.assert_array_equal(getattr(instance, name), value)

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


# RandomShootingPlanner

from models.dynamics_model import DynamicsModel
from planners.planner import RandomShootingPlanner


class OneDimensionalModel(DynamicsModel):
    """A fixed 1-D model s' = s + 0.5 u, used so the best action sequences are known."""

    def __init__(self) -> None:
        super().__init__(state_dim=1)

    def fit(self, states: np.ndarray, inputs: np.ndarray, next_states: np.ndarray) -> dict[str, float]:
        return {"train_mse": 0.0}

    def predict(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return states + 0.5 * inputs[..., None]

    def parameters(self) -> dict[str, float]:
        return {}

    def evaluate(self, states: np.ndarray, inputs: np.ndarray, next_states: np.ndarray) -> dict[str, float]:
        return {"mse": 0.0}

    def is_fitted(self) -> bool:
        return True


def make_planner(horizon: int = 3, target: float = 0.0, seed: int = 0) -> RandomShootingPlanner:
    """A RandomShootingPlanner for the 1-D model with 200 samples."""
    return RandomShootingPlanner(
        horizon=horizon,
        action_space_size=2,
        target_state=np.array([target]),
        num_samples=200,
        seed=seed,
    )


def test_random_shooting_pushes_down_from_positive_state() -> None:
    plan = make_planner().plan(OneDimensionalModel(), np.array([3.0]))
    np.testing.assert_array_equal(plan, [0, 0, 0])


def test_random_shooting_pushes_up_from_negative_state() -> None:
    plan = make_planner().plan(OneDimensionalModel(), np.array([-3.0]))
    np.testing.assert_array_equal(plan, [1, 1, 1])


def test_random_shooting_moves_towards_non_zero_target() -> None:
    plan = make_planner(horizon=1, target=2.0).plan(OneDimensionalModel(), np.array([0.0]))
    np.testing.assert_array_equal(plan, [1])


def test_random_shooting_output_has_horizon_length_and_valid_actions() -> None:
    plan = make_planner(horizon=5).plan(OneDimensionalModel(), np.array([0.7]))
    assert plan.shape == (5,)
    assert np.issubdtype(plan.dtype, np.integer)
    assert set(plan.tolist()) <= {0, 1}


def test_random_shooting_same_seed_gives_same_plan() -> None:
    first_plan = make_planner(seed=3).plan(OneDimensionalModel(), np.array([0.7]))
    second_plan = make_planner(seed=3).plan(OneDimensionalModel(), np.array([0.7]))
    np.testing.assert_array_equal(first_plan, second_plan)


def test_random_shooting_does_not_mutate_state_or_target() -> None:
    state = np.array([0.7])
    target = np.array([0.2])
    planner = RandomShootingPlanner(
        horizon=3, action_space_size=2, target_state=target, num_samples=50, seed=0
    )
    planner.plan(OneDimensionalModel(), state)
    np.testing.assert_array_equal(state, [0.7])
    np.testing.assert_array_equal(target, [0.2])
    np.testing.assert_array_equal(planner.target_state, [0.2])


# RandomShootingPlanner.hyperparameters


def test_random_shooting_hyperparameters() -> None:
    planner = RandomShootingPlanner(
        horizon=7,
        action_space_size=2,
        target_state=np.array([0.5, -1.0]),
        num_samples=321,
        seed=9,
    )
    assert planner.hyperparameters() == {
        "type": "RandomShootingPlanner",
        "horizon": 7,
        "action_space_size": 2,
        "target_state": [0.5, -1.0],
        "num_samples": 321,
        "seed": 9,
    }


def test_random_shooting_hyperparameters_are_json_serialisable() -> None:
    import json

    planner = RandomShootingPlanner(horizon=3, action_space_size=2, target_state=np.zeros(4))
    assert json.loads(json.dumps(planner.hyperparameters())) == planner.hyperparameters()

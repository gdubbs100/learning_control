import inspect

import gymnasium as gym
import pytest

import envs.cost_wrapper as family_module
from envs.cost_wrapper import CostWrapper

BASE_CLASS = CostWrapper
# Arguments for the base class __init__ parameters, used to construct every class in the family.
BASE_INIT_ARGUMENTS = {"env": gym.make("CartPole-v1")}
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


# QuadraticCostCartPole

import numpy as np

from envs.cost_wrapper import QuadraticCostCartPole

ACTION_SEQUENCE = [1, 0, 1, 1, 0]
TOLERANCE = 1e-12


def test_quadratic_cost_cartpole_compute_reward_is_negative_cost_for_both_actions() -> None:
    wrapper = QuadraticCostCartPole(gym.make("CartPole-v1"))
    observation = np.array([1.0, 2.0, 0.0, 0.0])
    assert wrapper.compute_reward(observation, 0) == -6.0
    assert wrapper.compute_reward(observation, 1) == -6.0


def test_quadratic_cost_cartpole_step_reward_is_negative_cost_of_next_observation() -> None:
    wrapper = QuadraticCostCartPole(gym.make("CartPole-v1"))
    wrapper.reset(seed=0)
    for action in ACTION_SEQUENCE:
        observation, reward, terminated, truncated, _ = wrapper.step(action)
        expected_reward = -(float(np.sum(observation.astype(np.float64) ** 2)) + 1.0)
        assert abs(reward - expected_reward) < TOLERANCE
        if terminated or truncated:
            break


def test_quadratic_cost_cartpole_respects_non_zero_target_state() -> None:
    wrapper = QuadraticCostCartPole(gym.make("CartPole-v1"), target_state=np.array([1.0, 0.0, 0.0, 0.0]))
    wrapper.reset(seed=0)
    observation, reward, _, _, _ = wrapper.step(1)
    state = observation.astype(np.float64)
    expected_reward = -((state[0] - 1.0) ** 2 + state[1] ** 2 + state[2] ** 2 + state[3] ** 2 + 1.0)
    assert abs(reward - expected_reward) < TOLERANCE


def test_quadratic_cost_cartpole_default_target_is_zeros() -> None:
    wrapper = QuadraticCostCartPole(gym.make("CartPole-v1"))
    np.testing.assert_array_equal(wrapper.target_state, np.zeros(4))


def test_quadratic_cost_cartpole_reset_matches_unwrapped_env() -> None:
    wrapped_observation, _ = QuadraticCostCartPole(gym.make("CartPole-v1")).reset(seed=7)
    plain_observation, _ = gym.make("CartPole-v1").reset(seed=7)
    np.testing.assert_array_equal(wrapped_observation, plain_observation)


def test_quadratic_cost_cartpole_only_changes_the_reward() -> None:
    wrapper = QuadraticCostCartPole(gym.make("CartPole-v1"))
    plain_env = gym.make("CartPole-v1")
    wrapper.reset(seed=0)
    plain_env.reset(seed=0)
    for action in ACTION_SEQUENCE:
        wrapped_step = wrapper.step(action)
        plain_step = plain_env.step(action)
        np.testing.assert_array_equal(wrapped_step[0], plain_step[0])
        assert wrapped_step[2] == plain_step[2]
        assert wrapped_step[3] == plain_step[3]
        if wrapped_step[2] or wrapped_step[3]:
            break


def test_quadratic_cost_cartpole_does_not_mutate_target_argument() -> None:
    target = np.array([0.5, 0.0, 0.0, 0.0])
    wrapper = QuadraticCostCartPole(gym.make("CartPole-v1"), target_state=target)
    wrapper.reset(seed=0)
    wrapper.step(1)
    np.testing.assert_array_equal(target, [0.5, 0.0, 0.0, 0.0])

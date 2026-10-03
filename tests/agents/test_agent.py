import inspect

import pytest

import agents.agent as family_module
from agents.agent import Agent

BASE_CLASS = Agent
# Arguments for the base class __init__ parameters, used to construct every class in the family.
BASE_INIT_ARGUMENTS = {"action_space_size": 2}
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


# RandomAgent

import numpy as np

from agents.agent import RandomAgent

CARTPOLE_LIKE_OBSERVATION = np.zeros(4, dtype=np.float32)


def select_actions(agent: Agent, number_of_calls: int) -> list[int]:
    """Call select_action repeatedly on the same observation and collect the actions."""
    return [agent.select_action(CARTPOLE_LIKE_OBSERVATION) for _ in range(number_of_calls)]


def test_random_agent_actions_are_in_action_space() -> None:
    agent = RandomAgent(action_space_size=2, seed=0)
    actions = select_actions(agent, 100)
    assert all(action in (0, 1) for action in actions)


def test_random_agent_same_seed_gives_identical_sequences() -> None:
    first_agent = RandomAgent(action_space_size=2, seed=0)
    second_agent = RandomAgent(action_space_size=2, seed=0)
    assert select_actions(first_agent, 50) == select_actions(second_agent, 50)


def test_random_agent_different_seeds_give_different_sequences() -> None:
    first_agent = RandomAgent(action_space_size=2, seed=0)
    second_agent = RandomAgent(action_space_size=2, seed=1)
    assert select_actions(first_agent, 50) != select_actions(second_agent, 50)


def test_random_agent_both_actions_appear() -> None:
    agent = RandomAgent(action_space_size=2, seed=0)
    actions = select_actions(agent, 100)
    assert set(actions) == {0, 1}


# Agent learning and logging hooks (no-ops by default, inherited by RandomAgent)


def test_random_agent_observe_transition_returns_none() -> None:
    agent = RandomAgent(action_space_size=2, seed=0)
    result = agent.observe_transition(
        CARTPOLE_LIKE_OBSERVATION, 1, -1.0, CARTPOLE_LIKE_OBSERVATION, False
    )
    assert result is None


def test_random_agent_end_episode_returns_none() -> None:
    agent = RandomAgent(action_space_size=2, seed=0)
    assert agent.end_episode() is None


def test_random_agent_diagnostics_is_empty_dict() -> None:
    agent = RandomAgent(action_space_size=2, seed=0)
    assert agent.diagnostics() == {}


def test_random_agent_hooks_do_not_change_action_sequence() -> None:
    plain_agent = RandomAgent(action_space_size=2, seed=0)
    hooked_agent = RandomAgent(action_space_size=2, seed=0)
    hooked_actions = []
    for _ in range(20):
        hooked_actions.append(hooked_agent.select_action(CARTPOLE_LIKE_OBSERVATION))
        hooked_agent.observe_transition(
            CARTPOLE_LIKE_OBSERVATION, hooked_actions[-1], -1.0, CARTPOLE_LIKE_OBSERVATION, False
        )
    hooked_agent.end_episode()
    assert hooked_actions == select_actions(plain_agent, 20)

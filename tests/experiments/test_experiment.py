import inspect

import gymnasium as gym
import pytest

import experiments.experiment as family_module
from agents.agent import RandomAgent
from experiments.experiment import Experiment

BASE_CLASS = Experiment
# Values for keyword-only parameters that child classes add, chosen by type annotation.
EXTRA_ARGUMENT_VALUES_BY_TYPE = {int: 0, float: 0.0, str: "", bool: False}


def base_init_arguments() -> dict:
    """Fresh arguments for the base class __init__ parameters (a new agent and env each call)."""
    return {
        "agent": RandomAgent(action_space_size=2, seed=0),
        "env": gym.make("CartPole-v1"),
        "num_episodes": 1,
        "seed": 0,
    }


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


def construction_arguments(cls: type, base_arguments: dict) -> dict:
    """Base arguments plus a typed value for each required added keyword-only parameter."""
    arguments = dict(base_arguments)
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

    # It instantiates, stores the base arguments and starts with empty results.
    base_arguments = base_init_arguments()
    instance = cls(**construction_arguments(cls, base_arguments))
    for name, value in base_arguments.items():
        assert getattr(instance, name) is value
    assert instance.results == {"episode_returns": [], "episode_lengths": []}

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


# EpisodicExperiment

import numpy as np

from experiments.experiment import EpisodicExperiment


def make_seeded_cartpole_experiment(num_episodes: int, seed: int) -> EpisodicExperiment:
    """An EpisodicExperiment on CartPole-v1 with a RandomAgent seeded by the same seed."""
    agent = RandomAgent(action_space_size=2, seed=seed)
    env = gym.make("CartPole-v1")
    return EpisodicExperiment(agent, env, num_episodes, seed)


def test_episodic_experiment_results_empty_before_run() -> None:
    experiment = make_seeded_cartpole_experiment(num_episodes=3, seed=0)
    assert experiment.results == {"episode_returns": [], "episode_lengths": []}


def test_episodic_experiment_records_one_value_per_episode() -> None:
    experiment = make_seeded_cartpole_experiment(num_episodes=3, seed=0)
    experiment.run()
    assert len(experiment.results["episode_returns"]) == 3
    assert len(experiment.results["episode_lengths"]) == 3


def test_episodic_experiment_return_equals_length_and_is_in_range() -> None:
    experiment = make_seeded_cartpole_experiment(num_episodes=3, seed=0)
    experiment.run()
    for episode_return, episode_length in zip(
        experiment.results["episode_returns"], experiment.results["episode_lengths"]
    ):
        assert episode_return == episode_length
        assert 8 <= episode_return <= 500


def test_episodic_experiment_same_seeds_give_identical_results() -> None:
    first_experiment = make_seeded_cartpole_experiment(num_episodes=3, seed=0)
    second_experiment = make_seeded_cartpole_experiment(num_episodes=3, seed=0)
    first_experiment.run()
    second_experiment.run()
    assert first_experiment.results == second_experiment.results


def test_episodic_experiment_run_mutates_env() -> None:
    experiment = make_seeded_cartpole_experiment(num_episodes=3, seed=0)
    experiment.env.reset(seed=12345)
    state_before_run = np.array(experiment.env.unwrapped.state, copy=True)
    experiment.run()
    state_after_run = np.array(experiment.env.unwrapped.state, copy=True)
    assert not np.array_equal(state_before_run, state_after_run)

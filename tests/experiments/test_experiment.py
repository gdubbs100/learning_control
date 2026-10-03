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


# LearningExperiment

from agents.agent import Agent
from experiments.experiment import LearningExperiment


class RecordingAgent(Agent):
    """An agent that always pushes left and records every hook call, for checking the experiment."""

    def __init__(self) -> None:
        super().__init__(action_space_size=2)
        self.chosen_from: list[np.ndarray] = []
        self.observed_from: list[np.ndarray] = []
        self.episodes_ended = 0

    def select_action(self, observation: np.ndarray) -> int:
        self.chosen_from.append(np.array(observation, copy=True))
        return 0

    def observe_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.observed_from.append(np.array(observation, copy=True))

    def end_episode(self) -> None:
        self.episodes_ended += 1

    def diagnostics(self) -> dict[str, float]:
        return {"episodes_ended": float(self.episodes_ended)}


def make_learning_experiment(agent: Agent, num_episodes: int, **options) -> LearningExperiment:
    """A LearningExperiment on CartPole-v1 with seed 0."""
    return LearningExperiment(agent, gym.make("CartPole-v1"), num_episodes, 0, **options)


def test_learning_experiment_starts_with_empty_results_and_diagnostics_log() -> None:
    experiment = make_learning_experiment(RecordingAgent(), num_episodes=3)
    assert experiment.results == {"episode_returns": [], "episode_lengths": []}
    assert experiment.diagnostics_log == []


def test_learning_experiment_matches_episodic_experiment_for_a_non_learning_agent() -> None:
    learning_experiment = make_learning_experiment(RandomAgent(action_space_size=2, seed=0), 3)
    episodic_experiment = make_seeded_cartpole_experiment(num_episodes=3, seed=0)
    learning_experiment.run()
    episodic_experiment.run()
    assert learning_experiment.results == episodic_experiment.results


def test_learning_experiment_calls_hooks_once_per_step_and_per_episode() -> None:
    agent = RecordingAgent()
    experiment = make_learning_experiment(agent, num_episodes=3)
    experiment.run()
    assert len(agent.observed_from) == int(sum(experiment.results["episode_lengths"]))
    assert len(agent.chosen_from) == len(agent.observed_from)
    assert agent.episodes_ended == 3


def test_learning_experiment_reports_the_observation_the_action_was_chosen_from() -> None:
    agent = RecordingAgent()
    make_learning_experiment(agent, num_episodes=2).run()
    for chosen, observed in zip(agent.chosen_from, agent.observed_from):
        np.testing.assert_array_equal(chosen, observed)


def test_learning_experiment_logs_agent_diagnostics_after_each_episode() -> None:
    experiment = make_learning_experiment(RecordingAgent(), num_episodes=3)
    experiment.run()
    assert experiment.diagnostics_log == [
        {"episodes_ended": 1.0},
        {"episodes_ended": 2.0},
        {"episodes_ended": 3.0},
    ]


def test_learning_experiment_diagnostics_log_entries_are_independent_of_the_agent() -> None:
    agent = RecordingAgent()
    experiment = make_learning_experiment(agent, num_episodes=2)
    experiment.run()
    experiment.diagnostics_log[0]["episodes_ended"] = -1.0
    assert agent.diagnostics() == {"episodes_ended": 2.0}


def test_learning_experiment_max_seconds_zero_stops_after_one_episode() -> None:
    experiment = make_learning_experiment(RecordingAgent(), num_episodes=10, max_seconds=0.0)
    experiment.run()
    assert len(experiment.results["episode_returns"]) == 1
    assert len(experiment.diagnostics_log) == 1


def test_learning_experiment_stops_early_when_returns_converge() -> None:
    experiment = make_learning_experiment(
        RecordingAgent(), num_episodes=10, convergence_window=2, convergence_tolerance=1e6
    )
    experiment.run()
    assert len(experiment.results["episode_returns"]) == 4


def test_learning_experiment_without_stopping_options_runs_all_episodes() -> None:
    experiment = make_learning_experiment(RecordingAgent(), num_episodes=5)
    experiment.run()
    assert len(experiment.results["episode_returns"]) == 5
    assert len(experiment.diagnostics_log) == 5

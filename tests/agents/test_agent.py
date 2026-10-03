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


# ModelBasedMPCAgent

from buffers.replay_buffer import ListReplayBuffer
from agents.agent import ModelBasedMPCAgent
from models.dynamics_model import LinearDynamicsModel
from planners.planner import RandomShootingPlanner

MPC_TRUE_A = 0.9 * np.eye(4)
MPC_TRUE_B = np.array([0.0, 0.1, 0.0, 0.2])


def mpc_state(index: int) -> np.ndarray:
    """A deterministic, varied 4-d state for the given index (no randomness)."""
    return 1.5 * np.array(
        [np.sin(index), np.cos(2 * index), np.sin(3 * index + 1), np.cos(5 * index)]
    )


def feed_exact_episode(
    agent: ModelBasedMPCAgent, a_matrix: np.ndarray, b_vector: np.ndarray, first_index: int, count: int
) -> None:
    """Give the agent `count` noise-free transitions s' = A s + B u (action 1 is u=+1, action 0 is
    u=-1, alternating by index), then end the episode."""
    for index in range(first_index, first_index + count):
        state = mpc_state(index)
        action = index % 2
        control_input = 2.0 * action - 1.0
        next_state = a_matrix @ state + b_vector * control_input
        agent.observe_transition(state, action, -1.0, next_state, False)
    agent.end_episode()


def make_mpc_agent(**overrides) -> ModelBasedMPCAgent:
    """A ModelBasedMPCAgent with a short horizon and few samples, retraining every episode by default."""
    settings = dict(retrain_every_k_episodes=1, horizon=1, num_samples=50, seed=0)
    settings.update(overrides)
    return ModelBasedMPCAgent(action_space_size=2, **settings)


def test_mpc_agent_builds_default_components() -> None:
    agent = make_mpc_agent(max_transitions=123)
    assert isinstance(agent.model, LinearDynamicsModel)
    assert isinstance(agent.planner, RandomShootingPlanner)
    assert isinstance(agent.buffer, ListReplayBuffer)
    assert agent.buffer.max_transitions == 123


def test_mpc_agent_uses_injected_components() -> None:
    buffer = ListReplayBuffer(max_transitions=10)
    model = LinearDynamicsModel(state_dim=4)
    agent = make_mpc_agent(buffer=buffer, model=model)
    assert agent.buffer is buffer
    assert agent.model is model


def test_mpc_agent_acts_randomly_but_validly_before_first_fit() -> None:
    agent = make_mpc_agent()
    actions = select_actions(agent, 50)
    assert set(actions) == {0, 1}


def test_mpc_agent_same_seed_gives_same_actions_before_fit() -> None:
    assert select_actions(make_mpc_agent(seed=3), 30) == select_actions(make_mpc_agent(seed=3), 30)


def test_mpc_agent_buffer_grows_with_each_observed_transition() -> None:
    agent = make_mpc_agent()
    for step in range(3):
        agent.observe_transition(mpc_state(step), 1, -1.0, mpc_state(step + 1), step == 2)
        assert agent.buffer.size() == step + 1
    np.testing.assert_array_equal(agent.buffer.as_arrays()["terminated"], [False, False, True])


def test_mpc_agent_model_fits_only_every_k_episodes() -> None:
    agent = make_mpc_agent(retrain_every_k_episodes=2)
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=15)
    assert agent.model.is_fitted() is False
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=15, count=15)
    assert agent.model.is_fitted() is True


def test_mpc_agent_diagnostics_empty_before_any_episode_ends() -> None:
    assert make_mpc_agent().diagnostics() == {}


def test_mpc_agent_diagnostics_without_fit_only_report_buffer_size() -> None:
    agent = make_mpc_agent(retrain_every_k_episodes=2)
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=15)
    assert agent.diagnostics() == {"buffer_size": 15.0}


def test_mpc_agent_first_fit_diagnostics_have_train_mse_but_no_one_step_mse() -> None:
    agent = make_mpc_agent()
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=30)
    diagnostics = agent.diagnostics()
    assert set(diagnostics) == {"buffer_size", "model_train_mse"}
    assert diagnostics["buffer_size"] == 30.0
    assert diagnostics["model_train_mse"] < 1e-12


def test_mpc_agent_one_step_mse_is_near_zero_when_dynamics_are_unchanged() -> None:
    agent = make_mpc_agent()
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=30)
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=30, count=30)
    assert agent.diagnostics()["model_one_step_mse_new_data"] < 1e-12


def test_mpc_agent_one_step_mse_is_large_when_dynamics_change() -> None:
    agent = make_mpc_agent()
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=30)
    feed_exact_episode(agent, 0.5 * np.eye(4), np.array([1.0, 0.0, 0.0, 0.0]), first_index=30, count=30)
    assert agent.diagnostics()["model_one_step_mse_new_data"] > 1e-3


def test_mpc_agent_planned_action_reduces_cart_velocity_with_fitted_model() -> None:
    # With s' = 0.9 s + [0, 0.1, 0, 0.2] u and horizon 1, from [0, 1, 0, 0] the next-step cost is
    # 0.64 + 0.04 = 0.68 for action 0 (u = -1) and 1.0 + 0.04 = 1.04 for action 1 (u = +1).
    agent = make_mpc_agent()
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=30)
    assert agent.select_action(np.array([0.0, 1.0, 0.0, 0.0])) == 0


def test_mpc_agent_same_seed_and_data_give_same_planned_action() -> None:
    agents = [make_mpc_agent(seed=5, horizon=4) for _ in range(2)]
    for agent in agents:
        feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=30)
    observation = np.array([0.1, 0.2, 0.3, 0.4])
    assert agents[0].select_action(observation) == agents[1].select_action(observation)


def test_mpc_agent_does_not_mutate_observation() -> None:
    agent = make_mpc_agent()
    feed_exact_episode(agent, MPC_TRUE_A, MPC_TRUE_B, first_index=0, count=30)
    observation = np.array([0.1, 0.2, 0.3, 0.4])
    original_observation = observation.copy()
    agent.select_action(observation)
    np.testing.assert_array_equal(observation, original_observation)


# ReinforceAgent

import math

import torch

from agents.agent import ReinforceAgent

REINFORCE_DIAGNOSTIC_KEYS = {"loss", "grad_norm", "entropy", "episode_return"}
EPISODE_REWARDS = [-1.0, -2.0, -3.0]


def policy_parameters(agent: ReinforceAgent) -> list[torch.Tensor]:
    """Copies of all the policy network's parameters."""
    return [parameter.detach().clone() for parameter in agent.policy.parameters()]


def play_scripted_episode(agent: ReinforceAgent, rewards: list[float]) -> None:
    """Choose an action for each reward, report the transition, then end the episode."""
    for reward in rewards:
        action = agent.select_action(CARTPOLE_LIKE_OBSERVATION)
        agent.observe_transition(
            CARTPOLE_LIKE_OBSERVATION, action, reward, CARTPOLE_LIKE_OBSERVATION, False
        )
    agent.end_episode()


def test_reinforce_agent_actions_are_in_action_space_and_both_appear() -> None:
    actions = select_actions(ReinforceAgent(action_space_size=2, seed=0), 100)
    assert set(actions) == {0, 1}


def test_reinforce_agent_same_seed_gives_identical_sequences() -> None:
    first_actions = select_actions(ReinforceAgent(action_space_size=2, seed=0), 50)
    second_actions = select_actions(ReinforceAgent(action_space_size=2, seed=0), 50)
    assert first_actions == second_actions


def test_reinforce_agent_different_seeds_give_different_sequences() -> None:
    first_actions = select_actions(ReinforceAgent(action_space_size=2, seed=0), 50)
    second_actions = select_actions(ReinforceAgent(action_space_size=2, seed=1), 50)
    assert first_actions != second_actions


def test_reinforce_agent_diagnostics_empty_before_any_episode_ends() -> None:
    assert ReinforceAgent(action_space_size=2, seed=0).diagnostics() == {}


def test_reinforce_agent_end_episode_changes_policy_parameters() -> None:
    agent = ReinforceAgent(action_space_size=2, seed=0)
    parameters_before = policy_parameters(agent)
    play_scripted_episode(agent, EPISODE_REWARDS)
    parameters_after = policy_parameters(agent)
    assert any(
        not torch.equal(before, after) for before, after in zip(parameters_before, parameters_after)
    )


def test_reinforce_agent_end_episode_without_transitions_changes_nothing() -> None:
    agent = ReinforceAgent(action_space_size=2, seed=0)
    parameters_before = policy_parameters(agent)
    agent.end_episode()
    for before, after in zip(parameters_before, policy_parameters(agent)):
        assert torch.equal(before, after)
    assert agent.diagnostics() == {}


def test_reinforce_agent_diagnostics_after_an_episode() -> None:
    agent = ReinforceAgent(action_space_size=2, seed=0)
    play_scripted_episode(agent, EPISODE_REWARDS)
    diagnostics = agent.diagnostics()
    assert REINFORCE_DIAGNOSTIC_KEYS <= set(diagnostics)
    assert diagnostics["episode_return"] == -6.0
    assert 0.0 < diagnostics["entropy"] <= math.log(2) + 1e-6
    assert diagnostics["grad_norm"] > 0.0
    assert math.isfinite(diagnostics["loss"])


def test_reinforce_agent_same_seed_and_data_give_identical_updates() -> None:
    first_agent = ReinforceAgent(action_space_size=2, seed=0)
    second_agent = ReinforceAgent(action_space_size=2, seed=0)
    play_scripted_episode(first_agent, EPISODE_REWARDS)
    play_scripted_episode(second_agent, EPISODE_REWARDS)
    for first, second in zip(policy_parameters(first_agent), policy_parameters(second_agent)):
        assert torch.equal(first, second)


def test_reinforce_agent_learns_to_prefer_the_rewarded_action_on_a_toy_problem() -> None:
    # Constant observation; action 1 earns +1 and action 0 earns -1, over 10-step episodes.
    agent = ReinforceAgent(action_space_size=2, seed=0)
    for _ in range(300):
        for _ in range(10):
            action = agent.select_action(CARTPOLE_LIKE_OBSERVATION)
            reward = 1.0 if action == 1 else -1.0
            agent.observe_transition(
                CARTPOLE_LIKE_OBSERVATION, action, reward, CARTPOLE_LIKE_OBSERVATION, False
            )
        agent.end_episode()
    action_probabilities = torch.softmax(
        agent.policy(torch.as_tensor(CARTPOLE_LIKE_OBSERVATION)), dim=-1
    )
    assert float(action_probabilities[1]) > 0.9


def test_reinforce_agent_does_not_mutate_observation() -> None:
    agent = ReinforceAgent(action_space_size=2, seed=0)
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    original_observation = observation.copy()
    agent.select_action(observation)
    agent.observe_transition(observation, 1, -1.0, observation, False)
    agent.end_episode()
    np.testing.assert_array_equal(observation, original_observation)

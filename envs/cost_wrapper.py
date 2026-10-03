from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from utils.control.action_to_input import action_to_input
from utils.control.quadratic_cost import quadratic_cost


class CostWrapper(gym.Wrapper, ABC):
    """Base class for environment wrappers that replace the environment's reward.

    A wrapper in this family wraps one environment and, on each step, reports a
    reward computed by `compute_reward` from the resulting observation and the
    action taken, instead of the wrapped environment's own reward. Everything else
    (observations, termination, truncation) is left as the wrapped environment
    produces it.
    """

    @abstractmethod
    def compute_reward(self, observation: np.ndarray, action: int) -> float:
        """Compute the reward for one step.

        Args:
            observation: the observation after the step.
            action: the action taken in the step.

        Returns:
            The reward the wrapper reports for the step.
        """


class QuadraticCostCartPole(CostWrapper):
    """A CartPole wrapper whose reward is the negative quadratic cost -((s - s*)^2 + u^2).

    s is the observation after the step (the CartPole state), s* is `target_state`,
    and u is the control input of the action (action 0 is -1, action 1 is +1). The
    squared state error is summed over the state dimensions. Termination and
    truncation are the wrapped environment's own.

    Attributes:
        target_state: the state s* the cost measures distance from, shape (4,).
    """

    target_state: np.ndarray

    def __init__(self, env: gym.Env, *, target_state: np.ndarray | None = None) -> None:
        """Wrap a CartPole environment and store the target state.

        Args:
            env: the CartPole environment to wrap.
            target_state: the state s*, shape (4,). Defaults to zeros.

        Returns:
            None.
        """
        super().__init__(env)
        if target_state is None:
            self.target_state = np.zeros(4)
        else:
            self.target_state = np.array(target_state, copy=True)

    def compute_reward(self, observation: np.ndarray, action: int) -> float:
        """Compute the negative quadratic cost of the observation and action.

        Differs from the base class by implementing the reward as -((s - s*)^2 + u^2).

        Args:
            observation: the CartPole observation after the step, shape (4,).
            action: the action taken, 0 or 1.

        Returns:
            The reward, -(sum((observation - target_state)^2) + u^2) with u = -1 or +1.
        """
        control_input = action_to_input(np.array(action))
        cost = quadratic_cost(observation.astype(np.float64), control_input, self.target_state)
        return -float(cost)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the wrapped environment and replace its reward with the negative cost.

        Differs from the wrapped environment's step only in the reward returned.

        Args:
            action: the action to take, 0 or 1.

        Returns:
            The usual (observation, reward, terminated, truncated, info) tuple, where
            reward is `compute_reward(observation, action)`.
        """
        observation, _, terminated, truncated, info = self.env.step(action)
        reward = self.compute_reward(observation, action)
        return observation, reward, terminated, truncated, info

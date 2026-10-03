from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np


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

from abc import ABC, abstractmethod

import numpy as np


class ReplayBuffer(ABC):
    """Base class for stores of past experience that agents learn from later.

    A buffer holds transitions, one per environment step, each made of an
    observation, the action taken, the reward received, the next observation and
    whether the step terminated the episode.
    """

    @abstractmethod
    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition.

        Args:
            observation: the observation the action was chosen from.
            action: the action taken.
            reward: the reward received for the step.
            next_observation: the observation after the step.
            terminated: whether the step ended the episode by termination.

        Returns:
            None.
        """

    @abstractmethod
    def size(self) -> int:
        """Count the stored transitions.

        Args:
            None.

        Returns:
            The number of transitions stored so far.
        """

    @abstractmethod
    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return all stored transitions as arrays, oldest first.

        Args:
            None.

        Returns:
            A dict with keys "observations" and "next_observations" (shape
            (size, observation_dim)), "actions", "rewards" and "terminated"
            (shape (size,)).
        """

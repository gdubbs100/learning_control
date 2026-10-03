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


class ListReplayBuffer(ReplayBuffer):
    """A replay buffer that keeps every transition in Python lists, with no size limit.

    Arrays passed to `add` are copied, so later changes to them do not alter the buffer.

    Attributes:
        observations: the stored observations, oldest first.
        actions: the stored actions.
        rewards: the stored rewards.
        next_observations: the stored next observations.
        terminated: the stored termination flags.
    """

    observations: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    next_observations: list[np.ndarray]
    terminated: list[bool]

    def __init__(self) -> None:
        """Start with an empty buffer.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.terminated = []

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Append one transition, copying the observation arrays.

        Differs from the base class by actually storing the transition.

        Args:
            observation: the observation the action was chosen from.
            action: the action taken.
            reward: the reward received for the step.
            next_observation: the observation after the step.
            terminated: whether the step ended the episode by termination.

        Returns:
            None.
        """
        self.observations.append(np.array(observation, copy=True))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_observations.append(np.array(next_observation, copy=True))
        self.terminated.append(bool(terminated))

    def size(self) -> int:
        """Count the stored transitions.

        Differs from the base class by counting the stored list entries.

        Args:
            None.

        Returns:
            The number of transitions stored so far.
        """
        return len(self.actions)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return all stored transitions as arrays, oldest first.

        Differs from the base class by stacking the stored lists into arrays.

        Args:
            None.

        Returns:
            A dict with keys "observations" and "next_observations" (shape
            (size, observation_dim)), "actions", "rewards" and "terminated"
            (shape (size,), with bool dtype for "terminated").
        """
        return {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "next_observations": np.array(self.next_observations),
            "terminated": np.array(self.terminated, dtype=np.bool_),
        }

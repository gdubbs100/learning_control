from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class ReplayBuffer(ABC):
    """Base class for stores of past experience that agents learn from later.

    A buffer holds transitions, one per environment step, each made of an
    observation, the action taken, the reward received, the next observation and
    whether the step terminated the episode. A buffer holds at most
    `max_transitions` transitions; once full, adding another drops the oldest.

    Attributes:
        max_transitions: the largest number of transitions the buffer holds.
    """

    max_transitions: int

    def __init__(self, max_transitions: int) -> None:
        """Store the buffer's capacity.

        Args:
            max_transitions: the largest number of transitions to hold. Must be at least 1.

        Returns:
            None. Raises ValueError if `max_transitions` is below 1.
        """
        if max_transitions < 1:
            raise ValueError(f"max_transitions must be at least 1, got {max_transitions}")
        self.max_transitions = max_transitions

    @abstractmethod
    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition, dropping the oldest one if the buffer is full.

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
    """A replay buffer that keeps its transitions in bounded deques, dropping the oldest when full.

    Arrays passed to `add` are copied, so later changes to them do not alter the buffer.

    Attributes:
        observations: the stored observations, oldest first.
        actions: the stored actions.
        rewards: the stored rewards.
        next_observations: the stored next observations.
        terminated: the stored termination flags.
    """

    observations: deque[np.ndarray]
    actions: deque[int]
    rewards: deque[float]
    next_observations: deque[np.ndarray]
    terminated: deque[bool]

    def __init__(self, max_transitions: int) -> None:
        """Create an empty buffer that holds at most `max_transitions` transitions.

        Args:
            max_transitions: the largest number of transitions to hold. Must be at least 1.

        Returns:
            None. Raises ValueError if `max_transitions` is below 1.
        """
        super().__init__(max_transitions)
        self.observations = deque(maxlen=max_transitions)
        self.actions = deque(maxlen=max_transitions)
        self.rewards = deque(maxlen=max_transitions)
        self.next_observations = deque(maxlen=max_transitions)
        self.terminated = deque(maxlen=max_transitions)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Append one transition, copying the observation arrays, and drop the oldest if full.

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

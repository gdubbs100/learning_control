from abc import ABC, abstractmethod

import numpy as np
import torch


class Agent(ABC):
    """Base class for agents that choose discrete actions from environment observations.

    Every agent acts in an environment with a discrete action space of
    `action_space_size` actions, numbered 0 to `action_space_size - 1`.

    Attributes:
        action_space_size: the number of discrete actions the agent can choose from.
    """

    action_space_size: int

    def __init__(self, action_space_size: int) -> None:
        """Store the size of the discrete action space.

        Args:
            action_space_size: the number of discrete actions available.

        Returns:
            None.
        """
        self.action_space_size = action_space_size

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """Choose an action given the current observation.

        Args:
            observation: the current observation from the environment.

        Returns:
            The chosen action, an integer in [0, action_space_size).
        """


class RandomAgent(Agent):
    """An agent that ignores observations and picks actions uniformly at random.

    Randomness comes from a torch.Generator seeded at construction, so two agents
    built with the same seed produce the same sequence of actions.

    Attributes:
        seed: the seed used for the agent's random number generator.
        random_generator: the seeded torch generator actions are drawn from.
    """

    seed: int
    random_generator: torch.Generator

    def __init__(self, action_space_size: int, *, seed: int) -> None:
        """Store the action space size and create a seeded random generator.

        Args:
            action_space_size: the number of discrete actions available.
            seed: the seed for the agent's random number generator.

        Returns:
            None.
        """
        super().__init__(action_space_size)
        self.seed = seed
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """Choose an action uniformly at random, ignoring the observation.

        Differs from the base class by not using the observation at all.

        Args:
            observation: the current observation from the environment (ignored).

        Returns:
            A uniformly random action, an integer in [0, action_space_size).
        """
        random_action = torch.randint(
            low=0, high=self.action_space_size, size=(1,), generator=self.random_generator
        )
        return int(random_action.item())

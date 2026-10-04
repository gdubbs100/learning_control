from abc import ABC, abstractmethod

import numpy as np

from models.dynamics_model import DynamicsModel
from utils.control.action_to_input import action_to_input
from utils.control.quadratic_cost import quadratic_cost


class Planner(ABC):
    """Base class for model predictive control optimisers that choose action sequences.

    Given a dynamics model and the current state, a planner searches for the
    sequence of discrete actions over `horizon` steps with the lowest predicted
    cost, where each step costs (s - s*)^2 + u^2 for the control input u of the action.

    Attributes:
        horizon: the number of steps in a planned action sequence.
        action_space_size: the number of discrete actions, numbered 0 to action_space_size - 1.
        target_state: the state s* the cost measures distance from, shape (state_dim,).
    """

    horizon: int
    action_space_size: int
    target_state: np.ndarray

    def __init__(self, horizon: int, action_space_size: int, target_state: np.ndarray) -> None:
        """Store the planning settings.

        Args:
            horizon: the number of steps in a planned action sequence.
            action_space_size: the number of discrete actions available.
            target_state: the state s* the cost measures distance from, shape (state_dim,).

        Returns:
            None.
        """
        self.horizon = horizon
        self.action_space_size = action_space_size
        self.target_state = np.array(target_state, copy=True)

    @abstractmethod
    def plan(self, model: DynamicsModel, state: np.ndarray) -> np.ndarray:
        """Choose an action sequence by predicting its cost with the model.

        Args:
            model: a fitted dynamics model used to predict future states.
            state: the current state, shape (state_dim,).

        Returns:
            The planned actions, an integer array of shape (horizon,) with values in
            [0, action_space_size). The first entry is the action to take now.
        """

    def hyperparameters(self) -> dict[str, str | int | float | bool | None | list[float]]:
        """Report the settings of the planner that affect performance, as JSON-safe values.

        Subclasses with further settings extend the returned dict.

        Args:
            None.

        Returns:
            A dict with "type" (the class name), "horizon", "action_space_size" and
            "target_state" (as a list of floats).
        """
        return {
            "type": type(self).__name__,
            "horizon": self.horizon,
            "action_space_size": self.action_space_size,
            "target_state": [float(value) for value in self.target_state],
        }


class RandomShootingPlanner(Planner):
    """A planner that samples random action sequences and keeps the cheapest.

    Each plan call draws `num_samples` uniformly random action sequences of length
    `horizon`, rolls them all through the model in one batch, adds up the cost
    (s - s*)^2 + u^2 of each step, and returns the sequence with the lowest total.

    Attributes:
        num_samples: the number of random action sequences tried per plan.
        seed: the seed of the planner's random number generator.
        random_generator: the seeded generator the sequences are drawn from.
    """

    num_samples: int
    seed: int
    random_generator: np.random.Generator

    def __init__(
        self,
        horizon: int,
        action_space_size: int,
        target_state: np.ndarray,
        *,
        num_samples: int = 1000,
        seed: int = 0,
    ) -> None:
        """Store the planning settings and create a seeded random generator.

        Args:
            horizon: the number of steps in a planned action sequence.
            action_space_size: the number of discrete actions available.
            target_state: the state s* the cost measures distance from, shape (state_dim,).
            num_samples: the number of random action sequences tried per plan.
            seed: the seed for the planner's random number generator.

        Returns:
            None.
        """
        super().__init__(horizon, action_space_size, target_state)
        self.num_samples = num_samples
        self.seed = seed
        self.random_generator = np.random.default_rng(seed)

    def plan(self, model: DynamicsModel, state: np.ndarray) -> np.ndarray:
        """Choose the cheapest of `num_samples` random action sequences.

        Differs from the base class by implementing the search as random shooting.
        Each call advances the planner's random generator.

        Args:
            model: a fitted dynamics model used to predict future states.
            state: the current state, shape (state_dim,).

        Returns:
            The lowest-cost sampled actions, an integer array of shape (horizon,)
            with values in [0, action_space_size).
        """
        action_sequences = self.random_generator.integers(
            0, self.action_space_size, size=(self.num_samples, self.horizon)
        )
        predicted_states = np.tile(state, (self.num_samples, 1))
        total_costs = np.zeros(self.num_samples)
        for step_index in range(self.horizon):
            control_inputs = action_to_input(action_sequences[:, step_index])
            predicted_states = model.predict(predicted_states, control_inputs)
            total_costs += quadratic_cost(predicted_states, control_inputs, self.target_state)
        return action_sequences[int(np.argmin(total_costs))].copy()

    def hyperparameters(self) -> dict[str, str | int | float | bool | None | list[float]]:
        """Report the base planner settings plus the number of samples and the seed.

        Differs from the base class by adding "num_samples" and "seed".

        Args:
            None.

        Returns:
            A dict with the base class entries and "num_samples" and "seed".
        """
        return {**super().hyperparameters(), "num_samples": self.num_samples, "seed": self.seed}

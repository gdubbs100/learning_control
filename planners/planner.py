from abc import ABC, abstractmethod

import numpy as np

from models.dynamics_model import DynamicsModel


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

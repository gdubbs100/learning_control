from abc import ABC, abstractmethod

import numpy as np


class DynamicsModel(ABC):
    """Base class for learned models of how a system's state changes under a control input.

    A model predicts the next state from the current state and a scalar control
    input. It is trained on observed transitions with `fit`, and can be used for
    planning once `is_fitted` is True.

    Attributes:
        state_dim: the number of dimensions of the state.
    """

    state_dim: int

    def __init__(self, state_dim: int) -> None:
        """Store the size of the state.

        Args:
            state_dim: the number of dimensions of the state.

        Returns:
            None.
        """
        self.state_dim = state_dim

    @abstractmethod
    def fit(self, states: np.ndarray, inputs: np.ndarray, next_states: np.ndarray) -> dict[str, float]:
        """Train the model from scratch on observed transitions, replacing any earlier fit.

        Args:
            states: the states before each step, shape (num_transitions, state_dim).
            inputs: the scalar control input applied at each step, shape (num_transitions,).
            next_states: the states after each step, shape (num_transitions, state_dim).

        Returns:
            Training metrics as a dict of name to value. Always includes "train_mse",
            the mean squared one-step prediction error on the training data.
        """

    @abstractmethod
    def predict(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Predict the next states for a batch of states and control inputs.

        Args:
            states: the current states, shape (..., state_dim).
            inputs: the scalar control inputs, shape (...).

        Returns:
            The predicted next states, shape (..., state_dim). Raises RuntimeError
            if the model has not been fitted.
        """

    @abstractmethod
    def is_fitted(self) -> bool:
        """Say whether the model has been trained and can make predictions.

        Args:
            None.

        Returns:
            True once `fit` has been called, else False.
        """

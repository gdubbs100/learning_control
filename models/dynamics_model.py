from abc import ABC, abstractmethod

import numpy as np
import torch


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


class LinearDynamicsModel(DynamicsModel):
    """A linear model of the dynamics, s' = A s + B u, fitted by least squares.

    Fitting solves the least-squares problem in closed form with torch in float64,
    and replaces any earlier fit. `A` and `B` are None until the model is fitted.

    Attributes:
        A: the state matrix, shape (state_dim, state_dim), or None before fitting.
        B: the input vector, shape (state_dim,), or None before fitting.
    """

    A: torch.Tensor | None
    B: torch.Tensor | None

    def __init__(self, state_dim: int) -> None:
        """Create an unfitted model.

        Args:
            state_dim: the number of dimensions of the state.

        Returns:
            None.
        """
        super().__init__(state_dim)
        self.A = None
        self.B = None

    def fit(self, states: np.ndarray, inputs: np.ndarray, next_states: np.ndarray) -> dict[str, float]:
        """Fit A and B by least squares on the transitions, replacing any earlier fit.

        Differs from the base class by solving s' = A s + B u in closed form with
        `torch.linalg.lstsq`.

        Args:
            states: the states before each step, shape (num_transitions, state_dim).
            inputs: the scalar control input applied at each step, shape (num_transitions,).
            next_states: the states after each step, shape (num_transitions, state_dim).

        Returns:
            A dict with "train_mse", the mean squared one-step prediction error on the
            training data.
        """
        states_tensor = torch.as_tensor(states, dtype=torch.float64)
        inputs_tensor = torch.as_tensor(inputs, dtype=torch.float64)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float64)
        regressors = torch.cat([states_tensor, inputs_tensor[:, None]], dim=1)
        coefficients = torch.linalg.lstsq(regressors, next_states_tensor).solution
        self.A = coefficients[: self.state_dim].T.contiguous()
        self.B = coefficients[self.state_dim].contiguous()
        residuals = regressors @ coefficients - next_states_tensor
        return {"train_mse": float(torch.mean(residuals**2))}

    def predict(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Predict next states as A s + B u.

        Differs from the base class by using the fitted matrices.

        Args:
            states: the current states, shape (..., state_dim).
            inputs: the scalar control inputs, shape (...).

        Returns:
            The predicted next states as a float64 numpy array, shape (..., state_dim).
            Raises RuntimeError if the model has not been fitted.
        """
        if self.A is None or self.B is None:
            raise RuntimeError("LinearDynamicsModel must be fitted before predict is called")
        states_tensor = torch.as_tensor(states, dtype=torch.float64)
        inputs_tensor = torch.as_tensor(inputs, dtype=torch.float64)
        next_states = states_tensor @ self.A.T + inputs_tensor[..., None] * self.B
        return next_states.numpy()

    def parameters(self) -> dict[str, float]:
        """Report the entries of A and B as a flat dict.

        Differs from the base class by naming the fitted matrix entries.

        Args:
            None.

        Returns:
            A dict with "A_i_j" for each entry of A (row i, column j) followed by "B_i"
            for each entry of B, all as floats. Raises RuntimeError if the model has not
            been fitted.
        """
        if self.A is None or self.B is None:
            raise RuntimeError("LinearDynamicsModel must be fitted before parameters is called")
        parameters = {
            f"A_{row}_{column}": float(self.A[row, column])
            for row in range(self.state_dim)
            for column in range(self.state_dim)
        }
        parameters.update({f"B_{row}": float(self.B[row]) for row in range(self.state_dim)})
        return parameters

    def evaluate(self, states: np.ndarray, inputs: np.ndarray, next_states: np.ndarray) -> dict[str, float]:
        """Score the fitted model's one-step predictions on the given transitions.

        Differs from the base class by computing the mean squared error from `predict`.

        Args:
            states: the states before each step, shape (num_transitions, state_dim).
            inputs: the scalar control input applied at each step, shape (num_transitions,).
            next_states: the states after each step, shape (num_transitions, state_dim).

        Returns:
            A dict with "mse", the mean squared one-step prediction error over all
            transitions and state dimensions. Raises RuntimeError if the model has not
            been fitted.
        """
        predictions = self.predict(states, inputs)
        errors = predictions - np.asarray(next_states, dtype=np.float64)
        return {"mse": float(np.mean(errors**2))}

    def is_fitted(self) -> bool:
        """Say whether `fit` has been called.

        Differs from the base class by checking whether A and B have been set.

        Args:
            None.

        Returns:
            True once `fit` has been called, else False.
        """
        return self.A is not None and self.B is not None

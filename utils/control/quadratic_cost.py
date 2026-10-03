import numpy as np


def quadratic_cost(states: np.ndarray, inputs: np.ndarray, target_state: np.ndarray) -> np.ndarray:
    """Compute the quadratic cost (s - s*)^2 + u^2, summing the squared state error over dimensions.

    Args:
        states: states with the state dimension last, shape (..., state_dim).
        inputs: scalar control inputs, one per state, shape (...).
        target_state: the target state s*, shape (state_dim,).

    Returns:
        The cost for each state and input pair, shape (...): the sum over state
        dimensions of (s - s*)^2, plus u^2.
    """
    state_error = states - target_state
    return np.sum(state_error**2, axis=-1) + inputs**2

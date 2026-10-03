import numpy as np


def action_to_input(actions: np.ndarray) -> np.ndarray:
    """Convert discrete CartPole actions into signed control inputs.

    Action 0 (push left) maps to -1.0 and action 1 (push right) maps to +1.0.

    Args:
        actions: integer actions, each 0 or 1, in an array of any shape.

    Returns:
        A float64 array of the same shape with -1.0 where the action is 0 and
        +1.0 where the action is 1.
    """
    return 2.0 * actions.astype(np.float64) - 1.0

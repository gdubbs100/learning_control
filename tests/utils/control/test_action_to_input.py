import numpy as np

from utils.control.action_to_input import action_to_input


def test_action_to_input_maps_zero_to_minus_one_and_one_to_plus_one() -> None:
    inputs = action_to_input(np.array([0, 1, 1, 0]))
    np.testing.assert_array_equal(inputs, np.array([-1.0, 1.0, 1.0, -1.0]))
    assert inputs.dtype == np.float64


def test_action_to_input_preserves_shape() -> None:
    inputs = action_to_input(np.array([[0, 1, 0], [1, 1, 0]]))
    assert inputs.shape == (2, 3)
    np.testing.assert_array_equal(inputs, np.array([[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]))


def test_action_to_input_is_deterministic() -> None:
    actions = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(action_to_input(actions), action_to_input(actions))


def test_action_to_input_does_not_mutate_input() -> None:
    actions = np.array([0, 1, 1, 0])
    original_actions = actions.copy()
    action_to_input(actions)
    np.testing.assert_array_equal(actions, original_actions)

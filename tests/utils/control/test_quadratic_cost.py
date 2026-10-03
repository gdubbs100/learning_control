import numpy as np

from utils.control.quadratic_cost import quadratic_cost

ZERO_TARGET = np.zeros(4)


def test_quadratic_cost_single_state_with_zero_target() -> None:
    cost = quadratic_cost(np.array([1.0, 2.0, 0.0, 0.0]), np.array(1.0), ZERO_TARGET)
    assert float(cost) == 6.0


def test_quadratic_cost_state_equal_to_target_leaves_only_input_penalty() -> None:
    state = np.array([0.5, -0.5, 0.1, 0.2])
    cost = quadratic_cost(state, np.array(-1.0), state)
    assert float(cost) == 1.0


def test_quadratic_cost_batch_returns_one_cost_per_row() -> None:
    states = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    inputs = np.array([1.0, -1.0])
    costs = quadratic_cost(states, inputs, ZERO_TARGET)
    assert costs.shape == (2,)
    np.testing.assert_allclose(costs, np.array([6.0, 1.0]), atol=1e-12)


def test_quadratic_cost_uses_non_zero_target() -> None:
    target = np.array([1.0, 1.0, 1.0, 1.0])
    cost = quadratic_cost(np.array([2.0, 1.0, 1.0, -1.0]), np.array(1.0), target)
    assert float(cost) == 1.0 + 0.0 + 0.0 + 4.0 + 1.0


def test_quadratic_cost_is_deterministic() -> None:
    states = np.array([[1.0, 2.0, 0.0, 0.0]])
    inputs = np.array([1.0])
    np.testing.assert_array_equal(
        quadratic_cost(states, inputs, ZERO_TARGET), quadratic_cost(states, inputs, ZERO_TARGET)
    )


def test_quadratic_cost_does_not_mutate_inputs() -> None:
    states = np.array([[1.0, 2.0, 0.0, 0.0]])
    inputs = np.array([1.0])
    target = np.array([0.5, 0.0, 0.0, 0.0])
    original = (states.copy(), inputs.copy(), target.copy())
    quadratic_cost(states, inputs, target)
    np.testing.assert_array_equal(states, original[0])
    np.testing.assert_array_equal(inputs, original[1])
    np.testing.assert_array_equal(target, original[2])

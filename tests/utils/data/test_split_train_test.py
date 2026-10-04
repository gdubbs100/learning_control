import copy

import numpy as np
import pytest

from utils.data.split_train_test import split_train_test


def make_arrays(num_rows: int) -> dict[str, np.ndarray]:
    return {
        "states": np.arange(num_rows * 2, dtype=np.float64).reshape(num_rows, 2),
        "actions": np.arange(num_rows),
    }


def test_split_sizes_follow_test_fraction() -> None:
    train, test = split_train_test(make_arrays(10), test_fraction=0.2)
    assert train["actions"].shape == (8,)
    assert test["actions"].shape == (2,)
    assert train["states"].shape == (8, 2)
    assert test["states"].shape == (2, 2)


def test_first_rows_go_to_test_and_the_rest_to_train() -> None:
    train, test = split_train_test(make_arrays(10), test_fraction=0.2)
    np.testing.assert_array_equal(test["actions"], [0, 1])
    np.testing.assert_array_equal(train["actions"], [2, 3, 4, 5, 6, 7, 8, 9])


def test_rows_stay_aligned_across_keys() -> None:
    train, test = split_train_test(make_arrays(10), test_fraction=0.3)
    for part in (train, test):
        np.testing.assert_array_equal(part["states"][:, 0], part["actions"] * 2)


def test_keys_are_preserved() -> None:
    train, test = split_train_test(make_arrays(4), test_fraction=0.5)
    assert set(train) == {"states", "actions"}
    assert set(test) == {"states", "actions"}


def test_test_size_is_at_least_one_when_fraction_rounds_to_zero() -> None:
    train, test = split_train_test(make_arrays(5), test_fraction=0.01)
    assert test["actions"].shape == (1,)
    assert train["actions"].shape == (4,)


def test_train_size_is_at_least_one_when_fraction_rounds_to_all() -> None:
    train, test = split_train_test(make_arrays(5), test_fraction=0.99)
    assert train["actions"].shape == (1,)
    assert test["actions"].shape == (4,)


def test_two_rows_split_one_and_one() -> None:
    train, test = split_train_test(make_arrays(2), test_fraction=0.2)
    assert train["actions"].shape == (1,)
    assert test["actions"].shape == (1,)


def test_fewer_than_two_rows_raises() -> None:
    with pytest.raises(ValueError):
        split_train_test(make_arrays(1), test_fraction=0.2)


@pytest.mark.parametrize("test_fraction", [0.0, 1.0, -0.1, 1.5])
def test_test_fraction_outside_open_unit_interval_raises(test_fraction: float) -> None:
    with pytest.raises(ValueError):
        split_train_test(make_arrays(10), test_fraction=test_fraction)


def test_arrays_of_different_lengths_raise() -> None:
    arrays = {"states": np.zeros((5, 2)), "actions": np.zeros(4)}
    with pytest.raises(ValueError):
        split_train_test(arrays, test_fraction=0.2)


def test_split_is_deterministic() -> None:
    arrays = make_arrays(10)
    first_train, first_test = split_train_test(arrays, 0.2)
    second_train, second_test = split_train_test(arrays, 0.2)
    for key in arrays:
        np.testing.assert_array_equal(first_train[key], second_train[key])
        np.testing.assert_array_equal(first_test[key], second_test[key])


def test_split_does_not_mutate_input() -> None:
    arrays = make_arrays(10)
    original_arrays = copy.deepcopy(arrays)
    split_train_test(arrays, 0.2)
    assert set(arrays) == set(original_arrays)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], original_arrays[key])

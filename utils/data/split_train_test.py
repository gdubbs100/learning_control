import numpy as np


def split_train_test(
    arrays: dict[str, np.ndarray], test_fraction: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split a dict of equal-length arrays into a train part and a test part by rows.

    The first `round(num_rows * test_fraction)` rows go to the test part and the rest
    to the train part, with at least one row in each. The rows are not shuffled, so
    shuffle first if a random split is wanted.

    Args:
        arrays: arrays that all have the same number of rows (first dimension), at least 2.
        test_fraction: the fraction of rows for the test part, strictly between 0 and 1.

    Returns:
        A tuple (train, test) of dicts with the same keys as `arrays`, whose rows stay
        aligned across keys. Raises ValueError if `test_fraction` is not in (0, 1), if the
        arrays have different numbers of rows, or if there are fewer than 2 rows.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be strictly between 0 and 1, got {test_fraction}")
    row_counts = {len(array) for array in arrays.values()}
    if len(row_counts) != 1:
        raise ValueError(f"all arrays must have the same number of rows, got {sorted(row_counts)}")
    num_rows = row_counts.pop()
    if num_rows < 2:
        raise ValueError(f"need at least 2 rows to split, got {num_rows}")
    num_test_rows = min(max(round(num_rows * test_fraction), 1), num_rows - 1)
    train = {key: array[num_test_rows:].copy() for key, array in arrays.items()}
    test = {key: array[:num_test_rows].copy() for key, array in arrays.items()}
    return train, test

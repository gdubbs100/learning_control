import copy

import numpy as np
import pandas as pd

from utils.reporting.select_columns_by_prefix import select_columns_by_prefix


def example_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode": [0, 1],
            "model_param_A_0_0": [0.9, np.nan],
            "buffer_size": [10.0, 20.0],
            "model_param_B_1": [0.1, np.nan],
            "model_train_mse": [0.0, np.nan],
        }
    )


def test_select_columns_by_prefix_returns_matching_columns_in_table_order() -> None:
    assert select_columns_by_prefix(example_table(), "model_param_") == ["model_param_A_0_0", "model_param_B_1"]


def test_select_columns_by_prefix_matches_only_the_start_of_the_name() -> None:
    assert select_columns_by_prefix(example_table(), "mse") == []
    assert select_columns_by_prefix(example_table(), "model_") == [
        "model_param_A_0_0",
        "model_param_B_1",
        "model_train_mse",
    ]


def test_select_columns_by_prefix_with_no_match_returns_empty_list() -> None:
    assert select_columns_by_prefix(example_table(), "loss") == []


def test_select_columns_by_prefix_on_table_without_rows_still_selects_columns() -> None:
    table = pd.DataFrame({"episode": [], "model_param_B_0": []})
    assert select_columns_by_prefix(table, "model_param_") == ["model_param_B_0"]


def test_select_columns_by_prefix_returns_a_list_of_strings() -> None:
    selected = select_columns_by_prefix(example_table(), "model_")
    assert isinstance(selected, list)
    assert all(isinstance(name, str) for name in selected)


def test_select_columns_by_prefix_is_deterministic() -> None:
    table = example_table()
    assert select_columns_by_prefix(table, "model_") == select_columns_by_prefix(table, "model_")


def test_select_columns_by_prefix_does_not_mutate_input() -> None:
    table = example_table()
    original_table = copy.deepcopy(table)
    select_columns_by_prefix(table, "model_")
    pd.testing.assert_frame_equal(table, original_table)

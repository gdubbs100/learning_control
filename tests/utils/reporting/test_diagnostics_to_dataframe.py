import copy

import numpy as np

from utils.reporting.diagnostics_to_dataframe import diagnostics_to_dataframe


def test_diagnostics_to_dataframe_one_row_per_episode() -> None:
    table = diagnostics_to_dataframe(
        [{"loss": 1.0, "grad_norm": 2.0}, {"loss": 0.5, "grad_norm": 1.5}]
    )
    assert list(table.columns) == ["episode", "loss", "grad_norm"]
    assert table["episode"].tolist() == [0, 1]
    assert table["loss"].tolist() == [1.0, 0.5]
    assert table["grad_norm"].tolist() == [2.0, 1.5]


def test_diagnostics_to_dataframe_empty_list_has_only_episode_column() -> None:
    table = diagnostics_to_dataframe([])
    assert list(table.columns) == ["episode"]
    assert len(table) == 0


def test_diagnostics_to_dataframe_missing_key_is_nan() -> None:
    table = diagnostics_to_dataframe([{"loss": 1.0}, {"loss": 0.5, "mse": 0.1}])
    assert np.isnan(table["mse"].iloc[0])
    assert table["mse"].iloc[1] == 0.1


def test_diagnostics_to_dataframe_is_deterministic() -> None:
    diagnostics_log = [{"loss": 1.0}, {"loss": 0.5}]
    assert diagnostics_to_dataframe(diagnostics_log).equals(diagnostics_to_dataframe(diagnostics_log))


def test_diagnostics_to_dataframe_does_not_mutate_input() -> None:
    diagnostics_log = [{"loss": 1.0}, {"loss": 0.5}]
    original_diagnostics_log = copy.deepcopy(diagnostics_log)
    diagnostics_to_dataframe(diagnostics_log)
    assert diagnostics_log == original_diagnostics_log

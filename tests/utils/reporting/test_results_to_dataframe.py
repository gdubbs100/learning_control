import copy

import pandas as pd

from utils.reporting.results_to_dataframe import results_to_dataframe


def example_results() -> dict[str, list[float]]:
    return {"episode_returns": [10.0, 20.0], "episode_lengths": [10.0, 20.0]}


def test_results_to_dataframe_two_episodes() -> None:
    dataframe = results_to_dataframe(example_results())
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 2
    assert list(dataframe.columns) == ["episode", "return", "steps"]
    assert list(dataframe["episode"]) == [0, 1]
    assert list(dataframe["return"]) == [10.0, 20.0]
    assert list(dataframe["steps"]) == [10.0, 20.0]


def test_results_to_dataframe_is_deterministic() -> None:
    first_dataframe = results_to_dataframe(example_results())
    second_dataframe = results_to_dataframe(example_results())
    pd.testing.assert_frame_equal(first_dataframe, second_dataframe)


def test_results_to_dataframe_does_not_mutate_input() -> None:
    results = example_results()
    original_results = copy.deepcopy(results)
    results_to_dataframe(results)
    assert results == original_results

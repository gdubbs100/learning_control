import copy

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from utils.plotting.plot_columns_over_episodes import plot_columns_over_episodes


def example_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode": [0, 1, 2, 3],
            "cost": [4.0, 3.0, 2.0, 1.0],
            "sparse": [np.nan, 5.0, np.nan, 7.0],
        }
    )


def line_by_label(figure: Figure) -> dict[str, tuple[list[float], list[float]]]:
    """Plotted (x, y) data of each line, keyed by its legend label."""
    return {
        line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
        for line in figure.axes[0].get_lines()
    }


def test_plot_columns_over_episodes_returns_figure_with_one_axes() -> None:
    figure = plot_columns_over_episodes(example_table(), ["cost"], "Cost", "Cost per episode")
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1


def test_plot_columns_over_episodes_draws_one_line_per_column_against_episode() -> None:
    figure = plot_columns_over_episodes(example_table(), ["cost", "sparse"], "Value", "Title")
    lines = line_by_label(figure)
    assert set(lines) == {"cost", "sparse"}
    assert lines["cost"] == ([0, 1, 2, 3], [4.0, 3.0, 2.0, 1.0])


def test_plot_columns_over_episodes_skips_missing_values() -> None:
    figure = plot_columns_over_episodes(example_table(), ["sparse"], "Value", "Title")
    assert line_by_label(figure)["sparse"] == ([1, 3], [5.0, 7.0])


def test_plot_columns_over_episodes_column_of_only_missing_values_has_an_empty_line() -> None:
    table = pd.DataFrame({"episode": [0, 1], "empty": [np.nan, np.nan]})
    figure = plot_columns_over_episodes(table, ["empty"], "Value", "Title")
    assert line_by_label(figure)["empty"] == ([], [])


def test_plot_columns_over_episodes_labels_title_and_legend() -> None:
    figure = plot_columns_over_episodes(example_table(), ["cost", "sparse"], "My y label", "My title")
    axes = figure.axes[0]
    assert axes.get_xlabel() == "Episode"
    assert axes.get_ylabel() == "My y label"
    assert axes.get_title() == "My title"
    legend_texts = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend_texts == ["cost", "sparse"]


def test_plot_columns_over_episodes_with_no_columns_gives_an_empty_labelled_figure() -> None:
    figure = plot_columns_over_episodes(example_table(), [], "Value", "Nothing here")
    axes = figure.axes[0]
    assert axes.get_lines() == []
    assert axes.get_title() == "Nothing here"


def test_plot_columns_over_episodes_is_deterministic() -> None:
    first = plot_columns_over_episodes(example_table(), ["cost", "sparse"], "Value", "Title")
    second = plot_columns_over_episodes(example_table(), ["cost", "sparse"], "Value", "Title")
    assert line_by_label(first) == line_by_label(second)


def test_plot_columns_over_episodes_does_not_mutate_input() -> None:
    table = example_table()
    original_table = copy.deepcopy(table)
    columns = ["cost", "sparse"]
    plot_columns_over_episodes(table, columns, "Value", "Title")
    pd.testing.assert_frame_equal(table, original_table)
    assert columns == ["cost", "sparse"]

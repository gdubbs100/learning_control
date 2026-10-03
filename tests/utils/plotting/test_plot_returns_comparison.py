import copy

from matplotlib.figure import Figure

from utils.plotting.plot_returns_comparison import plot_returns_comparison

EXAMPLE_RETURNS_BY_AGENT = {"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0]}


def figure_contents(figure: Figure) -> tuple:
    """The plotted line data, legend labels and axes labels, for comparing figures."""
    axes = figure.axes[0]
    lines = [(list(line.get_xdata()), list(line.get_ydata())) for line in axes.get_lines()]
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    return lines, labels, axes.get_xlabel(), axes.get_ylabel(), axes.get_title()


def test_plot_returns_comparison_returns_figure_with_one_axes_and_a_line_per_agent() -> None:
    figure = plot_returns_comparison(copy.deepcopy(EXAMPLE_RETURNS_BY_AGENT))
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].get_lines()) == 2


def test_plot_returns_comparison_legend_labels_are_agent_names() -> None:
    figure = plot_returns_comparison(copy.deepcopy(EXAMPLE_RETURNS_BY_AGENT))
    legend_labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert legend_labels == ["a", "b"]


def test_plot_returns_comparison_line_data_matches_input() -> None:
    figure = plot_returns_comparison(copy.deepcopy(EXAMPLE_RETURNS_BY_AGENT))
    first_line, second_line = figure.axes[0].get_lines()
    assert list(first_line.get_xdata()) == [0, 1, 2]
    assert list(first_line.get_ydata()) == [1.0, 2.0, 3.0]
    assert list(second_line.get_xdata()) == [0, 1]
    assert list(second_line.get_ydata()) == [2.0, 2.0]


def test_plot_returns_comparison_axes_have_labels_and_title() -> None:
    axes = plot_returns_comparison(copy.deepcopy(EXAMPLE_RETURNS_BY_AGENT)).axes[0]
    assert axes.get_xlabel() != ""
    assert axes.get_ylabel() != ""
    assert axes.get_title() != ""


def test_plot_returns_comparison_is_deterministic() -> None:
    first_figure = plot_returns_comparison(copy.deepcopy(EXAMPLE_RETURNS_BY_AGENT))
    second_figure = plot_returns_comparison(copy.deepcopy(EXAMPLE_RETURNS_BY_AGENT))
    assert figure_contents(first_figure) == figure_contents(second_figure)


def test_plot_returns_comparison_does_not_mutate_input() -> None:
    returns_by_agent = copy.deepcopy(EXAMPLE_RETURNS_BY_AGENT)
    plot_returns_comparison(returns_by_agent)
    assert returns_by_agent == EXAMPLE_RETURNS_BY_AGENT

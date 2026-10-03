import copy

from matplotlib.figure import Figure

from utils.plotting.plot_episode_returns import plot_episode_returns

EXAMPLE_EPISODE_RETURNS = [3.0, 1.0, 4.0, 1.0, 5.0]


def figure_contents(figure: Figure) -> tuple:
    """The plotted line data and the axes labels and title, for comparing figures."""
    axes = figure.axes[0]
    line = axes.get_lines()[0]
    return (
        list(line.get_xdata()),
        list(line.get_ydata()),
        axes.get_xlabel(),
        axes.get_ylabel(),
        axes.get_title(),
    )


def test_plot_episode_returns_returns_figure_with_one_axes() -> None:
    figure = plot_episode_returns(list(EXAMPLE_EPISODE_RETURNS))
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1


def test_plot_episode_returns_line_data_matches_input() -> None:
    figure = plot_episode_returns(list(EXAMPLE_EPISODE_RETURNS))
    line = figure.axes[0].get_lines()[0]
    assert list(line.get_xdata()) == [0, 1, 2, 3, 4]
    assert list(line.get_ydata()) == EXAMPLE_EPISODE_RETURNS


def test_plot_episode_returns_axes_have_labels_and_title() -> None:
    figure = plot_episode_returns(list(EXAMPLE_EPISODE_RETURNS))
    axes = figure.axes[0]
    assert axes.get_xlabel() != ""
    assert axes.get_ylabel() != ""
    assert axes.get_title() != ""


def test_plot_episode_returns_is_deterministic() -> None:
    first_figure = plot_episode_returns(list(EXAMPLE_EPISODE_RETURNS))
    second_figure = plot_episode_returns(list(EXAMPLE_EPISODE_RETURNS))
    assert figure_contents(first_figure) == figure_contents(second_figure)


def test_plot_episode_returns_does_not_mutate_input() -> None:
    episode_returns = list(EXAMPLE_EPISODE_RETURNS)
    original_episode_returns = copy.deepcopy(episode_returns)
    plot_episode_returns(episode_returns)
    assert episode_returns == original_episode_returns

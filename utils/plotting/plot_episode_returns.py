from matplotlib.figure import Figure


def plot_episode_returns(episode_returns: list[float]) -> Figure:
    """Plot the return of each episode as a line, without saving or showing it.

    The figure is built directly (not through pyplot), so no global plotting
    state is touched.

    Args:
        episode_returns: the return of each episode, in order.

    Returns:
        A Figure with one Axes holding a line of return (y) against the
        zero-based episode index (x), with axis labels and a title.
    """
    episode_indices = list(range(len(episode_returns)))
    figure = Figure()
    axes = figure.add_subplot()
    axes.plot(episode_indices, list(episode_returns))
    axes.set_xlabel("Episode")
    axes.set_ylabel("Return")
    axes.set_title("Return per episode")
    return figure

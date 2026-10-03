from matplotlib.figure import Figure


def plot_returns_comparison(returns_by_agent: dict[str, list[float]]) -> Figure:
    """Plot the per-episode returns of several agents on one set of axes, without saving or showing it.

    The figure is built directly (not through pyplot), so no global plotting
    state is touched.

    Args:
        returns_by_agent: for each agent name, the return of each episode in order.
            Agents may have run different numbers of episodes.

    Returns:
        A Figure with one Axes holding one line per agent of return (y) against
        the zero-based episode index (x), labelled with the agent name in a
        legend, with axis labels and a title.
    """
    figure = Figure()
    axes = figure.add_subplot()
    for agent_name, episode_returns in returns_by_agent.items():
        axes.plot(list(range(len(episode_returns))), list(episode_returns), label=agent_name)
    axes.set_xlabel("Episode")
    axes.set_ylabel("Return")
    axes.set_title("Return per episode by agent")
    axes.legend()
    return figure

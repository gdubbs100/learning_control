def format_episode_log(episode_index: int, episode_return: float, num_steps: int) -> str:
    """Format one episode's results as a single log line.

    Args:
        episode_index: the zero-based index of the episode.
        episode_return: the total reward collected in the episode.
        num_steps: the number of steps the episode lasted.

    Returns:
        A line of the form "episode=<index> return=<return> steps=<steps>",
        e.g. "episode=3 return=22.0 steps=22".
    """
    return f"episode={episode_index} return={float(episode_return)} steps={num_steps}"

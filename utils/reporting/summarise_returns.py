import statistics


def summarise_returns(episode_returns: list[float]) -> dict[str, float]:
    """Summarise a list of episode returns with simple statistics.

    Args:
        episode_returns: the return of each episode, in order. Must not be empty.

    Returns:
        A dictionary with keys "mean", "std" (population standard deviation),
        "min", "max" and "num_episodes" (the number of returns, as a float).

    Raises:
        ValueError: if `episode_returns` is empty.
    """
    if len(episode_returns) == 0:
        raise ValueError("episode_returns must contain at least one return")
    return {
        "mean": statistics.fmean(episode_returns),
        "std": statistics.pstdev(episode_returns),
        "min": min(episode_returns),
        "max": max(episode_returns),
        "num_episodes": float(len(episode_returns)),
    }

def has_converged(episode_returns: list[float], window: int, tolerance: float) -> bool:
    """Say whether recent episode returns have stopped changing.

    Compares the mean of the last `window` returns with the mean of the `window`
    returns before them.

    Args:
        episode_returns: the return of each episode so far, in order.
        window: the number of episodes in each of the two windows compared.
        tolerance: the largest absolute difference between the two window means
            that still counts as converged.

    Returns:
        True if there are at least 2 * `window` returns and the absolute
        difference between the two window means is at most `tolerance`, else False.
    """
    if len(episode_returns) < 2 * window:
        return False
    recent_returns = episode_returns[-window:]
    previous_returns = episode_returns[-2 * window : -window]
    recent_mean = sum(recent_returns) / window
    previous_mean = sum(previous_returns) / window
    return abs(recent_mean - previous_mean) <= tolerance

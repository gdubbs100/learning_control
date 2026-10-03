def is_last_episode(episode_index: int, num_episodes: int) -> bool:
    """Say whether an episode is the final one of an experiment.

    Args:
        episode_index: the zero-based index of the episode.
        num_episodes: the total number of episodes in the experiment.

    Returns:
        True if `episode_index` is the last episode, `num_episodes - 1`, else False.
    """
    return episode_index == num_episodes - 1

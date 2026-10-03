import pandas as pd


def results_to_dataframe(results: dict[str, list[float]]) -> pd.DataFrame:
    """Convert per-episode experiment results into a table with one row per episode.

    Args:
        results: per-episode metrics with keys "episode_returns" and
            "episode_lengths", each a list with one value per episode.

    Returns:
        A DataFrame with columns "episode" (zero-based episode index),
        "return" (from "episode_returns") and "steps" (from "episode_lengths").
    """
    episode_returns = list(results["episode_returns"])
    episode_lengths = list(results["episode_lengths"])
    return pd.DataFrame(
        {
            "episode": list(range(len(episode_returns))),
            "return": episode_returns,
            "steps": episode_lengths,
        }
    )

import pandas as pd


def diagnostics_to_dataframe(diagnostics_log: list[dict[str, float]]) -> pd.DataFrame:
    """Convert per-episode agent diagnostics into a table with one row per episode.

    Args:
        diagnostics_log: one dict of diagnostic name to value for each episode, in order.
            Dicts may have different keys.

    Returns:
        A DataFrame with an "episode" column (zero-based episode index) followed by
        one column per diagnostic name, in order of first appearance. A diagnostic
        missing from an episode is NaN. An empty log gives only the "episode" column.
    """
    table = pd.DataFrame(list(diagnostics_log))
    table.insert(0, "episode", list(range(len(diagnostics_log))))
    return table

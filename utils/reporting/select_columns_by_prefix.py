import pandas as pd


def select_columns_by_prefix(table: pd.DataFrame, prefix: str) -> list[str]:
    """List the names of the columns of a table that start with a prefix.

    Args:
        table: the table whose column names are searched.
        prefix: the text the column names must start with.

    Returns:
        The matching column names as strings, in the order they appear in the table.
        Empty if none match.
    """
    return [str(name) for name in table.columns if str(name).startswith(prefix)]

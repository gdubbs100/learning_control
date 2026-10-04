import pandas as pd
from matplotlib.figure import Figure


def plot_columns_over_episodes(table: pd.DataFrame, columns: list[str], ylabel: str, title: str) -> Figure:
    """Plot columns of a per-episode table against the episode index, without saving or showing it.

    Missing values (NaN) are skipped for each column separately, so a column that is
    only recorded in some episodes is drawn as a line joining the recorded points.
    The figure is built directly (not through pyplot), so no global plotting state is touched.

    Args:
        table: a table with an "episode" column and one column per series to plot.
        columns: the names of the columns to plot, one line each, in legend order.
        ylabel: the label of the y axis.
        title: the title of the plot.

    Returns:
        A Figure with one Axes holding a line (with point markers) per column, labelled
        with the column name in a legend. With no columns, the Axes is empty and has no legend.
    """
    figure = Figure()
    axes = figure.add_subplot()
    for column in columns:
        recorded = table[["episode", column]].dropna()
        axes.plot(list(recorded["episode"]), list(recorded[column]), marker="o", markersize=3, label=column)
    axes.set_xlabel("Episode")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if columns:
        axes.legend(fontsize="small")
    return figure

"""Data access.

At a firm this module is where the shared data layer lives: clients for the
market-data databases and the object store. Here it is a thin CSV helper, but
the principle is the same. Every project loads data through one place instead
of each reinventing the plumbing.
"""

import pandas as pd


def load_prices(path: str, date_column: str = "date") -> pd.DataFrame:
    """Load a price CSV into a DataFrame indexed by a sorted date column.

    Args:
        path: path to the CSV file.
        date_column: name of the column holding dates.

    Returns:
        A DataFrame indexed by date, sorted ascending.
    """
    frame = pd.read_csv(path, parse_dates=[date_column])
    return frame.set_index(date_column).sort_index()

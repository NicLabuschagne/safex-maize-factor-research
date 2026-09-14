"""Performance metrics for return and equity series."""

import pandas as pd

# Trading days in a year, used to annualize daily figures
TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio of a return series.

    Args:
        returns: periodic (e.g. daily) returns.
        periods_per_year: scaling factor, 252 for daily data.

    Returns:
        The annualized Sharpe, or 0.0 if the series has no variation.
    """
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * (periods_per_year ** 0.5)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualized return from a series of periodic returns.

    Compounds the actual returns, then scales to a one-year horizon.
    """
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    total_growth = (1 + returns).prod()
    return total_growth ** (periods_per_year / n_periods) - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough drop of an equity curve, as a negative fraction.

    Args:
        equity_curve: cumulative value over time (not returns).

    Returns:
        The worst drawdown, e.g. -0.25 for a 25 percent fall from a peak.
    """
    running_peak = equity_curve.cummax()          # highest value seen so far
    drawdown = equity_curve / running_peak - 1.0  # drop from that peak
    return drawdown.min()

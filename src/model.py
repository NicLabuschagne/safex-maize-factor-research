"""XGBoost model with walk-forward validation.

The layout is deliberately two levels:

  OUTER loop - walk forward through time. Each fold trains on everything
               before a date and tests on the block after it. This is what
               gets reported.

  INNER loop - inside one outer fold's training data only, hold back the last
               slice as a validation set and try each hyperparameter
               combination on it. The winner is refitted on the full training
               data and used on the outer test block.

The inner loop never sees the outer test block, so hyperparameter choices
cannot leak into the reported score.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor

import metrics

# A sensible starting point for low signal-to-noise financial data: shallow
# trees, slow learning, and heavy sampling. Deeper trees memorise the noise.
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 0,
    "n_jobs": 4,
}


def make_model(kind: str = "regressor", **params):
    """Build an XGB model. kind is 'regressor' or 'classifier'."""
    settings = dict(DEFAULT_PARAMS)
    settings.update(params)

    if kind == "regressor":
        return XGBRegressor(**settings)
    if kind == "classifier":
        return XGBClassifier(**settings)
    raise ValueError("kind must be 'regressor' or 'classifier'")


def param_grid(options: dict) -> list[dict]:
    """Turn {'max_depth': [2, 3], 'learning_rate': [0.03, 0.1]} into a list of dicts.

    Returns every combination, so the list length is the product of the option
    counts. Keep it small - each combination is a full model fit per fold.
    """
    names = list(options.keys())
    combinations = []

    for values in itertools.product(*options.values()):
        combinations.append(dict(zip(names, values)))

    return combinations


def walk_forward_splits(n_rows: int, n_splits: int, test_size: int, embargo: int,
                        min_test_start: int = 0, min_train_rows: int = 250) -> list[tuple]:
    """Expanding train window, fixed size test block, with a gap in between.

    The embargo must be at least the forecast horizon. A 5-day target at row i
    reads prices up to row i+5, so training right up to the test block would
    put five days of the future into the fit.

    min_test_start - drop any fold that would test at or before this row. Set it
                     to the end of the feature-screening window so no fold is
                     scored on data the screen already chose features from.
    min_train_rows - drop any fold with less training data than this, otherwise
                     the earliest fold can end up fitting on a handful of rows.
    """
    splits = []
    first_test_start = n_rows - n_splits * test_size

    for fold in range(n_splits):
        test_start = first_test_start + fold * test_size
        train_end = test_start - embargo

        if test_start < min_test_start:
            continue
        if train_end < min_train_rows:
            continue

        train_index = np.arange(0, train_end)
        test_index = np.arange(test_start, test_start + test_size)
        splits.append((train_index, test_index))

    return splits


def score_predictions(prediction, actual) -> float:
    """Rank IC - the metric that matters here, not R squared."""
    ic, _ = spearmanr(prediction, actual)
    return ic


def choose_params(X: pd.DataFrame, y: pd.Series, train_index, grid: list[dict],
                  kind: str, validation_fraction: float, embargo: int) -> dict:
    """Inner loop: pick the best hyperparameters using the tail of the training data.

    Splits the training rows into an earlier part to fit on and a later part to
    score on, keeping an embargo gap between them for the same reason as the
    outer loop.
    """
    n_validation = int(len(train_index) * validation_fraction)
    inner_train = train_index[: len(train_index) - n_validation - embargo]
    inner_validation = train_index[len(train_index) - n_validation:]

    best_params = grid[0]
    best_score = -np.inf

    for params in grid:
        model = make_model(kind, **params)
        model.fit(X.iloc[inner_train], y.iloc[inner_train])
        prediction = model.predict(X.iloc[inner_validation])
        score = score_predictions(prediction, y.iloc[inner_validation])

        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def run_walk_forward(X: pd.DataFrame, y: pd.Series, n_splits: int = 6, embargo: int = 5,
                     kind: str = "regressor", grid: list[dict] | None = None,
                     validation_fraction: float = 0.2, min_test_start: int = 0,
                     min_train_rows: int = 250) -> dict:
    """Walk forward through the data and score each fold out of sample.

    Pass `grid` to turn the inner hyperparameter search on. Leave it as None to
    use DEFAULT_PARAMS everywhere, which is faster while you are still building
    features.

    Returns a dict with:
      folds       - one row per fold, with dates and the out of sample IC
      predictions - every out of sample prediction, indexed by date
      summary     - mean IC, how many folds were positive, and the spread
    """
    test_size = len(X) // 10
    splits = walk_forward_splits(len(X), n_splits, test_size, embargo,
                                 min_test_start=min_test_start, min_train_rows=min_train_rows)

    fold_rows = []
    predictions = []

    for fold, (train_index, test_index) in enumerate(splits):
        if grid is None:
            params = dict(DEFAULT_PARAMS)
        else:
            params = choose_params(X, y, train_index, grid, kind, validation_fraction, embargo)

        model = make_model(kind, **params)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        prediction = model.predict(X.iloc[test_index])

        predictions.append(pd.Series(prediction, index=X.index[test_index]))
        fold_rows.append({
            "fold": fold,
            "train_end": X.index[train_index[-1]].date(),
            "test_start": X.index[test_index[0]].date(),
            "test_end": X.index[test_index[-1]].date(),
            "n_train": len(train_index),
            "n_test": len(test_index),
            "oos_ic": score_predictions(prediction, y.iloc[test_index]),
            "params": params,
        })

    folds = pd.DataFrame(fold_rows)

    return {
        "folds": folds,
        "predictions": pd.concat(predictions),
        "summary": {
            "mean_oos_ic": float(folds["oos_ic"].mean()),
            "folds_positive": int((folds["oos_ic"] > 0).sum()),
            "n_folds": len(folds),
            "ic_std": float(folds["oos_ic"].std()),
        },
    }


def to_simple_returns(pnl: pd.Series, notional: pd.Series | None = None,
                      log_returns: bool = False) -> pd.Series:
    """Turn per-trade P&L into a pct of capital, so any product compares.

    Three cases, pick the one that matches your target:

      notional given   - P&L is in price units (a spread target in USD/tonne).
                         Divide by the value of the position to get a fraction.
                         For a spread this is a convention: you are quoting the
                         move as a share of one leg's notional, not of the
                         margin actually posted.
      log_returns=True - P&L is a log return.
      neither          - P&L is already a fraction, use it as is.
    """
    if notional is not None:
        return pnl / notional.reindex(pnl.index)
    if log_returns:
        return np.expm1(pnl)
    return pnl


def equity_curve(returns: pd.Series) -> pd.Series:
    """Growth of 1 unit of capital. Compounded, so it is a product not a sum."""
    return (1.0 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional drop below the running peak, e.g. -0.25 for a 25 percent fall.

    Because this is a ratio it is unit free, which is the whole point: a 12
    percent drawdown means the same thing whether the product is maize in
    USD/tonne or an outright in log returns.
    """
    equity = equity_curve(returns)
    return equity / equity.cummax() - 1.0


def single_sample_stats(returns: pd.Series, periods_per_year: float) -> dict:
    """Trading statistics for ONE set of non-overlapping trade returns."""
    wins = returns > 0

    drawdown = drawdown_series(returns)
    max_draw_down = drawdown.min()

    # Sortino only counts downside moves, so a strategy with a few large wins
    # is not penalised the way Sharpe penalises it.
    downside = returns.where(returns < 0, 0.0)
    downside_deviation = np.sqrt((downside ** 2).mean())

    total_return = equity_curve(returns).iloc[-1] - 1.0
    annual_return = metrics.annualized_return(returns, periods_per_year)

    return {
        "n": int(len(returns)),
        "win_rate": float(wins.mean()),
        "ev_pct": float(returns.mean()),
        "median_pct": float(returns.median()),
        "avg_win_pct": float(returns[returns > 0].mean()),
        "avg_loss_pct": float(returns[returns < 0].mean()),
        "total_return_pct": float(total_return),
        "annual_return_pct": float(annual_return),
        "max_draw_down_pct": float(max_draw_down),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else float("nan"),
        "sortino": float(returns.mean() / downside_deviation * np.sqrt(periods_per_year)) if downside_deviation > 0 else float("nan"),
        "calmar": float(annual_return / abs(max_draw_down)) if max_draw_down < 0 else float("nan"),
        "t_stat": float(returns.mean() / (returns.std() / np.sqrt(len(returns)))) if returns.std() > 0 else float("nan"),
    }


def performance_stats(signal: pd.Series, forward_return: pd.Series, periods_per_year: float,
                      notional: pd.Series | None = None, log_returns: bool = False,
                      horizon: int = 1) -> dict:
    """Treat the prediction as a long/short signal and report trading statistics.

    Everything with a _pct suffix is a fraction of capital: 0.12 is 12 percent.
    Pass `notional` or `log_returns` so the P&L can be converted - see
    to_simple_returns. Without one of them the numbers stay in whatever units
    forward_return came in, and stop being comparable across products.

    horizon - how many rows a trade is held for. Pass the FULL overlapping
              sample and let this do the thinning.

              Above 1, consecutive rows share days, so only every horizon-th
              trade is independent. There are `horizon` equally valid ways to
              pick that subset (start at row 0, row 1, ...) and they can give
              very different answers on a short sample. So every offset is
              measured and the results averaged, which uses each day exactly
              once and removes an arbitrary choice. sharpe_min and sharpe_max
              show how much the offset actually mattered - if they are far
              apart, no single figure here means much.
    """
    pnl = np.sign(signal) * forward_return
    returns = to_simple_returns(pnl, notional, log_returns).dropna()

    if horizon <= 1:
        return single_sample_stats(returns, periods_per_year)

    per_offset = []
    for offset in range(horizon):
        per_offset.append(single_sample_stats(returns.iloc[offset::horizon], periods_per_year))

    table = pd.DataFrame(per_offset)
    averaged = table.mean().to_dict()

    averaged["n"] = int(round(averaged["n"]))
    averaged["n_offsets"] = horizon
    averaged["sharpe_min"] = float(table["sharpe"].min())
    averaged["sharpe_max"] = float(table["sharpe"].max())

    return averaged

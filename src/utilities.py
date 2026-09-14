"""Feature transforms used across research projects."""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

# ── Optional heavy imports — never break the workspace if one is missing ──────

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
    mdates = None

try:
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.stattools import acf, adfuller, coint, kpss, pacf
except ImportError:  # pragma: no cover
    sm = None
    adfuller = kpss = coint = None
    acf = pacf = acorr_ljungbox = None
    AutoReg = None
    plot_acf = plot_pacf = None

try:
    from scipy import optimize, stats
except ImportError:  # pragma: no cover
    optimize = stats = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    px = go = None

try:
    import sklearn
    from sklearn import linear_model, metrics, preprocessing
    from sklearn.model_selection import TimeSeriesSplit
except ImportError:  # pragma: no cover
    sklearn = linear_model = metrics = preprocessing = TimeSeriesSplit = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = nn = None

try:
    from data_profiling import ProfileReport
except ImportError:  # pragma: no cover
    ProfileReport = None

try:
    from lse import LSE
except ImportError:  # pragma: no cover
    LSE = None

# Set Themes
from theme import set_theme as _set_plot_theme

_set_plot_theme()


# ── Workspace paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("research")


def data_path(product: str, layer: str = "processed") -> Dict[str, Any]:
    """
    Resolve the data directory for a product without hardcoding paths.

    Parameters
    ----------
    product : 'energy' | 'fx' | 'crypto' (any folder at the workspace root)
    layer   : 'raw' | 'processed'

    Returns
    -------
    dict with keys: path (Path), exists (bool)
    """
    path = ROOT / product / "data" / layer
    return {"path": path, "exists": path.exists()}


def save_processed(df: pd.DataFrame, product: str, name: str) -> Dict[str, Any]:
    """
    Save a DataFrame to <product>/data/processed/<name>.parquet.

    Returns
    -------
    dict with keys: path (Path), rows (int), size_mb (float)
    """
    directory = data_path(product, "processed")["path"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    df.to_parquet(path)
    return {
        "path": path,
        "rows": len(df),
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }


def load_processed(product: str, name: str) -> Dict[str, Any]:
    """
    Load <product>/data/processed/<name>.parquet.

    Returns
    -------
    dict with keys: df (DataFrame), path (Path), rows (int)
    """
    path = data_path(product, "processed")["path"] / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed file: {path}")
    df = pd.read_parquet(path)
    return {"df": df, "path": path, "rows": len(df)}


# ── Notebook / display setup ───────────────────────────────────────────────────

def setup_notebook(
    figsize: Tuple[int, int] = (12, 5),
    max_rows: int = 60,
    max_columns: int = 40,
) -> Dict[str, Any]:
    """
    Apply standard pandas display options and matplotlib defaults.
    Call once at the top of every notebook.

    Returns
    -------
    dict with keys: pandas_options (dict), matplotlib (bool)
    """
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    mpl_ok = plt is not None
    if mpl_ok:
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3

    warnings.filterwarnings("ignore", category=FutureWarning)

    return {
        "pandas_options": {
            "max_rows": max_rows,
            "max_columns": max_columns,
        },
        "matplotlib": mpl_ok,
    }


# ── Frame inspection ───────────────────────────────────────────────────────────

def frame_summary(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    """
    Structured snapshot of a DataFrame: shape, per-column diagnostics, index
    health, and head/tail samples.

    Parameters
    ----------
    df : DataFrame to inspect
    n  : rows to keep in the head/tail samples

    Returns
    -------
    dict with keys: shape (tuple), memory_mb (float), columns (DataFrame),
                    index (dict), head (DataFrame), tail (DataFrame)
    """
    columns = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(),
    })

    numeric = df.select_dtypes("number")
    if not numeric.empty:
        columns = columns.join(pd.DataFrame({
            "min": numeric.min(),
            "mean": numeric.mean(),
            "max": numeric.max(),
        }))

    index: Dict[str, Any] = {
        "type": type(df.index).__name__,
        "monotonic": bool(df.index.is_monotonic_increasing),
        "duplicates": int(df.index.duplicated().sum()),
    }
    if len(df.index):
        index["start"] = df.index[0]
        index["end"] = df.index[-1]
    if isinstance(df.index, pd.DatetimeIndex):
        index["tz"] = str(df.index.tz)
        steps = df.index.to_series().diff().dropna()
        if not steps.empty:
            # modal step is the bar size; anything longer is a gap in the series
            step = steps.mode().iloc[0]
            index["step"] = step
            index["gaps"] = int((steps > step).sum())

    return {
        "shape": df.shape,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns": columns,
        "index": index,
        "head": df.head(n),
        "tail": df.tail(n),
    }


def peek(df: pd.DataFrame, n: int = 5) -> None:
    """
    Print `frame_summary` as one readable console block.

    """
    summary = frame_summary(df, n=n)
    rows, cols = summary["shape"]
    print(f"shape: {rows:,} rows x {cols} cols   ({summary['memory_mb']} MB)")

    index = summary["index"]
    line = [f"index: {index['type']}"]
    if "start" in index:
        line.append(f"{index['start']} -> {index['end']}")
    line.append(f"monotonic={index['monotonic']}  dupes={index['duplicates']}")
    if "gaps" in index:
        line.append(f"step={index['step']}  gaps={index['gaps']}")
    print("  ".join(line), end="\n\n")

    print(summary["columns"].to_string())
    print(f"\nhead({n})\n{summary['head'].to_string()}")
    print(f"\ntail({n})\n{summary['tail'].to_string()}")


# data series from LSE
# api key
api_key = os.environ.get("LSE_API_KEY", "")
client = LSE(api_key=api_key) if (LSE is not None and api_key) else None


# load data series

def load_lse_data(client, asset: str, timeframe: str, limit):
    # load
    rows = client.candles(asset, timeframe, limit=limit, order="desc")
    # build df
    df = (pd.DataFrame(rows)
          .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
          .set_index("timestamp")
          .sort_index())
    return df

# ── Common transforms ──────────────────────────────────────────────────────────
def quick_adf(series: pd.Series) -> Dict[str, Any]:
    """
    Run an ADF stationarity test with sensible defaults.

    Returns
    -------
    dict with keys: stat (float), pvalue (float), n_lags (int),
                    stationary_5pct (bool)
    """
    if adfuller is None:
        raise ImportError("statsmodels is not installed")
    clean = series.dropna()
    stat, pvalue, n_lags, *_ = adfuller(clean, autolag="AIC")
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "n_lags": int(n_lags),
        "stationary_5pct": bool(pvalue < 0.05),
    }

def log_transform(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        df[f"{col}_log_return"] = np.log(df[col]).diff()
    return df

# build stationarity test using ADfuller
def ad_fuller_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Takes in a log transformed df and outputs P-value statistics for stationarity"""
    results = adfuller(df, result_object=True)
    print(results)
    print(f"p_value: {results.pvalue:.3f}")
    return results


# lags of series
def add_lags(
    df: pd.DataFrame,
    cols: str | List[str],
    n_lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add `<col>_lag_<n>` columns for every (column, lag) pair.

    Returns a new frame; `df` is not modified.
    """
    if isinstance(cols, str):
        cols = [cols]
    n_lags = n_lags or []
    new_columns = {}
    for col_name in cols:
        for lag_value in n_lags:
            new_col_name = f"{col_name}_lag_{lag_value}"
            # bind loop vars as defaults so each lambda keeps its own pair
            new_columns[new_col_name] = (
                lambda df_inner, column=col_name, lag=lag_value: df_inner[column].shift(lag)
            )
    return df.assign(**new_columns)

# calculate sharpe ratio
def calculate_sharpe(
    df: pd.DataFrame,
    returns: str,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> Dict[str, Any]:
    """
    Annualised Sharpe ratio for a column of per-period returns.

    Sharpe = mean(excess) / std(excess) * sqrt(periods_per_year), where excess
    subtracts the risk-free rate scaled down to the bar frequency.

    Parameters
    ----------
    df               : frame holding the return series
    returns          : column name of per-period (not cumulative) returns
    periods_per_year : bars per year — 252 for daily, 252*390 for 1-minute US
                       equity bars. Getting this wrong rescales the result.
    risk_free        : annual risk-free rate as a decimal (0.05 = 5%)

    Returns
    -------
    dict with keys: sharpe (float), mean_period (float), std_period (float),
                    ann_return (float), ann_vol (float), n (int)
    """
    series = df[returns].dropna()
    if series.empty:
        raise ValueError(f"Column '{returns}' has no non-null values")

    excess = series - (risk_free / periods_per_year)
    mean = float(excess.mean())
    std = float(excess.std())
    scale = np.sqrt(periods_per_year)

    return {
        "sharpe": float(mean / std * scale) if std > 0 else float("nan"),
        "mean_period": mean,
        "std_period": std,
        "ann_return": mean * periods_per_year,
        "ann_vol": std * scale,
        "n": int(series.size),
    }

def data_profile_report(df):
    df = ProfileReport(df, title="Profiling Report", explorative=True)
    return df

def intraday_distribution(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Aggregate a column's mean/std by hour of day and day of week.

    Returns
    -------
    dict with keys: by_hour (DataFrame), by_day (DataFrame)
    """
    clean = df.copy()
    clean.index = pd.to_datetime(clean.index)
    clean["hour"] = clean.index.hour
    clean["day_of_week"] = clean.index.day_name()

    by_hour = clean.groupby("hour").agg(
        ret_mean=(col, "mean"),
        ret_std=(col, "std"),
        count=(col, "count"),
    )
    by_day = clean.groupby("day_of_week").agg(
        ret_mean=(col, "mean"),
        ret_std=(col, "std"),
        count=(col, "count"),
    )

    return {"by_hour": by_hour, "by_day": by_day}

# add volume features
def volume_features(
    df: pd.DataFrame,
    col: str = "volume",
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add rolling volume context: the rolling mean, volume relative to it, and a
    rolling z-score.

    Returns a new frame; `df` is not modified.
    """
    windows = windows or [20]
    out = df.copy()
    for window in windows:
        rolling = out[col].rolling(window)
        mean = rolling.mean()
        std = rolling.std()
        out[f"{col}_mean_{window}"] = mean
        out[f"{col}_rel_{window}"] = out[col] / mean
        out[f"{col}_z_{window}"] = (out[col] - mean) / std
    return out

# add time based features
def calculate_time_features(df, session_hour=1):
    df["hour"] = df.index.hour
    df["minutes_since_open"] = np.where(df.index.hour >= session_hour,
    (df.index.hour - session_hour) * 60 + df.index.minute,0)
    df["day_of_week"] = df.index.dayofweek
    return df

# fit ar model test
def ols_model(df: pd.DataFrame, col: str, lags: int):
    """
    Fit an AutoReg(`lags`) model with a constant to `df[col]`.

    Returns the fitted statsmodels results object (not a dict) so callers keep
    access to `.summary()`, `.params`, `.predict()` and the rest.
    """
    if AutoReg is None:
        raise ImportError("statsmodels is not installed")
    model = AutoReg(df[col].dropna(), lags=lags, trend="c")
    return model.fit()


# ── Visualizations ──────────────────────────────────────────────────────────

# plot distribution analsis with right or left skewed data confirmation

# Line series plot
def plot_line_series(df: pd.DataFrame, title: str, cols: list[str]):
    fig, ax = plt.subplots()
    ax.plot(df[cols])
    ax.set(
        xlabel="Date",
        ylabel = "Price",
        title = title,
    )

def plot_corr_matrix(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        corr = df[col].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        corr_map = sns.diverging_palette(160, 10, s=70, l=45, center="dark", as_cmap=True)
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=corr_map, ax=ax)
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
    return df

def plot_return_distribution(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        ax = df[col].hist(bins=80, figsize=(10, 4))
        ax.set_title(f"Return distribution: {col}")
        plt.tight_layout()
        plt.show()

def plot_holding_period_comp(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        table = df.pivot_table(index="Holding Period", columns=[col],values="return",aggfunc="mean")
        table.reset_index(inplace=True)
        ax = table.plot(kind="bar", stacked=True, figsize=(10, 7))
        ax.set_title(f"Holding Time Distribution: {col}")
        ax.set_ylabel("Return")
        ax.set_xlabel("Holding Period")
        plt.show()

# auto correlation
def pacf_plot(df, col, lags):
    if plot_pacf is None:
        raise ImportError("statsmodels is not installed")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_pacf(df[col].dropna(), lags=lags, ax=ax)
    ax.set_title("Partial Autocorrelation (PACF) — identify significant lags")
    plt.tight_layout()
    plt.show()


def ljung_box(
    series: pd.Series,
    lags: Optional[List[int]] = None,
    n_plot_lags: int = 60,
) -> Dict[str, Any]:
    """
    Formally test the joint null that a series has no autocorrelation.

    An ACF/PACF plot is not a test. Eyeballing 60 lags against a 5% band
    expects three exceedances under a true white-noise null, so "some bars
    poke out" and "there is structure" are not the same statement. Ljung-Box
    tests all lags up to k jointly, which is the claim actually being made.

    Also reports the same test on squared returns. Uncorrelated is not
    independent: a return series can pass in the mean and fail hard in the
    variance, which is the normal case for a commodity and matters for
    position sizing even when the mean has no memory.

    Parameters
    ----------
    series      : per-period returns (not levels, not cumulative)
    lags        : lag horizons to test jointly; defaults to [5, 10, 20]
    n_plot_lags : horizon for the band-exceedance count

    Returns
    -------
    dict with keys: n (int), lags (DataFrame of lb_stat/lb_pvalue for the
                    level series), squared (DataFrame, same for squared),
                    white_noise_5pct (bool), acf (ndarray), band (float),
                    n_outside_band (int), expected_outside_band (float)
    """
    if acorr_ljungbox is None or acf is None:
        raise ImportError("statsmodels is not installed")

    clean = series.dropna()
    if clean.empty:
        raise ValueError("series has no non-null values")
    lags = lags or [5, 10, 20]

    level = acorr_ljungbox(clean, lags=lags, return_df=True)
    squared = acorr_ljungbox(clean ** 2, lags=lags, return_df=True)

    n = int(clean.size)
    n_plot_lags = min(n_plot_lags, n // 2)
    correlations = acf(clean, nlags=n_plot_lags, fft=True)[1:]
    band = 1.96 / np.sqrt(n)

    return {
        "n": n,
        "lags": level,
        "squared": squared,
        "white_noise_5pct": bool((level["lb_pvalue"] > 0.05).all()),
        "acf": correlations,
        "band": float(band),
        "n_outside_band": int((np.abs(correlations) > band).sum()),
        "expected_outside_band": 0.05 * len(correlations),
    }


def print_ljung_box(series: pd.Series, label: str, **kwargs) -> Dict[str, Any]:
    """Run `ljung_box` and print it as one readable block. Returns the dict."""
    result = ljung_box(series, **kwargs)

    print(f"{label}  (n={result['n']:,})")
    table = pd.DataFrame({
        "lb_p_level": result["lags"]["lb_pvalue"],
        "lb_p_squared": result["squared"]["lb_pvalue"],
    })
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        f"  ACF bars outside +/-{result['band']:.4f}: "
        f"{result['n_outside_band']}/{len(result['acf'])} "
        f"(expected by chance: {result['expected_outside_band']:.0f})"
    )
    verdict = "no autocorrelation in the mean" if result["white_noise_5pct"] \
        else "AUTOCORRELATION PRESENT in the mean"
    print(f"  -> {verdict}")
    if not bool((result["squared"]["lb_pvalue"] > 0.05).all()):
        print("  -> squared returns ARE autocorrelated (volatility clustering)")
    return result

# ── Factor screening ───────────────────────────────────────────────────────────

def ic_rank_spearman(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Rank IC of each feature against one target column.

    Rank IC (Spearman) is used instead of plain correlation because it only
    cares about the ORDER of the predictions, which is what a signal is judged
    on, and it is not dragged around by outliers.

    Returns one row per feature with the IC, its p-value, and the sample size.
    """
    from scipy.stats import spearmanr

    rows = {}
    for feature in features:
        pair = df[[feature, target]].dropna()
        ic, p_value = spearmanr(pair[feature], pair[target])
        rows[feature] = {"ic": ic, "p_value": p_value, "n": len(pair)}

    return pd.DataFrame(rows).T


def nw_t_stat(df: pd.DataFrame, features: list[str], target: str, lags: int) -> pd.DataFrame:
    """Newey-West t-statistic for each feature's rank IC.

    Overlapping forward returns make the plain p-value far too optimistic: a
    10-day target re-uses 9 of its 10 days at every step, so the observations
    are not independent. Setting lags to the forecast horizon corrects for it.

    The slope of standardised ranks IS the Spearman IC, so this is the same
    number as ic_rank_spearman with an honest standard error attached.
    """
    rows = {}
    for feature in features:
        pair = df[[feature, target]].dropna()

        x = pair[feature].rank()
        y = pair[target].rank()
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        rows[feature] = {"t_nw": fit.tvalues.iloc[1], "p_nw": fit.pvalues.iloc[1], "n": len(pair)}

    return pd.DataFrame(rows).T


def ic_decay(df: pd.DataFrame, features: list[str], horizons: list[int],
             target_template: str = "target_{horizon}_d") -> pd.DataFrame:
    """Rank IC of each feature at several forward horizons.

    The target columns must already exist - build them with
    features.forward_horizons() first.

    Returns a table with one row per feature and one column per horizon.
    """
    columns = {}
    for horizon in horizons:
        target = target_template.format(horizon=horizon)
        result = ic_rank_spearman(df, features, target)
        columns[horizon] = result["ic"].astype(float)

    return pd.DataFrame(columns)


def plot_ic_decay(ic_table: pd.DataFrame, title: str = "IC decay by horizon"):
    """Line plot of the table returned by ic_decay().

    A real signal decays smoothly from a peak. A jagged line that jumps around
    zero is noise.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for feature in ic_table.index:
        ax.plot(ic_table.columns, ic_table.loc[feature], marker="o", label=feature)

    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xlabel("Forward horizon (trading days)")
    ax.set_ylabel("Rank IC")
    ax.set_title(title)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


def feature_correlation(df: pd.DataFrame, features: list[str],
                        method: str = "spearman") -> pd.DataFrame:
    """Correlation matrix of the feature set."""
    return df[features].corr(method=method)


def plot_feature_correlation(df: pd.DataFrame, features: list[str], method: str = "spearman",
                             title: str = "Feature correlation"):
    """Heatmap of feature_correlation(), with the upper triangle hidden.

    Anything above about 0.7 is close to a duplicate: it will split importance
    in a model and make the feature set look wider than it is.
    """
    corr = feature_correlation(df, features, method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    corr_map = sns.diverging_palette(160, 10, s=70, l=45, center="dark", as_cmap=True)
    sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, cmap=corr_map, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return corr


def plot_equity_curve(returns: pd.Series, title: str = "Equity curve"):
    """Compounded growth of capital, with the drawdown underneath, both in percent.

    `returns` must be a FRACTION of capital per trade, not raw P&L - use
    model.to_simple_returns() to convert first. Percentages are what make two
    different products comparable on the same axes.

    Pass the NON-OVERLAPPING sample, or the curve counts each trade several
    times and looks far smoother than the strategy really is.
    """
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax_equity.plot(equity.index, (equity.values - 1.0) * 100, linewidth=1.6)
    ax_equity.axhline(0, color="grey", linewidth=1)
    ax_equity.set_ylabel("Cumulative return (%)")
    ax_equity.set_title(title)

    ax_drawdown.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.6)
    ax_drawdown.set_ylabel("Drawdown (%)")
    ax_drawdown.set_xlabel("Date")

    plt.tight_layout()
    plt.show()

    return equity

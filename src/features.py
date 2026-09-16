"""Feature generation for the SAFEX factor pipeline.

Every function takes a DataFrame, adds columns to it, and returns it, so calls
can be stacked one after another in a notebook.

The price series these run on must be a CONTINUOUS log price, built by
build_log_returns() below. That matters because the panel is a stitched
front-contract series: differencing the raw price across a roll measures the
gap between two delivery months, not a market move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_log_returns(df: pd.DataFrame, contract_cols: list[str], spot_cols: list[str],
                      spread_cols: list[str]) -> pd.DataFrame:
    """Add log prices, per-expiry log returns, and a continuous log price.

    Three kinds of column get three different treatments:

      contract_cols - a single delivery month's price. Log return is taken
                      WITHIN expiry so no return ever spans a roll.
      spot_cols     - not contract specific (e.g. usdzar), so no grouping.
      spread_cols   - can cross zero, so differences instead of logs.

    For each contract column you get three new columns:
      log_<col>   the log price
      ret_<col>   the within-contract log return
      clog_<col>  the continuous (roll-adjusted) log price, which is the
                  cumulative sum of ret_<col>. Build features on this one.

    Spread columns get ret_<col> and c_<col>, the same idea without the logs.
    Build features on c_<col>, never on the raw spread: the raw level steps at
    every roll, and because rolls follow the calendar, a seasonal feature will
    happily "predict" those steps.
    """
    df = df.sort_index()

    for col in contract_cols:
        positive_price = df[col].where(df[col] > 0)
        df[f"log_{col}"] = np.log(positive_price)
        df[f"ret_{col}"] = df.groupby("expiry")[f"log_{col}"].diff()

        # Rebuild a continuous series from the returns. The starting level is
        # arbitrary, which is fine because every feature below uses differences.
        first_value = df[f"log_{col}"].dropna().iloc[0]
        running_total = df[f"ret_{col}"].fillna(0.0).cumsum() + first_value
        df[f"clog_{col}"] = running_total.where(df[f"log_{col}"].notna())

    for col in spot_cols:
        positive_price = df[col].where(df[col] > 0)
        df[f"log_{col}"] = np.log(positive_price)
        df[f"ret_{col}"] = df[f"log_{col}"].diff()
        df[f"clog_{col}"] = df[f"log_{col}"]

    for col in spread_cols:
        df[f"ret_{col}"] = df.groupby("expiry")[col].diff()

        # Same rebuild as the contract prices, but with no logs - a spread can
        # cross zero, so changes are plain differences. Use c_<col> as the level
        # to build features on; the raw column jumps at every roll.
        first_value = df[col].dropna().iloc[0]
        running_total = df[f"ret_{col}"].fillna(0.0).cumsum() + first_value
        df[f"c_{col}"] = running_total.where(df[col].notna())

    return df


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score using differences, so a series that crosses zero is fine."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def moving_avg_feat(df: pd.DataFrame, col: str, windows: list[int], z_window: int = 252) -> pd.DataFrame:
    """Distance from a moving average, z-scored.

    This measures how STRETCHED price is from its own mean. It is a level,
    not a direction - see ma_momentum_feat for direction.
    """
    for window in windows:
        moving_average = df[col].rolling(window).mean()
        df[f"{col}_ma_{window}"] = moving_average

        distance = df[col] - moving_average
        df[f"{col}_ma_dist_z_{window}"] = zscore(distance, z_window)

    return df


def ma_momentum_feat(df: pd.DataFrame, col: str, cross_pairs: list[tuple], above_windows: list[int],
                     macd_spans: tuple = (12, 26), z_window: int = 252) -> pd.DataFrame:
    """Direction and persistence of the trend, rather than distance from it.

    cross_pairs   - list of (fast, slow) window pairs, e.g. [(10, 50)]
    above_windows - share of the last N days spent above the N-day average
    macd_spans    - (fast, slow) spans for the EMA difference
    """
    for fast, slow in cross_pairs:
        fast_average = df[col].rolling(fast).mean()
        slow_average = df[col].rolling(slow).mean()
        df[f"{col}_macross_{fast}_{slow}"] = zscore(fast_average - slow_average, z_window)

    for window in above_windows:
        is_above = df[col] > df[col].rolling(window).mean()
        df[f"{col}_above_ma_{window}"] = is_above.rolling(window).mean()

    fast_span, slow_span = macd_spans
    macd = df[col].ewm(span=fast_span).mean() - df[col].ewm(span=slow_span).mean()
    df[f"{col}_macd"] = zscore(macd, z_window)

    return df


def seasonality_feat(df: pd.DataFrame) -> pd.DataFrame:
    """Day of year as a sine/cosine pair, so December and January sit next to each other."""
    day_of_year = df.index.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def vol_regime_feat(df: pd.DataFrame, col: str, windows: list[int],
                    regime_window: int = 252) -> pd.DataFrame:
    """Realised volatility over several windows, plus where the last one sits historically.

    vol_regime is a percentile rank, so 0.9 means today is more volatile than
    90 percent of the trailing window.
    """
    returns = df[col].diff()

    for window in windows:
        df[f"vol_{window}"] = returns.rolling(window).std()

    last_window = windows[-1]
    df["vol_regime"] = df[f"vol_{last_window}"].rolling(regime_window).rank(pct=True)

    return df


def volume_feat(df: pd.DataFrame, col: str, ratio_window: int = 20, short_window: int = 5,
                long_window: int = 20) -> pd.DataFrame:
    """Volume against its own recent average, computed within each contract.

    Volume is grouped by expiry because a contract's volume ramps up while it is
    the front month and falls away into delivery - that lifecycle is contract
    specific, so it must not be carried across a roll.

    Keep long_window well under the typical contract length or most rows come
    back empty. The median contract here is about 98 rows.
    """
    grouped = df.groupby("expiry")[col]

    df["volume_ratio"] = df[col] / grouped.transform(lambda s: s.rolling(ratio_window).mean())

    short_average = grouped.transform(lambda s: s.rolling(short_window).mean())
    long_average = grouped.transform(lambda s: s.rolling(long_window).mean())
    df["volume_trend"] = short_average / long_average

    return df


def days_to_harvest_feat(df: pd.DataFrame, harvest_month: int = 4) -> pd.DataFrame:
    """Calendar days until the next harvest window (roughly April for SA maize).

    Dates already past this year's harvest point at next year's.
    """
    year = df.index.year + (df.index.month > harvest_month).astype(int)
    next_harvest = pd.to_datetime(dict(year=year, month=harvest_month, day=1))

    days = (next_harvest.values - df.index.values) / np.timedelta64(1, "D")
    df["days_to_harvest"] = days

    return df


def forward_horizons(df: pd.DataFrame, col: str, horizons: list[int],
                     vol_window: int = 20) -> pd.DataFrame:
    """Forward returns, scaled by expected volatility, one column per horizon.

    `col` must be a LOG PRICE LEVEL (a clog_ column), not a return series -
    the forward return is a difference of two points on that level.

    Dividing by expected vol puts every horizon on the same scale, so a 1-day
    and a 10-day target are comparable.
    """
    returns = df[col].diff()

    for horizon in horizons:
        forward_return = df[col].shift(-horizon) - df[col]
        expected_vol = returns.rolling(vol_window).std() * np.sqrt(horizon)
        df[f"target_{horizon}_d"] = forward_return / expected_vol

    return df

# Hedged beta spread

def rolling_hedge_ratio_naive(df, base_col, quote_col, window=120, rebalance_days=5):
    """Estimate beta by rolling OLS, then freeze it between rebalances.

    A beta of 1 assumes SAFEX and CBOT move one-for-one. They don't -- freight,
    local basis and FX pass-through mean the true ratio drifts. Freezing between
    rebalances stops the hedge jittering on daily noise.
    """
    base = df[base_col]
    quote = df[quote_col]

    # Rolling OLS slope: cov(base, quote) / var(base).
    covariance = quote.rolling(window).cov(base)
    variance = base.rolling(window).var()
    beta_raw = covariance / variance

    # Freeze: only let beta update every `rebalance_days` bars.
    beta_frozen = beta_raw.copy()
    beta_frozen[:] = np.nan
    beta_frozen.iloc[::rebalance_days] = beta_raw.iloc[::rebalance_days]
    return beta_frozen.ffill()

def hedged_spread(df, base_col, quote_col, window=120, rebalance_days=5):
    """Continuous hedged spread built from returns, not level differences.

    Differencing the level across a beta change measures the revaluation from
    re-hedging, not a market move -- the same trap as differencing across a
    roll. So take the hedged return each bar using the beta already in force,
    then sum back up into a continuous series.
    """
    base_ret = df[base_col].diff()
    quote_ret = df[quote_col].diff()

    # Estimate on returns: regressing log levels on log levels is spurious.
    covariance = quote_ret.rolling(window).cov(base_ret)
    variance = base_ret.rolling(window).var()
    beta_raw = covariance / variance

    # Freeze between rebalances.
    beta_frozen = pd.Series(np.nan, index=df.index)
    beta_frozen.iloc[::rebalance_days] = beta_raw.iloc[::rebalance_days]
    beta_frozen = beta_frozen.ffill()

    # Apply the beta that was already in force at the start of the bar, so the
    # hedge ratio is never set with knowledge of the bar it prices.
    hedged_return = quote_ret - beta_frozen.shift(1) * base_ret

    return beta_frozen, hedged_return.cumsum()

def ou_half_life(series, window=60):
    """Half-life of mean reversion, from an AR(1) fit on the rolling window.

    Regress the change on the lagged level: delta_t = a + b * level_{t-1}.
    A negative b means reversion; half-life = -ln(2)/ln(1+b). Short half-life
    means reversion trades make sense, long or undefined means they don't.
    """
    def fit(x):
        level = x[:-1]
        delta = np.diff(x)
        if np.var(level) == 0:
            return np.nan
        b = np.cov(level, delta)[0, 1] / np.var(level)
        if b >= 0 or (1 + b) <= 0:
            return np.nan          # not reverting over this window
        return -np.log(2) / np.log(1 + b)

    return series.rolling(window).apply(fit, raw=True)


def variance_ratio(series, window=60, lag=5):
    """Variance of k-period returns divided by k times variance of 1-period.

    Above 1 means trending (moves compound), below 1 means mean-reverting
    (moves offset), around 1 means random walk.
    """
    returns_1 = series.diff()
    returns_k = series.diff(lag)
    var_1 = returns_1.rolling(window).var()
    var_k = returns_k.rolling(window).var()
    return var_k / (lag * var_1)


def cross_leg_correlation(df, col_a, col_b, window=60):
    """Rolling correlation between the two legs' returns.

    A direct measure of whether the arb relationship is holding. Falling
    correlation is the decoupling regime where fading the spread is dangerous.
    """
    return df[col_a].diff().rolling(window).corr(df[col_b].diff())


def relative_volatility(df, col_a, col_b, window=60):
    """Which leg is driving. Ratio of the two legs' realised vol."""
    vol_a = df[col_a].diff().rolling(window).std()
    vol_b = df[col_b].diff().rolling(window).std()
    return vol_a / vol_b






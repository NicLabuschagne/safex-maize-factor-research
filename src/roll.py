"""
Roll spine and liquid series construction.

Two objects come out of here, and keeping them separate is the point:

  * roll spine   - which contract is held on each date
  * liquid series - that contract's arb LEVEL (never spliced) alongside a
                    same-contract CHANGE (so a roll never creates a move)

The level stays contract-specific because each delivery month references a
different physical window and therefore a different parity band. Splicing
levels across a roll injects a median jump of about USD 6/t and up to USD 66/t,
against a typical daily move of USD 1.6/t, so it is never done here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Outright price legs that get a same-contract log return. These are LEVELS of
# a single delivery month, so differencing them across a roll measures the
# calendar spread between two contracts rather than a market move.
RETURN_LEGS = ("safex_zar_per_tonne", "safex_usd_per_tonne", "cbot_usd_per_tonne")

# SAFEX maize delivery months that carry meaningful volume. March, July and
# December account for roughly 86% of traded volume; May and September have
# open interest an order of magnitude lower and their marks are more often
# exchange-derived than traded.
DEFAULT_ELIGIBLE_MONTHS = (3, 5, 7, 9, 12)


def build_roll_spine(
    safex: pd.DataFrame,
    contract: str = "YMAZ",
    eligible_months: tuple[int, ...] = DEFAULT_ELIGIBLE_MONTHS,
    exit_day_of_prior_month: int = 20,
    enforce_monotonic: bool = True,
) -> pd.DataFrame:
    """
    Decide which contract is held on each trade date.

    Selection is by highest open interest among contracts still inside the
    trading window, using open interest LAGGED ONE DAY. The lag matters:
    JSE open interest disagrees with third-party feeds on roughly 23% of
    days in a way consistent with a publication lag, so selecting on
    same-day open interest risks a mild look-ahead.

    exit_day_of_prior_month encodes the prop-desk constraint that a position
    cannot be held into the delivery month. A July contract must be closed by
    20 June, so it stops being eligible on that date.

    enforce_monotonic prevents rolling backwards to an earlier expiry. Without
    it, open interest crossings produce 9 backward switches over 2010-2026,
    each injecting a calendar spread jump and then reversing it a few weeks
    later. With it, 45 switches and none backwards.

    Returns one row per date: dt, expiry, expiry_month, safex_open_interest.
    """
    frame = safex[
        (safex["contract"] == contract)
        & (safex["expiry_month"].isin(eligible_months))
    ].copy()

    frame["dt"] = pd.to_datetime(frame["trade_date"])
    frame["expiry_dt"] = pd.to_datetime(frame["expiry"])

    # Last date the position may still be open.
    frame["exit_date"] = (
        frame["expiry_dt"] - pd.DateOffset(months=1)
    ) + pd.Timedelta(days=exit_day_of_prior_month - 1)

    frame = frame.sort_values(["expiry", "dt"])
    frame["open_interest_lag"] = frame.groupby("expiry")["safex_open_interest"].shift(1)

    eligible = frame[
        (frame["dt"] <= frame["exit_date"]) & frame["open_interest_lag"].notna()
    ]

    if not enforce_monotonic:
        chosen = eligible.sort_values("open_interest_lag").groupby("dt").tail(1)
        return _spine_columns(chosen)

    held_expiry: str | None = None
    selected_rows = []

    for _, day in eligible.groupby("dt", sort=True):
        # If the held contract is no longer eligible we are forced to move.
        if held_expiry is not None and (day["expiry"] == held_expiry).sum() == 0:
            held_expiry = None

        # Never roll backwards: candidates must expire on or after what we hold.
        if held_expiry is None:
            candidates = day
        else:
            candidates = day[day["expiry_dt"] >= pd.to_datetime(held_expiry)]
            if len(candidates) == 0:
                candidates = day

        pick = candidates.sort_values("open_interest_lag").iloc[-1]
        held_expiry = pick["expiry"]
        selected_rows.append(pick)

    chosen = pd.DataFrame(selected_rows)
    return _spine_columns(chosen)


def _spine_columns(chosen: pd.DataFrame) -> pd.DataFrame:
    columns = ["dt", "expiry", "expiry_month", "safex_open_interest"]
    return chosen[columns].sort_values("dt").reset_index(drop=True)


def build_liquid_series(
    spine: pd.DataFrame,
    arb_panel: pd.DataFrame,
    spread_column: str,
) -> pd.DataFrame:
    """
    Attach the held contract's arb level and its same-contract daily change.

    The change is differenced WITHIN each contract before the join, so a roll
    can never produce a change. On a roll date the change would belong to a
    position that was not held the previous day, so it is blanked. That costs
    45 observations out of 4,334 and removes every splice artifact.

    The same treatment is applied to the outright legs, as
    "<leg>_log_return". Those columns exist because the raw levels are the
    obvious thing to reach for in a momentum study and the obvious thing to do
    with them - np.log(level).diff() down the date index - is wrong. It spans
    the roll, so the 44 roll dates in YMAZ become fabricated returns with 5.6x
    the standard deviation of a normal day, carrying 26% of the total sum of
    squares and supplying all ten of the largest "moves" in the sample.

    The LEVEL is still deliberately left raw and contract-specific, and is
    still never spliced. Convert it to a comparable state variable later using
    a band estimated for that delivery month, rather than trying to make one
    continuous level series.
    """
    panel = arb_panel.copy()
    panel["dt"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["expiry", "dt"])

    panel["arb_change"] = panel.groupby("expiry")[spread_column].diff()
    if f"{spread_column}_log" in panel.columns:
        panel["log_arb_change"] = panel.groupby("expiry")[f"{spread_column}_log"].diff()

    # Same-contract log returns of each outright leg, grouped by expiry for
    # exactly the reason arb_change is: a roll must not create a return.
    return_columns = []
    for leg in RETURN_LEGS:
        if leg in panel.columns:
            name = f"{leg}_log_return"
            panel[name] = np.log(panel[leg]).groupby(panel["expiry"]).diff()
            return_columns.append(name)

    # Carry the log columns through when the panel has them, so the liquid
    # series supports both the additive and multiplicative specifications.
    optional = [c for c in ("log_safex_usd", "log_safex_zar", "log_cbot",
                            f"{spread_column}_log")
                if c in panel.columns]

    series = spine.merge(
        panel[
            ["dt", "expiry", spread_column, "arb_change",
             "safex_volume", "cbot_volume", "staleness_hours",
             "safex_zar_per_tonne", "safex_usd_per_tonne", "cbot_usd_per_tonne"]
            + optional
            + return_columns
            + (["log_arb_change"] if "log_arb_change" in panel.columns else [])
        ],
        on=["dt", "expiry"],
        how="inner",
    ).sort_values("dt")

    series = series.rename(columns={spread_column: "arb_level"})
    series["is_roll"] = series["expiry"] != series["expiry"].shift(1)

    # A roll date's "change" would compare two different delivery months.
    blank = ["arb_change"] + return_columns
    if "log_arb_change" in series.columns:
        blank.append("log_arb_change")
    series.loc[series["is_roll"], blank] = np.nan

    return series.reset_index(drop=True)


def add_within_contract_lags(
    panel: pd.DataFrame,
    spread_column: str,
    lags: int = 1,
) -> pd.DataFrame:
    """
    Add lagged levels and first differences taken WITHIN each contract.

    This is the single most important line in the whole statistical setup.
    Any unit-root, error-correction or threshold model must use these rather
    than lags of a rolled series, because a rolled series contains structural
    breaks at every roll and those bias unit-root tests toward failing to
    reject. Grouping by expiry means a roll is simply not represented.
    """
    out = panel.copy()
    out["dt"] = pd.to_datetime(out["trade_date"])
    out = out.sort_values(["expiry", "dt"])

    for lag in range(1, lags + 1):
        out[f"{spread_column}_lag{lag}"] = out.groupby("expiry")[spread_column].shift(lag)

    out[f"{spread_column}_diff"] = (
        out[spread_column] - out[f"{spread_column}_lag1"]
    )
    return out
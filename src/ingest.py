"""
Ingest for the SAFEX / CBOT arbitrage spread study.

Three sources, three loaders, one aligned panel:

  1. CBOT corn      - Databento GLBX.MDP3 ohlcv-1h, ZC.FUT parent symbology
  2. SAFEX maize    - JSE "Physical Settled Grain contracts" per-expiry-year workbooks
  3. USDZAR         - hourly FX bars

Everything is snapped at 10:00 UTC, which is 12:00 SAST, the SAFEX close.
That makes all three legs contemporaneous and removes the non-synchronous
trading bias that would otherwise manufacture mean reversion in the spread.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# The hour (UTC) at which every leg is sampled. 10:00 UTC == 12:00 SAST,
# which is when the JSE agricultural market closes.
SAFEX_CLOSE_HOUR_UTC = 10

# CBOT corn delivery months. The five liquid ones are the only outrights
# we care about; corn also lists other months but they carry no volume.
CORN_MONTH_CODES = {"H": 3, "K": 5, "N": 7, "U": 9, "Z": 12}

# A Globex outright looks like "ZCZ5": root, month code, SINGLE year digit.
# Anything else in a parent-symbology pull is a calendar spread ("ZCZ0-ZCN1")
# or a butterfly ("ZC:BF H0-K0-N0") and must be excluded.
CBOT_OUTRIGHT_PATTERN = re.compile(r"^ZC([HKNUZ])(\d)$")

# Conversion from US cents per bushel to USD per tonne for corn.
BUSHELS_PER_TONNE_CORN = 39.3683

# SAFEX contract codes we keep. Wheat (WEAT) is loaded but filtered later,
# so the decision stays reversible without re-reading the workbooks.
SAFEX_CONTRACTS = ("WMAZ", "YMAZ", "WEAT")

# Output column name for each SAFEX-vs-CBOT pair. Keeping the pair explicit
# in the name stops a generic "spread" column being confused with a calendar
# spread once several of these sit in the same panel.
ARB_SPREAD_NAMES = {
    "YMAZ": "ym_corn_arb",
    "WMAZ": "wm_corn_arb",
    "WEAT": "wheat_arb",
}

# SAFEX months with meaningful volume. The board lists monthly expiries but
# open interest concentrates almost entirely in these five.
SAFEX_LIQUID_MONTHS = (3, 5, 7, 9, 12)


# --------------------------------------------------------------------------
# 1. CBOT corn
# --------------------------------------------------------------------------

def _resolve_cbot_expiry(
    month_code: str,
    year_digit: int,
    last_observation: pd.Timestamp,
) -> tuple[int, int]:
    """
    Resolve a single-digit Globex year into a real calendar year.

    CME encodes the year with one digit, so "ZCZ5" is ambiguous between
    December 2015 and December 2025. We disambiguate using the contract's
    own last observed bar: the true expiry must be the earliest candidate
    year whose delivery month is not before that last observation.

    Example: ZCN1 last trades 2011-07-14 -> July 2011, not July 2021.
             ZCH8 still trading on 2026-09-09 -> March 2028, not March 2018.
    """
    expiry_month = CORN_MONTH_CODES[month_code]

    candidate_years = [decade + year_digit for decade in range(2000, 2070, 10)]
    for year in sorted(candidate_years):
        # Compare at month granularity; a contract's last bar falls in or
        # just before its delivery month.
        candidate = pd.Timestamp(year=year, month=expiry_month, day=1, tz="UTC")
        last_month_start = last_observation.normalize().replace(day=1)
        if candidate >= last_month_start:
            return year, expiry_month

    raise ValueError(f"Could not resolve expiry for {month_code}{year_digit}")


def load_cbot_corn(
    dbn_path: str | Path,
    max_staleness_hours: int = 30,
) -> pd.DataFrame:
    """
    Read a Databento ohlcv-1h DBN file and return outright corn bars snapped
    at the SAFEX close, one row per (trade_date, contract).

    max_staleness_hours caps how old the last CBOT print may be relative to
    the SAFEX close. 30 hours allows a normal overnight gap plus a weekend
    edge, while excluding holiday stretches where the price is genuinely stale.

    Returns columns:
        trade_date, raw_symbol, expiry, cbot_usd_per_tonne, volume,
        staleness_hours
    """
    import databento as db

    store = db.DBNStore.from_file(str(dbn_path))
    bars = store.to_df(price_type="float").reset_index()

    # Drop spreads and butterflies. In a ZC.FUT parent pull these are roughly
    # two thirds of all rows, and their prices are spread differentials
    # (often negative), which would be nonsense in a level calculation.
    is_outright = bars["symbol"].str.match(CBOT_OUTRIGHT_PATTERN)
    bars = bars[is_outright].copy()

    # instrument_id is unique per real contract, so it separates the
    # 2015 and 2025 uses of the same "ZCZ5" text symbol.
    last_seen = bars.groupby("instrument_id")["ts_event"].max()

    expiry_lookup: dict[int, str] = {}
    for instrument_id, last_observation in last_seen.items():
        symbol = bars.loc[bars["instrument_id"] == instrument_id, "symbol"].iloc[0]
        month_code, year_digit = symbol[2], int(symbol[3])
        year, month = _resolve_cbot_expiry(month_code, year_digit, last_observation)
        expiry_lookup[instrument_id] = f"{year}-{month:02d}"

    bars["expiry"] = bars["instrument_id"].map(expiry_lookup)

    return _snap_to_safex_close(bars, max_staleness_hours=max_staleness_hours)


def _snap_to_safex_close(
    bars: pd.DataFrame,
    max_staleness_hours: int,
) -> pd.DataFrame:
    """
    Take the last CBOT print at or before 10:00 UTC on each date.

    A plain hour filter is not enough. CBOT grain electronic hours changed
    over the sample: in 2010-2012 the 10:00 UTC hour prints on only 45-58%
    of trading dates, rising to 88% in 2013 and 98%+ from 2014. Filtering on
    the hour alone would silently drop half the early sample.

    Forward-filling is the economically correct fix, not a patch. If corn did
    not trade between the previous close and 10:00 UTC, then the last print
    IS the price a SAFEX trader could see at 12:00 SAST. We record how stale
    that print is so the assumption can be tested rather than trusted.
    """
    bars = bars.copy()
    bars["date"] = bars["ts_event"].dt.normalize()

    # Cut-off is the END of the 10:00 UTC bar, i.e. 11:00 UTC.
    cutoff = bars["date"] + pd.Timedelta(hours=SAFEX_CLOSE_HOUR_UTC + 1)
    eligible = bars[bars["ts_event"] < cutoff].copy()

    # For each contract-date, the last print at or before the cut-off.
    eligible = eligible.sort_values("ts_event")
    latest = eligible.groupby(["instrument_id", "date"]).tail(1).copy()

    # Carry the last known print forward across dates where nothing traded
    # before the cut-off at all.
    frames = []
    for instrument_id, group in latest.groupby("instrument_id", sort=False):
        group = group.set_index("date").sort_index()
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="D", tz="UTC")
        reindexed = group.reindex(full_range)
        reindexed["instrument_id"] = instrument_id
        reindexed[["symbol", "expiry"]] = reindexed[["symbol", "expiry"]].ffill()
        reindexed["last_print_ts"] = reindexed["ts_event"].ffill()
        reindexed["close"] = reindexed["close"].ffill()
        frames.append(reindexed.reset_index(names="date"))

    snapped = pd.concat(frames, ignore_index=True).dropna(subset=["close"])

    # Hours between the last actual print and the SAFEX close on this date.
    reference = snapped["date"] + pd.Timedelta(hours=SAFEX_CLOSE_HOUR_UTC)
    snapped["staleness_hours"] = (
        (reference - snapped["last_print_ts"]).dt.total_seconds() / 3600.0
    )
    snapped = snapped[snapped["staleness_hours"] <= max_staleness_hours]

    snapped["trade_date"] = snapped["date"].dt.date

    # CBOT quotes corn in US cents per bushel.
    snapped["cbot_usd_per_tonne"] = (
        snapped["close"] / 100.0 * BUSHELS_PER_TONNE_CORN
    )

    result = snapped[
        ["trade_date", "symbol", "expiry", "cbot_usd_per_tonne",
         "volume", "staleness_hours"]
    ].rename(columns={"symbol": "raw_symbol", "volume": "cbot_volume"})

    return result.sort_values(["expiry", "trade_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. SAFEX
# --------------------------------------------------------------------------

def load_safex_workbooks(folder: str | Path) -> pd.DataFrame:
    """
    Walk a folder of JSE per-expiry-year workbooks and return a tidy panel.

    Handles two known schema quirks:

      * Column drift. The 2010 workbook has 18 columns; the 2026 workbook has
        21, adding FirstTradePrice, ContractSize and OI_in_Rand. We select a
        common core and fill the rest with NA.

      * "Open" is not an opening price. It equals the previous day's Close on
        every row tested, so it is a carry-forward and is dropped entirely.

    Also nulls the Low/High sentinel zeros that appear on no-trade days.
    """
    folder = Path(folder)
    workbooks = sorted(folder.glob("*.xlsx")) + sorted(folder.glob("*.xls"))
    if not workbooks:
        raise FileNotFoundError(f"No JSE workbooks found in {folder}")

    core_columns = [
        "TradeDate", "ExpiryDate", "Expiry", "ShortName",
        "Low", "High", "Close", "Change", "Volume", "OI",
    ]

    frames = []
    for workbook in workbooks:
        sheet = pd.read_excel(workbook, sheet_name="PricingDetail")

        missing = [column for column in core_columns if column not in sheet.columns]
        for column in missing:
            sheet[column] = pd.NA

        sheet = sheet[core_columns].copy()
        sheet["source_file"] = workbook.name
        frames.append(sheet)

    panel = pd.concat(frames, ignore_index=True)

    # Sentinel zeros on days with no trades. Close is still a valid exchange
    # mark on those days, so only Low/High are nulled.
    panel.loc[panel["Low"] == 0, "Low"] = pd.NA
    panel.loc[panel["High"] == 0, "High"] = pd.NA

    panel = panel[panel["ShortName"].isin(SAFEX_CONTRACTS)].copy()
    panel["expiry_month"] = pd.to_datetime(panel["ExpiryDate"]).dt.month
    panel["trade_date"] = pd.to_datetime(panel["TradeDate"]).dt.date
    panel["expiry"] = pd.to_datetime(panel["ExpiryDate"]).dt.strftime("%Y-%m")

    # The same (contract, expiry, date) can appear in two workbooks where
    # coverage overlaps; keep one.
    panel = panel.drop_duplicates(subset=["trade_date", "ShortName", "expiry"])

    renamed = panel.rename(
        columns={"ShortName": "contract", "Close": "safex_zar_per_tonne",
                 "Volume": "safex_volume", "OI": "safex_open_interest"}
    )

    keep = [
        "trade_date", "contract", "expiry", "expiry_month",
        "safex_zar_per_tonne", "safex_volume", "safex_open_interest",
        "source_file",
    ]
    return renamed[keep].sort_values(
        ["contract", "expiry", "trade_date"]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. FX
# --------------------------------------------------------------------------

def load_usdzar(parquet_path: str | Path) -> pd.DataFrame:
    """
    Read hourly USDZAR bars and snap to the SAFEX close hour.

    Sanity-checks the quote direction: USDZAR should be rand per dollar,
    so values sit roughly between 6 and 25 over 2010-2026. If the feed ever
    returns dollars per rand the arb would silently invert, so we fail loud.
    """
    fx = pd.read_parquet(parquet_path)
    fx["ts"] = pd.to_datetime(fx["ts"], utc=True)

    median_rate = fx["close"].median()
    if not 5.0 < median_rate < 30.0:
        raise ValueError(
            f"USDZAR median is {median_rate:.4f}, which is not rand per dollar. "
            "Check the quote direction before proceeding."
        )

    snapped = fx[fx["ts"].dt.hour == SAFEX_CLOSE_HOUR_UTC].copy()
    snapped["trade_date"] = snapped["ts"].dt.date

    return (
        snapped[["trade_date", "close"]]
        .rename(columns={"close": "usdzar"})
        .drop_duplicates(subset=["trade_date"])
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# 4. Join
# --------------------------------------------------------------------------

def build_arb_spread(
    safex: pd.DataFrame,
    cbot: pd.DataFrame,
    fx: pd.DataFrame,
    safex_contract: str = "YMAZ",
) -> pd.DataFrame:
    """
    Join the three legs into the USD arbitrage spread.

    Contracts are paired on delivery month, so a SAFEX July contract is
    matched to the CBOT July contract of the same year. Both legs therefore
    roll together and no calendar spread leaks into the arb.

    The output column is named for the pair, e.g. ym_corn_arb for
    SAFEX yellow maize against CBOT corn:

        ym_corn_arb = safex_zar / usdzar - cbot_usd_per_tonne
    """
    safex_leg = safex[
        (safex["contract"] == safex_contract)
        & (safex["expiry_month"].isin(SAFEX_LIQUID_MONTHS))
    ].copy()

    panel = safex_leg.merge(cbot, on=["trade_date", "expiry"], how="inner")
    panel = panel.merge(fx, on="trade_date", how="inner")

    panel["safex_usd_per_tonne"] = panel["safex_zar_per_tonne"] / panel["usdzar"]

    # Logs require strictly positive legs. Both are outright prices so this
    # should never bind, but a bad FX print or a zero mark would produce -inf
    # and poison every downstream test silently.
    positive = (panel["safex_usd_per_tonne"] > 0) & (panel["cbot_usd_per_tonne"] > 0)
    if not positive.all():
        dropped = int((~positive).sum())
        print(f"warning: dropping {dropped} rows with a non-positive leg")
        panel = panel[positive].copy()
    spread_column = ARB_SPREAD_NAMES[safex_contract]

    # Additive spread, in USD per tonne. Closest to the economics: most of
    # the parity cost stack (freight, port handling, inland transport) is a
    # dollar amount per tonne that does not scale with the corn price.
    panel[spread_column] = (
        panel["safex_usd_per_tonne"] - panel["cbot_usd_per_tonne"]
    )

    # Log legs and the log spread, which is log(SAFEX / CBOT). This models a
    # constant RATIO rather than a constant dollar wedge. Better behaved
    # statistically, because dollar variance is not constant across a corn
    # price that ranges from roughly 130 to 320 USD/t over the sample.
    #
    # Neither specification is obviously right. The additive one matches a
    # fixed cost stack; the multiplicative one matches the components that do
    # scale (insurance, financing, the rand-denominated share, the variable
    # import tariff). Build both and let the tests say which holds.
    #
    # Note: log_arb and the additive spread are NOT monotonic transforms of
    # each other, since the additive spread goes negative in surplus years
    # while the log spread does not. A threshold found in one does not map
    # to the other.
    panel["log_safex_usd"] = np.log(panel["safex_usd_per_tonne"])
    panel["log_safex_zar"] = np.log(panel["safex_zar_per_tonne"])
    panel["log_cbot"] = np.log(panel["cbot_usd_per_tonne"])
    panel[f"{spread_column}_log"] = panel["log_safex_usd"] - panel["log_cbot"]

    panel.attrs["spread_column"] = spread_column
    panel.attrs["log_spread_column"] = f"{spread_column}_log"

    return panel.sort_values(["expiry", "trade_date"]).reset_index(drop=True)
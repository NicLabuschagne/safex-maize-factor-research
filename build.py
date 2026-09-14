"""
Build the analysis panels from raw sources.

Run once from the terminal:

    python build.py

Notebooks should read the parquets in data/processed/ and never call the
loaders directly. The CBOT parse decompresses 1.7m rows and takes minutes,
so it is cached; delete data/processed/cbot_corn_snapped.parquet whenever
ingest.py changes or you will keep reading a stale cache.
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.append("src")

import ingest
import roll

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DATA = Path("data")
RAW_SAFEX = DATA / "jse"
RAW_CBOT = DATA / "cbot" / "GLBX-20260910-DGNDVSD8E6/glbx-mdp3-20100607-20260909.ohlcv-1h.dbn.zst"
RAW_FX = DATA / "fx" / "fx_USD_ZAR_1h.parquet"

PROCESSED = DATA / "processed"
CBOT_CACHE = PROCESSED / "cbot_corn_snapped.parquet"

# SAFEX contract -> the arb column it produces.
CONTRACTS = {
    "YMAZ": "ym_corn_arb",
    "WMAZ": "wm_corn_arb",
}


def get_cbot_corn() -> pd.DataFrame:
    """Parse the Databento DBN once, then read from parquet on later runs."""
    if CBOT_CACHE.exists():
        print(f"cbot: reading cache {CBOT_CACHE}")
        return pd.read_parquet(CBOT_CACHE)

    print("cbot: parsing DBN, this takes a few minutes")
    corn = ingest.load_cbot_corn(RAW_CBOT)
    corn.to_parquet(CBOT_CACHE)
    return corn


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    safex = ingest.load_safex_workbooks(RAW_SAFEX)
    print(f"safex: {len(safex):,} rows, "
          f"{safex.trade_date.min()} to {safex.trade_date.max()}")

    cbot = get_cbot_corn()
    print(f"cbot:  {len(cbot):,} rows, {cbot.expiry.nunique()} contracts")

    usdzar = ingest.load_usdzar(RAW_FX)
    print(f"fx:    {len(usdzar):,} daily snaps")

    for safex_contract, arb_column in CONTRACTS.items():
        # Stacked per-contract panel. Every expiry that traded on every date.
        # This is the object the statistics run on.
        arb_panel = ingest.build_arb_spread(safex, cbot, usdzar, safex_contract)
        arb_panel.to_parquet(PROCESSED / f"{arb_column}_panel.parquet")

        # Which contract is held each day, and that contract's level and
        # same-contract change. This is the object for charts and for the
        # trade as actually held.
        spine = roll.build_roll_spine(safex, contract=safex_contract)
        liquid = roll.build_liquid_series(spine, arb_panel, arb_column)
        liquid.to_parquet(PROCESSED / f"{arb_column}_liquid.parquet")

        print(
            f"{arb_column}: panel {len(arb_panel):,} rows across "
            f"{arb_panel.expiry.nunique()} expiries | "
            f"liquid {len(liquid):,} rows, {int(liquid.is_roll.sum())} rolls, "
            f"median OI held {int(liquid.safex_open_interest.median()):,}"
        )


if __name__ == "__main__":
    main()
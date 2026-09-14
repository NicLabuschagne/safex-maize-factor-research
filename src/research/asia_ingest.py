"""data preprocessing module to extract price series from akshare and output different frame types

"""
### Libraries
import numpy as np
import pandas as pd
import akshare as ak


# Build soy complex OHLCV data extraction from AkShare
# Symbology
soy_complex_symbols = {
    "A0": "bean1",
    "B0": "bean2",
    "M0": "meal",
    "Y0": "oil"
}

# Column Names
OHLCV_FIELDS = ["open", "high", "low", "close", "volume", "hold", "settle"]

# Function to fetch Sina OHLCV series into DF
def fetch_sina_ohlcv(symbols: dict[str, str]) -> pd.DataFrame:
    """ Fetch daily OHLCV for each Sina continuous contract and stack them long format.
    returns a frame with columns: date, symbol, open, high, low, close, volume, hold, settle"""
    frames = []
    for sym, name in symbols.items():
        raw_df = ak.futures_zh_daily_sina(symbol=sym)
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df = raw_df.assign(symbol=sym, name=name)
        frames.append(raw_df[["date", "symbol", "name"] + OHLCV_FIELDS])
    return pd.concat(frames, ignore_index=True).sort_values(["name", "date"])

def to_ohlcv_panel(df: pd.DataFrame, flat_df: True) -> pd.DataFrame:
    """Pivot the long df to date x name, filed - MultiIndex columns"""
    df = df.pivot(index="date", columns="name", values=OHLCV_FIELDS)
    df.columns = df.columns.swaplevel(0, 1)
    df.columns.names = ["name", "field"]
    # Manipulate df into flat close columns per product
    if flat_df:
    df = df.xs("close", level="field", axis=1)
    return df.sort_index(axis=1)

def flattened_df(df: pd.DataFrame, name, field) -> pd.DataFrame:
    """ translates df into a flattened structure with columns being product close and volume """
    idx = pd.IndexSlice
    sub = df.loc[:, idx[:, [name, field]]]
    flat = sub.copy()
    flat.columns = [f"{name}_{field}" for name, field in flat.colums]
    ts = flat.dropna()
    return ts

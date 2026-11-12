"""Data pipeline: raw files in data/ -> data/dataset.parquet (ML-ready).

Output: one row per hour, feature columns + target columns (forward log-returns
for 1h, 2h and 5h horizons). Every feature uses only information available at
or before time t. No splitting or modeling here.
Usage: python build_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HORIZON = 5  # hours ahead, kept for backward compatibility of the 5h target name

SPOT_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
             "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def load_binance(folder: Path, pattern: str, header: bool,
                 extra: bool = False) -> pd.DataFrame:
    """Concatenate all monthly Binance klines zips into one continuous UTC frame.
    extra=True also keeps quote_volume, count and taker_buy_volume."""
    dfs = []
    for f in sorted(folder.glob(pattern)):
        df = pd.read_csv(f, compression="zip", header=0 if header else None,
                         names=None if header else SPOT_COLS)
        if header:
            df = df.rename(columns=dict(zip(df.columns, SPOT_COLS)))
        # Binance files from 2025 onward use microseconds; older ones use milliseconds
        df["open_time"] = pd.to_datetime(df["open_time"],
                                         unit="us" if df["open_time"].iloc[0] > 10**14 else "ms",
                                         utc=True)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["time"] = out["open_time"]
    out = out.set_index("time").sort_index()
    out = out[~out.index.duplicated(keep="first")]
    cols = ["open", "high", "low", "close", "volume"]
    if extra:
        cols += ["quote_volume", "count", "taker_buy_volume"]
    return out[cols].astype(float)


def load_funding() -> pd.Series:
    dfs = []
    for f in sorted((DATA / "futures/um").glob("BTCUSDT-fundingRate-*.zip")):
        dfs.append(pd.read_csv(f, compression="zip"))
    out = pd.concat(dfs, ignore_index=True)
    out["time"] = pd.to_datetime(out["calc_time"], unit="ms", utc=True)
    return out.set_index("time")["last_funding_rate"].astype(float).sort_index()


def load_metrics() -> pd.DataFrame:
    """OI + long/short ratios. One file per day, contents are 5-minute snapshots
    with duplicated rows."""
    dfs = []
    for f in sorted((DATA / "metrics").glob("BTCUSDT-metrics-*.zip")):
        df = pd.read_csv(f, compression="zip").drop_duplicates(
            subset=["create_time"], keep="last")
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["time"] = pd.to_datetime(out["create_time"], utc=True)
    out = out.set_index("time").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[["sum_open_interest", "count_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]].astype(float)


def load_yahoo(name: str, interval: str) -> pd.Series:
    """Yahoo CSV has 3 junk header rows; first column is the date."""
    df = pd.read_csv(DATA / f"yahoo_{name}_{interval}.csv", skiprows=3, header=None,
                     names=["date", "close", "high", "low", "open", "volume"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    s = df.set_index("date")["close"].astype(float)
    return s[s > 0]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build() -> pd.DataFrame:
    print("[1/5] Loading spot 1m...")
    spot = load_binance(DATA / "spot", "BTCUSDT-1m-*.zip", header=False, extra=True)
    print(f"  {len(spot):,} 1m rows ({spot.index[0]} -> {spot.index[-1]})")
    sh = spot.resample("1h").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"), count=("count", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"))
    h = sh[["open", "high", "low", "close", "volume"]].copy()
    print(f"[2/5] Resampled to 1h: {len(h):,} candles")

    print("[3/5] Joining futures, funding, metrics, yahoo...")
    fut = load_binance(DATA / "futures/um", "BTCUSDT-1h-*.zip", header=True, extra=True)
    h["fut_volume"] = fut["volume"].reindex(h.index)
    h["basis"] = fut["close"].reindex(h.index) / h["close"] - 1  # futures premium/discount

    # hourly microstructure features (bar t is closed at t, safe for a target at t+5)
    h["spot_taker_ratio"] = sh["taker_buy_volume"] / sh["volume"]
    h["fut_taker_ratio"] = (fut["taker_buy_volume"] / fut["volume"]).reindex(h.index)
    h["trade_count"] = fut["count"].reindex(h.index)
    h["spot_fut_vol_ratio"] = h["volume"] / h["fut_volume"]

    met = load_metrics().resample("1h").last()
    for c, name in [("sum_open_interest", "oi"),
                    ("count_toptrader_long_short_ratio", "ls_toptrader"),
                    ("count_long_short_ratio", "ls_account"),
                    ("sum_taker_long_short_vol_ratio", "ls_taker_vol")]:
        h[name] = met[c].reindex(h.index)
    h.loc[h["oi"] <= 0, "oi"] = np.nan
    h["oi_change_24h"] = h["oi"] / h["oi"].shift(24) - 1
    h = h.replace([np.inf, -np.inf], np.nan)

    fund = load_funding()
    h["funding"] = fund.reindex(h.index, method="ffill")

    # daily macro: shift(1) so day-D values only enter after day D is closed (anti-leakage)
    for name in ["spx", "dxy", "gold", "eth"]:
        d = load_yahoo(name, "1d")
        ret = np.log(d / d.shift(1)).shift(1)
        h[f"ret_{name}_1d"] = ret.reindex(h.index, method="ffill")
        try:  # 1h only covers ~2022+; include whatever exists
            yh = load_yahoo(name, "1h").resample("1h").last()  # SPX bars start at 9:30, not on the hour
            h[f"ret_{name}_1h"] = np.log(yh / yh.shift(1)).reindex(h.index)
        except FileNotFoundError:
            pass

    print("[4/5] Technical features + targets...")
    ret1 = np.log(h["close"] / h["close"].shift(1))
    h["ret_1h"] = ret1
    h["ret_4h"] = np.log(h["close"] / h["close"].shift(4))
    h["ret_24h"] = np.log(h["close"] / h["close"].shift(24))
    h["rsi_14"] = rsi(h["close"])
    ma, sd = h["close"].rolling(20).mean(), h["close"].rolling(20).std()
    h["bb_pos"] = (h["close"] - ma) / (2 * sd)          # position inside Bollinger band [-1, 1]
    h["bb_width"] = 4 * sd / ma
    h["realvol_24h"] = ret1.rolling(24).std()
    h["vol_ratio"] = h["volume"] / h["volume"].rolling(24).mean()
    for hh in (1, 2, HORIZON):
        h[f"target_{hh}h"] = np.log(h["close"].shift(-hh) / h["close"])

    h = h.dropna(subset=["close"])

    print("[5/5] Sanity checks...")
    gaps = h.index.to_series().diff().dt.total_seconds().div(3600)
    n_gap = int((gaps > 1).sum())
    print(f"  Binance downtime gaps (missing hours): {n_gap} events - expected (2019 maintenance/hack)")
    # anti-leakage: target at t must equal log(close[t+5]/close[t])
    t = h.index[1000]
    assert np.isclose(h.loc[t, "target_5h"], np.log(h.loc[h.index[1005], "close"] / h.loc[t, "close"]))
    feat = [c for c in h.columns if not c.startswith("target_")]
    assert all(h[f"target_{hh}h"].isna().sum() >= hh for hh in (1, 2, HORIZON)), \
        "target NaNs must only come from the last horizon plus missing hours"

    out = DATA / "dataset.parquet"
    h.reset_index().to_parquet(out)
    print(f"OK: {out}  {h.shape[0]:,} rows x {h.shape[1]} cols "
          f"({h.index[0].date()} -> {h.index[-1].date()})")
    print(f"features: {len(feat)} cols, targets: 1h/2h/5h forward log-returns")
    print(h[feat + ["target_1h", "target_2h", "target_5h"]].describe().loc[["mean", "std"]].T.to_string())
    return h


if __name__ == "__main__":
    build()

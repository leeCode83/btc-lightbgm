"""Download free BTC datasets from Binance Vision and Yahoo Finance.

All URLs verified (HTTP 200/206) via curl:
  - spot monthly klines     : https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip
  - futures monthly klines  : https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-YYYY-MM.zip
  - futures monthly funding : https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-YYYY-MM.zip
  - futures DAILY metrics files (OI, long-short ratio; no monthly variant, starts 2020-09-01).
    Note: one file per day, but the contents are 5-minute snapshots, not daily aggregates.
      https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip
  - yfinance: ^GSPC, DX-Y.NYB, GC=F, ETH-USD (1h capped at ~730 days by Yahoo, so we
    also fetch 1d for full coverage)

Resume-friendly: existing files with size > 0 are skipped.
Usage: python download_data.py [--start 2019-01] [--skip-metrics]
"""

import argparse
import calendar
import sys
import time
from pathlib import Path

import requests

BASE = "https://data.binance.vision/data"
OUT = Path(__file__).resolve().parent.parent / "data"

SPOT_KLINES_START = (2019, 1)      # BTCUSDT spot 1m starts here
FUT_KLINES_START = (2019, 9)       # um futures 1h nominal start (earlier months are 404)
FUT_FUNDING_START = (2020, 1)      # um funding rate starts here
METRICS_START = (2020, 9, 1)       # metrics only as daily files, starts 2020-09-01

session = requests.Session()


def download(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False  # already downloaded, skip (resume)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=120, stream=True)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                return True
            if r.status_code == 404:
                return False  # that month simply has no data yet
            print(f"  HTTP {r.status_code}: {url}")
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}: {e}")
        time.sleep(2 * (attempt + 1))
    print(f"  FAILED: {url}")
    return False


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def binance_monthly(kind: str, symbol: str, interval: str, start, end) -> None:
    folder = OUT / kind
    n = 0
    for y, m in months(start, end):
        # funding rate files are fetched with interval="" and carry the datatype in the filename
        datatype = "fundingRate" if not interval else "klines"
        fname = f"{symbol}-{interval or datatype}-{y}-{m:02d}.zip"
        parts = [BASE, kind, "monthly", datatype, symbol]
        if interval:
            parts.append(interval)
        url = "/".join(parts) + "/" + fname
        if download(url, folder / fname):
            n += 1
            print(f"  {fname}")
    print(f"  -> {n} new files ({kind})")


def binance_daily_metrics(symbol: str, start, end) -> None:
    # metrics only exist as daily files; ~12KB each. Skip with --skip-metrics.
    folder = OUT / "metrics"
    from datetime import date, timedelta
    cur = date(*start)
    endd = date(*end)
    n = 0
    while cur <= endd:
        fname = f"{symbol}-metrics-{cur.isoformat()}.zip"
        url = f"{BASE}/futures/um/daily/metrics/{symbol}/{fname}"
        if download(url, folder / fname):
            n += 1
        cur += timedelta(days=1)
    print(f"  -> {n} new files (metrics daily)")


def yahoo(symbols: dict, period: str = "max", interval: str = "1h") -> None:
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed: pip install yfinance - skipping Yahoo.")
        return
    for sym, name in symbols.items():
        df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  empty: {sym}")
            continue
        df.to_csv(OUT / f"yahoo_{name}_{interval}.csv")
        print(f"  yahoo_{name}_{interval}.csv  ({len(df)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01", help="YYYY-MM, start of spot/futures klines")
    ap.add_argument("--end", default="2026-11", help="YYYY-MM, end of data")
    ap.add_argument("--skip-metrics", action="store_true", help="skip daily OI metrics (many files)")
    ap.add_argument("--skip-yahoo", action="store_true")
    args = ap.parse_args()

    sy, sm = map(int, args.start.split("-"))
    ey, em = map(int, args.end.split("-"))
    end_month = (ey, em)
    last_day = (ey, em, calendar.monthrange(ey, em)[1])

    print("[1/4] Spot klines 1m (BTCUSDT) - resample to 1h later")
    binance_monthly("spot", "BTCUSDT", "1m", (sy, sm), end_month)

    print("[2/4] Futures klines 1h (BTCUSDT)")
    binance_monthly("futures/um", "BTCUSDT", "1h", max((sy, sm), FUT_KLINES_START), end_month)

    print("[3/4] Funding rate (BTCUSDT)")
    binance_monthly("futures/um", "BTCUSDT", "", max((sy, sm), FUT_FUNDING_START), end_month)

    if not args.skip_metrics:
        print("[4/4] Daily metrics (OI, long/short ratio) - starts 2020-09, many files")
        binance_daily_metrics("BTCUSDT", max((sy, sm, 1), METRICS_START), last_day)
    else:
        print("[4/4] Metrics skipped (--skip-metrics)")

    if not args.skip_yahoo:
        symbols = {
            "^GSPC": "spx",
            "DX-Y.NYB": "dxy",
            "GC=F": "gold",
            "ETH-USD": "eth",
        }
        print("[+] Yahoo Finance 1h (capped at ~730 days by Yahoo)")
        yahoo(symbols, interval="1h")
        print("[+] Yahoo Finance 1d (full coverage, supplement)")
        yahoo(symbols, interval="1d")

    print("Done. Data in:", OUT)

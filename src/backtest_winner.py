"""Walk-forward with the winning config (num_leaves=31, lr=0.05, min_data=100),
then a cost-aware backtest on the high-conviction cell (90th percentile of |pred|),
with per-trade return statistics: min, max, mean, mode.

Mode note: per-trade returns are continuous, so the mode is reported as the most
frequent 5-bps bin rather than a single value.
Usage: python backtest_winner.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.parquet"
TARGET = "target_5h"
COST = 0.001
THR_Q = 0.90
PARAMS = dict(objective="regression", num_leaves=31, learning_rate=0.05,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, n_estimators=500, verbose=-1, random_state=42)


def folds(index):
    starts = pd.date_range(index[0], index[-1], freq=pd.Timedelta("182D"))
    for ts in starts:
        if ts - index[0] < pd.Timedelta("730D"):
            continue
        te = index[index.get_indexer([ts], method="bfill")[0]]
        te_end = index[index.get_indexer([te + pd.Timedelta("182D")], method="bfill")[0]]
        yield te, te_end


def main() -> None:
    df = pd.read_parquet(DATA)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").loc["2020-01-01":].dropna(subset=[TARGET])
    feat = [c for c in df.columns if not c.startswith("target_")]

    trades = []
    for i, (te, te_end) in enumerate(folds(df.index)):
        tr, tst = df.loc[:te - pd.Timedelta("1D")], df.loc[te:te_end]
        m = lgb.LGBMRegressor(**PARAMS).fit(tr[feat], tr[TARGET])
        pred = m.predict(tst[feat])
        fd = pd.DataFrame({"pred": pred, "actual": tst[TARGET].values})
        thr = fd["pred"].abs().quantile(THR_Q)
        sel = fd[fd["pred"].abs() > thr]
        trades.append(sel.assign(fold=i))
        print(f"fold {i}: test {te.date()} -> {te_end.date()} | trades {len(sel)}")

    t = pd.concat(trades)
    t["ret_net"] = np.sign(t["pred"]) * t["actual"] - COST
    total = t["ret_net"].sum()
    print(f"\n=== RETURN PER TRADE (net, {len(t)} trades, fee 0.1%) ===")
    print(f"min     : {t['ret_net'].min():+.4%}")
    print(f"max     : {t['ret_net'].max():+.4%}")
    print(f"mean    : {t['ret_net'].mean():+.4%}")
    hit = (t['ret_net'] > 0).mean()
    print(f"hit rate: {hit:.1%}")
    print(f"total   : {total:+.4f} ({total * 100:+.1f}%)")
    # mode: continuous returns -> report the most frequent 5-bps bin
    bins = (t["ret_net"] / 0.0005).round().astype(int)
    mode_bin = bins.mode().iloc[0]
    print(f"mode    : bin [{mode_bin * 0.05:+.2f}%, {(mode_bin + 1) * 0.05:+.2f}%) "
          f"({(bins == mode_bin).mean():.1%} of trades)")
    q = t["ret_net"].quantile([.1, .5, .9])
    print(f"p10 {q.loc[.1]:+.3%} | median {q.loc[.5]:+.3%} | p90 {q.loc[.9]:+.3%}")


if __name__ == "__main__":
    main()

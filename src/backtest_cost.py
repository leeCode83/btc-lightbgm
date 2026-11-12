"""Simple cost-aware backtest over the out-of-sample predictions.

Strategy: when |pred| exceeds a per-fold quantile threshold of |pred|, take a
sign(pred) position for one horizon. PnL = sign * actual. Round-trip fee 0.1%
per trade.
Usage: python backtest_cost.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COST = 0.001  # round trip
THRESH_Q = [0.0, 0.5, 0.7, 0.8, 0.9]


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "predictions_oof.csv", parse_dates=["time"])
    for tgt, g in df.groupby("target"):
        print(f"\n===== {tgt} (round-trip fee {COST:.1%}) =====")
        print(f"{'thr_q':>6} | {'trades':>7} | {'hit':>5} | {'gross/trade':>11} | "
              f"{'net/trade':>10} | {'total net':>9} | {'t-stat':>6}")
        d = g.copy()
        d["day"] = d["time"].dt.date
        for q in THRESH_Q:
            tr_all = []
            for _, fd in d.groupby("fold"):
                thr = fd["pred"].abs().quantile(q)
                sel = fd[fd["pred"].abs() > thr]
                if len(sel):
                    pnl = np.sign(sel["pred"]) * sel["actual"] - COST
                    tr_all.append(pd.DataFrame({"day": sel["day"], "pnl": pnl}))
            t = pd.concat(tr_all)
            n = len(t)
            daily = t.groupby("day")["pnl"].sum()
            tstat = daily.mean() / (daily.std() / np.sqrt(len(daily))) if len(daily) > 2 and daily.std() > 0 else 0.0
            gross = t["pnl"].mean() + COST if n else np.nan
            print(f"{q:>6.1f} | {n:>7} | {(t['pnl'] > 0).mean():>5.1%} | "
                  f"{gross:>11.5f} | {t['pnl'].mean():>10.5f} | {t['pnl'].sum():>+9.4f} | {tstat:>+6.2f}")


if __name__ == "__main__":
    main()

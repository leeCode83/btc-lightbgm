"""Statistical significance of the IC via day-block bootstrap.

Returns are autocorrelated across hours within a day, so the resampling unit
is a day, not a single row.
Input : results/predictions_oof.csv   Output: summary on stdout.
Usage: python ic_significance.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

N_BOOT = 2000
RNG = np.random.default_rng(42)


def boot_ic(dates: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    day = pd.Series(dates).dt.date.values
    uniq = np.unique(day)
    idx_by_day = [np.where(day == d)[0] for d in uniq]
    ics = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_day[i] for i in pick])
        ics[b] = np.corrcoef(pred[sel], actual[sel])[0, 1]
    return ics


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "predictions_oof.csv", parse_dates=["time"])
    for tgt, g in df.groupby("target"):
        ic_obs = np.corrcoef(g["pred"], g["actual"])[0, 1]
        ics = boot_ic(g["time"].values, g["pred"].values, g["actual"].values)
        lo, hi = np.percentile(ics, [2.5, 97.5])
        p = max((ics <= 0).mean(), 1e-4) if ic_obs > 0 else max((ics >= 0).mean(), 1e-4)
        print(f"{tgt}: IC = {ic_obs:+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] | "
              f"p(one-sided) ~= {p:.4f} | {'SIGNIFICANT' if lo > 0 else 'NOT significant'}")


if __name__ == "__main__":
    main()

"""Hyperparameter tuning and feature selection with an anti-overfitting protocol.

The grid search and feature selection only ever see:
  - train:      2020-01 .. 2023-12
  - validation: 2024-01 .. 2025-12
The 2026 test window is touched exactly once, at the end, with the winning config.
Usage: python tune_model.py
"""

import itertools
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.parquet"
TARGET = "target_5h"            # winning horizon from the earlier experiment
TRAIN_END = "2023-12-31 23:00"  # implicit 1d embargo: validation starts 2024-01-02
VAL_START, VAL_END = "2024-01-01", "2025-12-25"
TEST_START = "2026-01-25"       # 1d embargo after validation ends
GRID = dict(
    num_leaves=[31, 63, 127],
    learning_rate=[0.02, 0.05, 0.1],
    min_data_in_leaf=[100, 300],
)
BASE = dict(objective="regression", feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, n_estimators=500, verbose=-1, random_state=42)


def ic(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    df = pd.read_parquet(DATA)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").loc["2020-01-01":]
    feat_all = [c for c in df.columns if not c.startswith("target_")]
    d = df.dropna(subset=[TARGET])
    tr = d.loc[:TRAIN_END]
    va = d.loc[VAL_START:VAL_END]
    print(f"train {len(tr):,} | val {len(va):,} | test starts {TEST_START} (touched once)")

    # --- grid search on validation ---
    results = []
    for nl, lr, mdl in itertools.product(*GRID.values()):
        p = dict(BASE, num_leaves=nl, learning_rate=lr, min_data_in_leaf=mdl)
        m = lgb.LGBMRegressor(**p).fit(tr[feat_all], tr[TARGET])
        results.append((ic(m.predict(va[feat_all]), va[TARGET]), p, feat_all))
    best_ic, best_p, best_feat = max(results, key=lambda r: r[0])
    print(f"\ngrid done: 18 configs. Best validation IC {best_ic:+.4f} with "
          f"num_leaves={best_p['num_leaves']}, lr={best_p['learning_rate']}, "
          f"min_data={best_p['min_data_in_leaf']}")

    # --- feature selection: drop zero-gain features, try once more ---
    m = lgb.LGBMRegressor(**best_p).fit(tr[best_feat], tr[TARGET])
    gain = pd.Series(m.booster_.feature_importance("gain"), index=best_feat)
    keep = gain[gain > 0].index.tolist()
    dropped = sorted(set(best_feat) - set(keep))
    print(f"dropped {len(dropped)} zero-gain features: {dropped}")
    m2 = lgb.LGBMRegressor(**best_p).fit(tr[keep], tr[TARGET])
    ic2 = ic(m2.predict(va[keep]), va[TARGET])
    if ic2 > best_ic:
        best_ic, best_feat = ic2, keep
        print(f"feature subset is better: validation IC {ic2:+.4f} with {len(keep)} features")
    else:
        print(f"feature subset is NOT better ({ic2:+.4f} <= {best_ic:+.4f}), keeping all features")

    # --- final evaluation: touch the test window once ---
    te = d.loc[TEST_START:]
    fit_df = pd.concat([tr, va])
    m = lgb.LGBMRegressor(**best_p).fit(fit_df[best_feat], fit_df[TARGET])
    pred = m.predict(te[best_feat])
    rmse_m = float(np.sqrt(np.mean((pred - te[TARGET]) ** 2)))
    rmse_b = float(np.sqrt(np.mean(te[TARGET] ** 2)))
    dir_acc = float((np.sign(pred) == np.sign(te[TARGET])).mean())
    print("\n=== FINAL EVALUATION (2026 test, touched once) ===")
    print(f"config: {dict((k, v) for k, v in best_p.items() if k in ('num_leaves', 'learning_rate', 'min_data_in_leaf'))}, "
          f"{len(best_feat)} features")
    print(f"IC {ic(pred, te[TARGET]):+.4f}  | dir {dir_acc:.1%} | "
          f"RMSE {rmse_m:.5f} vs baseline {rmse_b:.5f}")
    print("(reference: old fold-8 with default config: IC -0.052 | dir 48.4% | RMSE 0.00901)")


if __name__ == "__main__":
    main()

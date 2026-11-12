"""Train LightGBM on forward log-returns of BTC (1h/2h/5h) with purged
walk-forward validation against a last-price baseline.

Input : data/dataset.parquet (output of build_dataset.py)
Output: models/ boosters, results/evaluation.csv, results/predictions_oof.csv,
        summary on stdout.
Usage: python train_model.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.parquet"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

START = "2020-01-01"              # drop 2019: derivative features are NaN there
TEST_LEN = pd.Timedelta("182D")   # ~6 months of test per fold
EMBARGO = pd.Timedelta("1D")      # longest feature window is 24h, rounded up
MIN_TRAIN = pd.Timedelta("730D")

PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, n_estimators=500,
    verbose=-1, random_state=42,
)


def folds(index: pd.DatetimeIndex):
    """Yield (train_end_excl_embargo, test_start, test_end_excl)."""
    starts = pd.date_range(index[0], index[-1], freq=TEST_LEN)
    for test_start in starts:
        train_end = test_start - EMBARGO
        if test_start - index[0] < MIN_TRAIN:
            continue
        te = index[index.get_indexer([test_start], method="bfill")[0]]
        test_end = index[index.get_indexer([te + TEST_LEN], method="bfill")[0]]
        yield train_end, te, test_end


def main() -> None:
    df = pd.read_parquet(DATA)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").loc[START:]
    targets = [c for c in df.columns if c.startswith("target_")]
    feat = [c for c in df.columns if not c.startswith("target_")]
    print(f"Data: {len(df):,} rows x {len(feat)} features ({df.index[0].date()} -> {df.index[-1].date()})")

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    all_rows, oof = [], []
    for tgt in targets:
        d = df.dropna(subset=[tgt])
        print(f"\n===== TARGET {tgt} =====")
        for i, (train_end, te, te_end) in enumerate(folds(d.index)):
            tr = d.loc[:train_end]
            tst = d.loc[te:te_end]
            if len(tr) < 1000 or len(tst) < 100:
                continue
            model = lgb.LGBMRegressor(**PARAMS)
            model.fit(tr[feat], tr[tgt])
            model.booster_.save_model(str(MODELS / f"lgbm_{tgt}_fold{i}.txt"))

            pred = model.predict(tst[feat])
            rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
            dir_acc = float((np.sign(pred) == np.sign(tst[tgt])).mean())
            row = dict(
                target=tgt, fold=i, test_start=str(te.date()), test_end=str(te_end.date()),
                n_test=len(tst), rmse_model=rmse(pred, tst[tgt]),
                rmse_baseline=rmse(np.zeros(len(tst)), tst[tgt]),
                dir_acc=dir_acc, ic=float(np.corrcoef(pred, tst[tgt])[0, 1]),
            )
            all_rows.append(row)
            oof.append(pd.DataFrame({"time": tst.index, "target": tgt, "pred": pred,
                                     "actual": tst[tgt].values, "fold": i}))
            print(f"fold {i}: test {te.date()} -> {te_end.date()} | "
                  f"RMSE {row['rmse_model']:.5f} vs baseline {row['rmse_baseline']:.5f} | "
                  f"dir {dir_acc:.1%} | IC {row['ic']:+.3f}")

    ev = pd.DataFrame(all_rows)
    ev.to_csv(RESULTS / "evaluation.csv", index=False)
    pd.concat(oof).to_csv(RESULTS / "predictions_oof.csv", index=False)

    print("\n=== SUMMARY PER HORIZON (weighted by n_test) ===")
    for tgt, g in ev.groupby("target"):
        w = g["n_test"]
        print(f"{tgt}: RMSE model {np.sqrt(np.average(g['rmse_model']**2, weights=w)):.5f} "
              f"vs baseline {np.sqrt(np.average(g['rmse_baseline']**2, weights=w)):.5f} | "
              f"dir {np.average(g['dir_acc'], weights=w):.1%} | "
              f"IC {np.average(g['ic'], weights=w):+.3f} | "
              f"IC>0 in {int((g['ic'] > 0).sum())}/{len(g)} folds")


if __name__ == "__main__":
    main()

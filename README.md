# BTC 5-Hour Price Prediction with LightGBM

A regression project that predicts the 5-hour forward log-return of BTC, validated
with purged walk-forward splits (no leakage) and evaluated with realistic
transaction costs. The design decisions follow the literature review in
[docs/research-brief.md](docs/research-brief.md).

## Conclusion

**The model finds a statistically significant signal, but no edge that reliably
survives transaction costs.**

- Information coefficient (IC) of +0.03 to +0.05 at the 5-hour horizon, significant
  under a day-block bootstrap (p ~= 0.001)
- Directional accuracy around 51%, inside the 50.9 to 56% range reported in the literature
- After a 0.1% round-trip fee, the net PnL per trade is near zero, and the one
  net-positive cell found in the sweep did not replicate under a second config
- This matches the brief's expectation and does not violate the efficient market
  hypothesis (Jaquart et al. 2021; Sobreiro et al. 2026)

## Reproducing (in order)

```bash
pip install -q requests yfinance lightgbm scikit-learn pyarrow
python src/download_data.py      # all raw data (~230 MB, resumable)
python src/build_dataset.py      # -> data/dataset.parquet (68,605 rows x 36 cols)
python src/train_model.py        # walk-forward over 3 horizons + baseline
python src/ic_significance.py    # bootstrap significance of the IC
python src/backtest_cost.py      # cost-aware threshold sweep (default config)
python src/tune_model.py         # 18-config grid on validation, test touched once
python src/backtest_winner.py    # backtest of the winning config + per-trade stats
```

## Data (all free)

| Source | Contents | Range | Notes |
|---|---|---|---|
| Binance Vision | Spot 1m (94 zips) | 2019-01 to 2026-10 | target + technical features; resampled to 1h |
| Binance Vision | Futures 1h (82 zips) | 2020-01 to 2026-10 | taker ratio, basis, trade count |
| Binance Vision | Funding rate (82 zips) | 2020-01 to 2026-10 | proxy for crowded positioning |
| Binance Vision | Metrics (2,262 zips) | 2020-09 to 2026-10 | files are per day but contain 5-minute snapshots: OI + 3 long/short ratios |
| Yahoo Finance | SPX, DXY, Gold, ETH (1h + 1d) | 1d covers full history | 1h is capped at ~730 days by Yahoo |
| not used | On-chain, sentiment | | needs paid APIs, skipped |

## Pipeline (`src/build_dataset.py`)

- 4.1 million 1m rows become 68,605 hourly candles (2019-01 to 2026-10)
- 33 features: technical (RSI, Bollinger position/width, realized vol, return lags,
  volume ratio), derivatives (basis, funding, spot and futures taker ratio, trade
  count, spot/futures volume ratio), metrics (OI, 24h OI change, 3 long/short
  ratios), macro (SPX/DXY/Gold/ETH returns at 1d and 1h)
- Targets: `log(close[t+h]/close[t])` for h in {1, 2, 5} hours
- Leakage control: daily macro shifted by 1 day; features only use data up to t;
  a self-check verifies the target formula on a random row
- One gotcha: Binance files from 2025 onward use microsecond timestamps while
  older files use milliseconds

## Validation

- Purged walk-forward: about 6 months of test per fold (9 folds, 2022 to 2026),
  train is everything before the test, with a 1-day embargo (longest feature
  window is 24h). Shuffled k-fold is deliberately not used; the brief notes it
  can produce an IC of +0.32 on pure noise
- Required baseline: predict zero (the last-price random walk), which is the RMSE
  minimizer on a near-efficient market
- Tuning protocol: the 18-config grid and feature selection only ever see
  validation (2024 to 2025); the 2026 test window is touched once at the end

## Results per experiment

| Experiment | Result |
|---|---|
| Baseline run (24 features, 5h) | IC -0.004, dir 50.8%, no signal |
| Added 9 derivative/metrics features | IC +0.033, dir 51.3%, OI and long/short enter the top-10 importances |
| Multi-horizon | 1h +0.028, 2h +0.024, 5h +0.033 (best and most stable) |
| Bootstrap IC, 5h pooled | +0.041, 95% CI [+0.016, +0.065], p ~= 0.0015, significant |
| Cost-aware sweep (default config) | only the 5h + 90th-percentile cell was net positive (+0.048%/trade, t = 1.24) |
| Tuning + final evaluation | 2026 test IC improved from -0.052 to +0.017 (more stable across regimes); yahoo-1h features had zero gain |
| Cost-aware (winning config) | mean -0.052%/trade. The positive cell from the default config did not replicate, so it was probably luck |

Per-trade statistics on the winning config (3,808 trades, 0.1% fee):
min -8.2%, max +11.1%, mean -0.05%, hit rate 47.3%.

## Methodological lessons

1. RMSE always loses to the "predict zero" baseline on return forecasting. That is
   not a model failure; IC is the fair skill metric.
2. A significant IC does not mean a profitable strategy. A 0.1% fee erases an
   edge of roughly 0.04% per trade.
3. Claims of 82%+ accuracy in the literature are a leakage red flag. Honest
   evaluation lands at 50 to 56%.
4. A backtest cell that does not replicate across configs is luck, not edge.
5. Derivatives microstructure features (OI, long/short ratios) add real value on
   top of technicals alone.

## Layout

```
src/download_data.py     # resumable downloader (spot/futures/funding/metrics/yahoo)
src/build_dataset.py     # raw files -> dataset.parquet (features + targets, no leakage)
src/train_model.py       # purged walk-forward over 3 horizons, LightGBM vs baseline
src/ic_significance.py   # day-block bootstrap of the IC
src/backtest_cost.py     # cost-aware threshold sweep
src/tune_model.py        # grid + feature selection on validation only
src/backtest_winner.py   # backtest of the winning config + per-trade stats
docs/research-brief.md   # literature review that motivated the design
data/                    # raw zips, yahoo CSVs, dataset.parquet (gitignored)
models/, results/        # per-fold boosters, evaluation.csv, predictions_oof.csv (gitignored)
```

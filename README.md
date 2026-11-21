# 🔋 Battery Arbitrage on the GB Electricity Market (RP3)

How much could a grid-scale battery (33.46 MWh, about 15 MW) earn by buying electricity
when it is cheap and selling it when it is expensive? This project compares three ways of
choosing when to trade, using real half-hourly GB wholesale prices from September 2016 to
January 2025:

| Strategy | How it picks the half-hours to trade |
|---|---|
| ⏰ **Time based** | Buys 01:00–04:59 and sells 17:00–20:59 every day |
| 🎯 **Price based** | Buys the 8 cheapest and sells the 8 dearest half-hours of each day, from the *real* prices (perfect foresight, so an upper benchmark) |
| 🧠 **ANN based** | The same, but picks the half-hours from a neural-network forecast of the next day's prices |

This is the third-year research project (RP3, 2025) that came before my fourth-year project (RP4). It
is deliberately simple: one battery model, three strategies, and one feed-forward network.

---

## 📊 Results

From `python run_backtests.py` (GBP millions, before any fixes from [Known issues](#%EF%B8%8F-known-issues)):

| Strategy | Profit 19 Sep 2016 → 20 Jan 2025 | Profit 19 Jun 2023 → 20 Jan 2025 | Full cycles |
|---|---:|---:|---:|
| Time based | 9.795 | 1.770 | 3045 |
| Price based (perfect foresight) | 13.445 | 2.432 | 2626 |
| ANN based | – | 2.059 | 574 |

The ANN only has forecasts for its test period (the last 20 % of the data, from May 2023), so
the three are compared from 19 June 2023. The report used an earlier ANN training run
(`results/ann_forecast_2025-03-01.csv`), which gives **2.055**. Run
`python run_backtests.py --forecast results/ann_forecast_2025-03-01.csv` to reproduce it exactly.

ANN test-set error: **RMSE 25.75, MAE 18.58 GBP/MWh**. That is measured against prices whose
outliers were replaced by 73; against the real prices it is RMSE 32.60, MAE 19.33 (see issue 4).

<p align="center">
  <img src="images/Rolling Cashflow Over Time for Predicted Price.png" width="48%">
  <img src="images/Total Profit per Year for Predicted Price.png" width="48%">
</p>

---

## 📥 Data sources

Both datasets are public and come from **Elexon's Balancing Mechanism Reporting Service (BMRS)**,
the official source of GB electricity market data. No API key is needed.

| | Prices and volumes | Wind and solar forecasts |
|---|---|---|
| **BMRS dataset** | `MID`: Market Index Data | `DGWS`: Day-ahead generation forecast for wind and solar |
| **What it is** | Half-hourly short-term wholesale price (GBP/MWh) and traded volume (MWh) | NESO's forecast of offshore wind, onshore wind and solar output for each half-hour, published about 17:45 UK time the day before |
| **Web page** | <https://bmrs.elexon.co.uk/market-index-prices> | <https://bmrs.elexon.co.uk/day-ahead-generation-forecast-wind-and-solar> |
| **API endpoint** | `https://data.elexon.co.uk/bmrs/api/v1/datasets/MID` | `https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind-and-solar/day-ahead` |
| **Raw file here** | `data/raw/market_index_data_raw.csv` (made by `download_data.py`) | `data/raw/DayAheadGenerationForecastWindAndSolar.csv` (BMRS website CSV export, Feb 2025) |
| **Processed file** | `data/processed/prices_2016_2025.csv` | `data/processed/{wind_offshore,wind_onshore,solar}_da_forecast_2016_2025.csv` |

* **Period:** 2016-09-19 00:00 to 2025-01-20 00:00 UTC, every half-hour (146,161 rows).
* **Market index providers:** two exchanges report the MID. Only APX (`APXMIDP`) has prices;
  N2EX (`N2EXMIDP`) reports zero for almost every period, so it is dropped.
* **API documentation:** <https://bmrs.elexon.co.uk/api-documentation>. Each request can
  cover at most 7 days, so `download_data.py` fetches the data week by week.
* **Checked against the source (Nov 2026):** rebuilding `prices_2016_2025.csv` from a fresh
  API download gives a byte-for-byte identical file. For the wind and solar forecasts, every
  half-hour present in both the original export and the API has identical values. The API now
  also has about 338 half-hours a year that were missing from the 2025 website export (the
  original code filled those with averages, see below).
* **Licence:** BMRS data is published under Elexon's
  [licence to use BMRS open data](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/),
  which allows copying and redistribution with attribution:
  *Contains BMRS data © Elexon Limited copyright and database right 2025.*

### How the raw data becomes the processed data (`prepare_data.py`)

1. **Prices:** drop the N2EX rows → place on a complete half-hourly timeline → fill each
   missing half-hour with the mean of the last value before it and the next value after it.
2. **Wind/solar:** keep the day-ahead forecasts (not the intraday updates) → split into the three
   types → place on the timeline → fill each missing half-hour with the mean of the same
   half-hour on the previous 7 days.
3. `SettlementPeriod` is recalculated from the UTC time (00:00 UTC is period 1).

---

## 🚀 Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_backtests.py
```

```bash
python make_plots.py
```

All data the backtests need is already in the repository, so those two commands reproduce the
results and figures. The full pipeline, in order:

| Step | Command | Makes | Time |
|---|---|---|---|
| 1 | `python download_data.py` | `data/raw/` from the Elexon API | ~5 min |
| 2 | `python prepare_data.py` | `data/processed/` (add `--check` to only look for gaps) | ~5 s |
| 3 | `python train_ann.py` (`--seed N`, `--seeds`, `--tune 50`) | `results/ann_forecast*.csv` | minutes, needs TensorFlow |
| 4 | `python run_backtests.py` (`--forecast FILE`) | `results/trades_*.csv` + a profit summary | ~1 s |
| 5 | `python make_plots.py` (`market generation ann trades`, `--show`, `--out DIR`) | `images/*.png` | ~30 s |

`pytest` runs the tests (about 5 s, no TensorFlow needed). They check the battery rules on small
hand-made inputs, and that the backtests still reproduce the original results to the penny.

> TensorFlow does not yet support the newest Python versions. Use Python 3.10–3.12 for
> `train_ann.py`; everything else works on any recent Python.

---

## 🗂️ Repository layout

```
├── download_data.py      step 1: Elexon API -> data/raw
├── prepare_data.py       step 2: data/raw -> data/processed (gap filling, checks)
├── train_ann.py          step 3: ANN price forecast
├── run_backtests.py      step 4: the three strategies
├── make_plots.py         step 5: every figure in the report
├── rp3/                  the reusable code the scripts call
│   ├── config.py         every path, date, battery and network setting
│   ├── elexon.py         API download
│   ├── data_prep.py      loading, timeline, gap filling, gap check
│   ├── battery.py        battery model and the three strategies
│   ├── forecasting.py    ANN features, training, multiple seeds, Optuna search
│   └── plotting.py       one function per figure
├── data/raw, data/processed
├── results/              ANN forecasts (trades_*.csv are made by run_backtests.py)
├── images/               report figures (nn_*.png are hand-drawn diagrams)
├── docs/feature_engineering_results.xlsx
└── tests/
```

To change an assumption, edit `rp3/config.py`. For example, set `SOC_STEP = 0.25` for a battery
that fills in four half-hours, or change `IMPORT_HOURS` for the time-based strategy.

---

## ⚙️ How the model works

### Battery (`rp3/battery.py`)

| Setting | Value |
|---|---|
| Capacity | 33.46 MWh, losing 2.5 % a year at one full cycle a day (0.025/365 per cycle) |
| Grid connection | capacity / 2.231 ≈ 15.0 MW |
| Round-trip efficiency | 94.56 %, applied to sales |
| State of charge (SoC) step per trade | 0.125 (8 half-hour trades fill or empty it) |

Every half-hour the strategy says *buy*, *sell* or *wait*. A buy is skipped if the battery is full,
and a sale is skipped if it is empty. A buy costs `power × price` and a sale earns
`power × price × efficiency`. One full cycle is a SoC distance of 2 (fill and empty), and the
capacity shrinks after each cycle.

**Speed:** the original scripts updated a DataFrame one row at a time (81 s for the time-based run,
59 s for the price-based one). Now only the SoC walk is a loop, over plain integers. The daily
ranking, cycle counting, degradation and cash flows are whole-array operations, so the runs take
0.06 s and 0.18 s.

### ANN forecast (`rp3/forecasting.py`)

To forecast the price at half-hour `t + 1 day`, the network uses information about half-hour `t`:
the price 1–6 days before `t`, day of week, hour, month, and the wind and solar forecasts.
It has two hidden layers of 160 ReLU neurons and was trained with Adam (learning rate 1.35e-4)
with early stopping. The settings came from a 50-trial Optuna search.

| Feature added | RMSE | MAE | RMSE change |
|---|---:|---:|---:|
| Hour of day | 39.76 | 27.87 | – |
| + Day of week | 39.16 | 27.23 | −1.5 % |
| + Month of year | 37.70 | 26.28 | −3.7 % |
| + Price lags (1–6 days) | 34.01 | 20.55 | −9.8 % |
| + Volume (later removed) | 37.11 | 24.06 | +9.1 % |
| + Outlier prices replaced | 26.83 | 19.41 | −27.7 % |
| + Offshore wind | 27.15 | 19.73 | +1.2 % |
| + Onshore wind | 26.74 | 19.60 | −1.5 % |
| + Solar | 26.67 | 19.39 | −0.3 % |
| + Wind forecasts ×100 | 25.57 | 19.00 | −4.1 % |

(from `docs/feature_engineering_results.xlsx`; see issues 4 and 5 before reading too much into the
last two rows)

---

## ⚠️ Known issues

Found while tidying the code. The refactor keeps the **original behaviour** so the numbers still
match the report. Fixing these would change the results, so they are listed here, with their
effect measured where possible.

**Modelling**

1. **The ANN strategy trades one day out of step.** The forecast row at time `t` predicts the price
   at `t + 1 day`, but the backtest picks slots using that row and then trades at the price at `t`.
   With the forecasts moved to the half-hour they predict, ANN profit from 19 June 2023 rises from
   £2.055m to **£2.097m**. That is not a big change, because daily price shapes repeat from one day to the next.
2. **Energy per trade doesn't match the SoC step.** A trade moves the SoC by 0.125 × 33.46 =
   4.18 MWh, but its cash flow is `15 MW × price`, i.e. 15 MWh (a full hour at 15 MW, or 30 MW for
   half an hour). Profits are therefore about 3.6 times what the modelled energy would earn.
   A consistent model would use 0.5 h × 15 MW = 7.5 MWh per trade and a SoC step of 0.224,
   or keep the 0.125 step and trade 4.18 MWh.
3. **All times are UTC.** The trading windows and `SettlementPeriod` use UTC hours, but GB settlement
   periods and the evening peak follow UK local time. During British Summer Time (58 % of rows) the
   "17:00–20:59" window is 18:00–21:59 local, and period 1 is 01:00 local. Using UK local time
   for the time-based windows raises its profit from £9.795m to **£10.155m**.
4. **The ANN error is measured against clipped prices.** Prices outside −35..220 are replaced
   by 73 in the *target* as well as the inputs, so the model is never scored on the spikes.
   Against the real next-day prices the test RMSE is 32.6, not 25.8. The "outlier prices
   replaced" gain of −27.7 % in the feature table is mostly this effect.
5. **The wind weight does nothing.** The features are min-max scaled, which cancels any constant
   multiplier (`tests/test_forecasting.py` checks this). The ×100 gain in the table is
   run-to-run noise. No seed was set, so each training run gives different results.
6. **The wind/solar forecasts describe the wrong day.** The row for `t` uses the forecast for `t`,
   while the target is `t + 1 day`. The forecast for `t + 1 day` is already published on the day
   before, so it could be used (the original had this shift commented out). Also, the most
   recent price the model sees is 2 days before the half-hour it predicts.
7. **Hyperparameters were tuned on the test set.** Each Optuna trial was scored on the test period,
   so the reported test error is slightly optimistic. Scoring on the validation split would fix this.
8. **Price-based ties.** When two half-hours have the same price at the 8th place, which one gets
   picked depended on the sorting algorithm (80 of 3,046 days). It is now always the earlier one,
   which moves the 2016–2025 total from £13,442,841 to £13,445,284.
9. **The import/export count chart shows planned trades.** It counts the slots chosen to trade,
   including ones skipped because the battery was full or empty.
10. **Units:** the DGWS forecast `Quantity` is average power in **MW**, but the figures label it MWh,
    and "total per year" sums MW values.

**Results files**

11. `results/ann_forecast.csv` (7 Mar 2025) is a later re-run, not the forecast behind the report's
    ANN trades and forecast figures. That forecast survives as `results/ann_forecast_2025-03-01.csv`
    (predictions only). `results/ann_forecast_multiple_seeds.csv` comes from an earlier version without
    the month feature.

### What the refactor changed

* 24 scripts → 5 step scripts plus 6 modules. Copy-pasted code is gone: the three trade scripts
  and seven plotting scripts each became one module, and two duplicate `ErrorCheck.py` files
  became one function.
* Hard-coded `C:\Users\...` paths → paths relative to the repository, all in `rp3/config.py`.
* Row-by-row loops → vectorised (backtests ~500× faster; the wind/solar gap filling no longer
  rescans the whole table for every gap and takes about 1 s per series).
* Scripts that had to be edited and re-run once per file (`Exogenous Data/CreateFullDataSet.py`
  had `Wind Onshore` hard-coded) → one function with a parameter.
* New: `download_data.py` (the raw price data was not in the folder before), tests, and checks
  that the results still match the original.
* Deprecated library calls updated (Keras `input_shape`, Optuna `suggest_loguniform`, seaborn
  `palette` without `hue`).

| Original script | Now |
|---|---|
| `ReadJson.py`, `FormatCSV.py` | `download_data.py`, `data_prep.build_prices` |
| `CreateFullTimeRange.py`, `CreateFullDataSet.py` (both copies), `Exogenous Data/FixSettlementPeriod.py`, `SeparateWindAndSolar.py` | `data_prep.on_full_timeline`, `fill_with_neighbour_mean`, `fill_with_same_period_last_week`, `build_generation` |
| `ErrorCheck.py` (both copies) | `data_prep.find_gaps`, `prepare_data.py --check` |
| `TimeBasedTrades.py`, `PriceBasedTrades.py`, `PredictedPriceBasedTrades.py`, `ActualAndPredictedPriceAggregate.py` | `battery.py`, `run_backtests.py` |
| `ANNForecast.py`, `MultipleModels.py`, `HyperparameterOptimisation.py` | `forecasting.py`, `train_ann.py` |
| `InitialPlots.py`, `FurtherPlots.py`, `VolumeAnalysis.py`, `ProfitPlots.py`, `PredictedPriceProfitPlots.py`, `ANNForecastVisualisation.py`, `Exogenous Data/AnalysisPlots.py` | `plotting.py`, `make_plots.py` |

The original code is kept unchanged in the first commit on `main`.

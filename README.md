# 🔋 Battery Arbitrage on the GB Electricity Market (RP3)

How much could a grid-scale battery (33.46 MWh, about 15 MW) earn by buying electricity
when it is cheap and selling it when it is expensive? This project compares three ways of
choosing when to trade, using real half-hourly GB wholesale prices from September 2016 to
January 2025:

| Strategy | How it picks the half-hours to trade |
|---|---|
| ⏰ **Time based** | Buys 01:00–04:59 and sells 17:00–20:59 (UK time) every day |
| 🎯 **Price based** | Buys the 5 cheapest and sells the 5 dearest half-hours of each day, from the *real* prices (perfect foresight, so an upper benchmark) |
| 🧠 **ANN based** | The same, but picks the half-hours from a neural-network forecast of the next day's prices |

This is the third-year research project (RP3, 2025) that came before my fourth-year project (RP4). It
is deliberately simple: one battery model, three strategies, and one feed-forward network.

---

## 📊 Results

From `python run_backtests.py` (GBP millions):

| Strategy | Profit 19 Sep 2016 → 20 Jan 2025 | Profit 24 May 2023 → 20 Jan 2025 | Full cycles |
|---|---:|---:|---:|
| Time based | 3.094 | 0.540 | 3045 |
| Price based (perfect foresight) | 4.343 | 0.778 | 2678 |
| ANN based | – | **0.694** | 561 |

The ANN only has forecasts for its test period (the last 20 % of the data), so the three
strategies are compared from the first full UK day of that period, 24 May 2023. Over that
period the forecast closes **65 %** of the gap between fixed hours and perfect foresight.

ANN test-set error, against the real (unclipped) price of each half-hour:

| Forecast | RMSE | MAE |
|---|---:|---:|
| **ANN** (seed 42) | **30.39** | **18.09** |
| ANN, seeds 42 / 100 / 7 | 30.39 / 32.30 / 31.96 | 18.09 / 20.47 / 20.07 |
| Naive: same as 2 days earlier | 38.54 | 23.78 |
| Naive: same as 7 days earlier | 39.79 | 25.11 |

The spread between seeds is as large as some feature effects, so differences of 1–2 GBP/MWh
between runs are noise.

<p align="center">
  <img src="images/Rolling Cashflow Over Time for Predicted Price.png" width="48%">
  <img src="images/Total Profit per Year for Predicted Price.png" width="48%">
</p>

> **These numbers are lower than in the 2025 report.** The report charged every trade for 15 MWh
> (`15 MW × price`), so 8 trades a day put 120 MWh through a 33.46 MWh battery. The battery now
> runs at its full 15 MW (7.5 MWh per half-hour) and can only move what it can store. See
> [Fixes since the report](#%EF%B8%8F-fixes-since-the-report) for this and the other corrections.

---

## 📥 Data sources

Both datasets are public and come from **Elexon's Balancing Mechanism Reporting Service (BMRS)**,
the official source of GB electricity market data. No API key is needed (see [API keys](#-api-keys)).

| | Prices and volumes | Wind and solar forecasts |
|---|---|---|
| **BMRS dataset** | `MID`: Market Index Data | `DGWS`: Day-ahead generation forecast for wind and solar |
| **What it is** | Half-hourly short-term wholesale price (GBP/MWh) and traded volume (MWh) | NESO's forecast of offshore wind, onshore wind and solar output (MW) for each half-hour, published about 17:45 UK time the day before |
| **Web page** | <https://bmrs.elexon.co.uk/market-index-prices> | <https://bmrs.elexon.co.uk/day-ahead-generation-forecast-wind-and-solar> |
| **API endpoint** | `https://data.elexon.co.uk/bmrs/api/v1/datasets/MID` | `https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind-and-solar/day-ahead` |
| **Raw file here** | `data/raw/market_index_data_raw.csv` (made by `download_data.py`, git-ignored) | `data/raw/DayAheadGenerationForecastWindAndSolar.csv` (made by `download_data.py`) |
| **Processed file** | `data/processed/prices_2016_2025.csv` | `data/processed/{wind_offshore,wind_onshore,solar}_da_forecast_2016_2025.csv` |

* **Period:** 2016-09-19 00:00 to 2025-01-20 00:00 UTC, every half-hour (146,161 rows).
* **Market index providers:** two exchanges report the MID. Only APX (`APXMIDP`) has prices;
  N2EX (`N2EXMIDP`) reports zero for almost every period, so it is dropped.
* **API documentation:** <https://bmrs.elexon.co.uk/api-documentation>. Each request can
  cover at most 7 days, so `download_data.py` fetches the data week by week.
* **Checked against the source (Nov 2026):** the prices rebuilt from the API are identical to
  the ones used in the report. The 2025 report used a CSV export of the wind/solar forecasts
  from the BMRS website; the API has the same values plus 3,230 half-hours that export was
  missing, so the API download is now used. 111 whole days (mostly 2020–2022) have no
  day-ahead forecast at all from NESO and are filled as described below.
* **Downloaded:** 21 November 2026, with `python download_data.py`.
* **Licence:** BMRS data is published under Elexon's
  [licence to use BMRS open data](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/),
  which allows copying and redistribution with attribution:
  *Contains BMRS data © Elexon Limited copyright and database right 2025.*

### How the raw data becomes the processed data (`prepare_data.py`)

1. **Prices:** drop the N2EX rows → place on a complete half-hourly timeline → fill each
   missing half-hour (2,773, 1.9 %) with the mean of the last value before it and the next value after it.
2. **Wind/solar:** keep the first day-ahead forecast for each half-hour → split into the three
   types → place on the timeline → fill each missing half-hour (5,328, 3.6 %) with the mean of the
   same half-hour on the previous 7 days.
3. `SettlementDate` and `SettlementPeriod` follow the GB market: the UK local day, with period 1
   starting at UK midnight (46 periods when the clocks go forward, 50 when they go back). They
   match Elexon's own values on every row (checked by `tests/test_data_prep.py`).
4. Every filled half-hour is marked `Filled = True`, so it can be left out of an analysis.

---

## 🔑 API keys

Elexon's API is open, so **no key is needed** and none is stored anywhere in this repository.
As in RP4, a key would go in a git-ignored `.env` file at the repository root, read by
`python-dotenv` only when a request is made and never printed:

```bash
cp .env.example .env
```

```
ELEXON_API_KEY=your-key-here
```

If `ELEXON_API_KEY` is set, `rp3/elexon.py` sends it with every request; if it's empty, requests
are sent without it.

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
| 2 | `python prepare_data.py` | `data/processed/` (add `--check` to only look for gaps) | ~10 s |
| 3 | `python train_ann.py` (`--seed N`, `--seeds`, `--tune 50`) | `results/ann_forecast*.csv` | ~12 s per model (`--tune 50`: ~30 min) |
| 4 | `python run_backtests.py` (`--forecast FILE`) | `results/trades_*.csv` + a profit summary | ~1 s |
| 5 | `python make_plots.py` (`market generation ann trades`, `--show`, `--out DIR`) | `images/*.png` | ~30 s |

`pytest` runs the tests (about 10 s). They check the battery rules on small hand-made inputs, the
settlement periods against Elexon's official values, and that the network trains.

Everything runs on any recent Python (tested on 3.14) on Windows, macOS or Linux. All paths are
relative to the repository, so it runs from wherever it is cloned.

---

## 🗂️ Repository layout

```
├── download_data.py      step 1: Elexon API -> data/raw
├── .env.example          where an API key would go (copy to .env)
├── prepare_data.py       step 2: data/raw -> data/processed (gap filling, checks)
├── train_ann.py          step 3: ANN price forecast
├── run_backtests.py      step 4: the three strategies
├── make_plots.py         step 5: every figure in the report
├── rp3/                  the reusable code the scripts call
│   ├── config.py         every path, date, battery and network setting
│   ├── elexon.py         API download
│   ├── data_prep.py      loading, timeline, gap filling, gap check
│   ├── battery.py        battery model and the three strategies
│   ├── forecasting.py    ANN (scikit-learn): features, training, seeds, Optuna search
│   └── plotting.py       one function per figure
├── data/raw, data/processed
├── results/              ANN forecasts (trades_*.csv are made by run_backtests.py)
├── images/               figures (nn_*.png are hand-drawn diagrams of the original network)
├── docs/feature_engineering_results.xlsx
├── LICENSE               MIT (code; the data licence is in the README)
└── tests/
```

To change an assumption, edit `rp3/config.py`. For example, set `CAPACITY_TO_POWER_RATIO = 4` for a
4-hour battery (8.4 MW, 8 half-hours to fill), or change `IMPORT_HOURS` for the time-based strategy.

---

## ⚙️ How the model works

### Battery (`rp3/battery.py`)

| Setting | Value |
|---|---|
| Capacity | 33.46 MWh, losing 2.5 % a year at one full cycle a day (0.025/365 per cycle) |
| Grid connection | capacity / 2.231 ≈ 15.0 MW |
| Round-trip efficiency | 94.56 %, applied to sales |
| Energy per half-hour trade | up to 15 MW × 0.5 h = 7.5 MWh (5 half-hours to fill: 4 × 7.5 + 3.46 MWh) |

Every half-hour the strategy says *buy*, *sell* or *wait*. The battery trades at its full grid
connection, so a half-hour moves 7.5 MWh, or only what still fits when it is nearly full or empty
(nothing at all when it is full or empty). A buy costs `energy × price` and a sale earns
`energy × price × efficiency`. One full cycle is a state-of-charge distance of 2 (fill and empty);
after each cycle the capacity, and with it the power, shrinks. Days and hours are UK local time.

Because a full charge takes 5 half-hours, the price-ranked strategies trade the 5 cheapest and
5 dearest half-hours of each day. The time-based strategy keeps its 4-hour windows and fills or
empties in the first 2.5 hours of each.

**Speed:** the original scripts updated a DataFrame one row at a time (81 s for the time-based run,
59 s for the price-based one). Now the daily ranking is a whole-array operation and the battery
is a plain loop over numbers (its state depends on every earlier trade), so the runs take 0.25 s
and 0.3 s.

### ANN forecast (`rp3/forecasting.py`)

Each row is one half-hour `T` to forecast, using only what is known on the evening before `T`'s
day, when the trades are chosen:

* the price 2–7 days before `T` (the price 1 day before is partly after the decision time)
* day of week, hour (UK time) and month of `T`
* NESO's day-ahead wind (offshore, onshore) and solar forecasts for `T`

The model is a scikit-learn `MLPRegressor`: one hidden layer of 256 ReLU neurons, Adam (learning
rate 0.0021, batch 64). The first 80 % of rows train it, with the last fifth of those used for
early stopping and for the 50-trial Optuna search that chose these settings. The last 20 % are
the test set, used once. It trains on prices clipped to −35..220 GBP/MWh, so a few spikes don't
dominate, but it is scored on the real prices.

The 2025 report's feature study (`docs/feature_engineering_results.xlsx`, done with the original
TensorFlow model and setup):

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

Read it with care: those errors were measured against clipped prices (which is most of the
"outlier prices" gain), the ×100 weight has no effect after scaling, and no seed was fixed. Seed
noise alone is about ±1 GBP/MWh (see Results).

---

## 🛠️ Fixes since the report

The first refactor kept the report's behaviour exactly. These modelling problems were then fixed
(the effects in the last column were measured one at a time on the original model):

| # | Problem in the 2025 code | Fix | Effect |
|---|---|---|---|
| 1 | **ANN traded one day out of step:** the forecast for `t + 1 day` was used to trade at the price at `t` | Each forecast row is the half-hour it predicts, and trades at that half-hour's price | ANN profit +2 % (£2.055m → £2.097m) |
| 2 | **Energy per trade didn't match the battery:** each trade moved 4.18 MWh of charge but was charged/paid for 15 MWh (`15 MW × price`), i.e. 120 MWh a day through a 33.46 MWh battery | The battery trades at its full 15 MW: 7.5 MWh per half-hour, limited by what it can store, 5 slots a day | Profits about 3.3× lower |
| 3 | **Everything in UTC,** while GB trading days, settlement periods and the evening peak follow UK time | Strategy hours, trading days and settlement periods use UK local time | Time-based +3.7 % (£9.795m → £10.155m) |
| 4 | **ANN scored against clipped prices** (spikes replaced by 73 in the target) | Trained on clipped prices, scored on real ones | Reported RMSE 25.8 was really 32.6 |
| 5 | **Wind ×100 weight** (cancelled by min-max scaling) and **no fixed seed** | Weight removed; `SEED = 42` | Runs are repeatable |
| 6 | **Wind/solar forecasts for the wrong day** (for `t`, while predicting `t + 1 day`) | Forecasts for the half-hour being predicted | Part of the new model |
| 7 | **Hyperparameters tuned on the test set** | Tuned on a validation period before the test set; re-tuned for the new model | Honest test error |
| 8 | **TensorFlow** (doesn't support current Python) | scikit-learn `MLPRegressor` with the same training method (Adam, chronological early stopping) | Trains in 12 s on Python 3.14 |
| 9 | **Import/export chart counted planned trades** | Counts executed trades | |
| 10 | **Wind/solar units:** MW labelled MWh; yearly "totals" summed MW | Labelled MW; yearly energy = MW × 0.5 h, in TWh | Yearly totals were 2× too high |
| 11 | Forecasts came from an incomplete website export; solar gaps rounded to whole MW; made-up `PublishTime` for gaps | API download (3,230 more real half-hours); same rounding for all; gaps marked `Filled` | |

Remaining limitations, by design: the price-based strategy has perfect foresight (an upper bound,
not a real strategy); the battery always fills and empties once a day in fixed-size trades; prices are the Market Index
(short-term spot), not a specific day-ahead auction; no fees, imbalance costs or bid-offer spread.

### What the refactor changed

* 24 scripts → 5 step scripts plus 6 modules. Copy-pasted code is gone: the three trade scripts
  and seven plotting scripts each became one module, and two duplicate `ErrorCheck.py` files
  became one function.
* Hard-coded `C:\Users\...` paths → paths relative to the repository, all in `rp3/config.py`.
* Row-by-row loops → vectorised (backtests ~500× faster; the wind/solar gap filling no longer
  rescans the whole table for every gap and takes about 1 s per series).
* Scripts that had to be edited and re-run once per file (`Exogenous Data/CreateFullDataSet.py`
  had `Wind Onshore` hard-coded) → one function with a parameter.
* New: `download_data.py` (the raw price data was not in the folder before), `.env` support, and tests.
* TensorFlow replaced by scikit-learn; deprecated Optuna and seaborn calls updated.

| Original script | Now |
|---|---|
| `ReadJson.py`, `FormatCSV.py` | `download_data.py`, `data_prep.build_prices` |
| `CreateFullTimeRange.py`, `CreateFullDataSet.py` (both copies), `Exogenous Data/FixSettlementPeriod.py`, `SeparateWindAndSolar.py` | `data_prep.on_full_timeline`, `fill_with_neighbour_mean`, `fill_with_same_period_last_week`, `build_generation` |
| `ErrorCheck.py` (both copies) | `data_prep.find_gaps`, `prepare_data.py --check` |
| `TimeBasedTrades.py`, `PriceBasedTrades.py`, `PredictedPriceBasedTrades.py`, `ActualAndPredictedPriceAggregate.py` | `battery.py`, `run_backtests.py` |
| `ANNForecast.py`, `MultipleModels.py`, `HyperparameterOptimisation.py` | `forecasting.py`, `train_ann.py` |
| `InitialPlots.py`, `FurtherPlots.py`, `VolumeAnalysis.py`, `ProfitPlots.py`, `PredictedPriceProfitPlots.py`, `ANNForecastVisualisation.py`, `Exogenous Data/AnalysisPlots.py` | `plotting.py`, `make_plots.py` |

The original code is kept unchanged in the first commit on `main`, and the refactor that
reproduces the report's numbers exactly is the commit after it.

---

## 📄 Licence

The code, figures and documentation are released under the [MIT License](LICENSE): anyone may
use, copy, modify and share them, including commercially, as long as the copyright notice is kept.

The data in `data/` is Elexon BMRS data, used under Elexon's
[licence to use BMRS open data](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/).
If you reuse it, keep the attribution:
*Contains BMRS data © Elexon Limited copyright and database right 2025.*

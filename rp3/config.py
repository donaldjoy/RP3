"""Every path, date range and model setting in one place.

All paths are relative to the repository root, so the project runs from wherever it
is cloned. Change a number here, not inside a script.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Folders and files
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "images"

# Raw downloads from Elexon BMRS (see README "Data sources")
RAW_PRICES_CSV = RAW_DIR / "market_index_data_raw.csv"                    # dataset MID
RAW_GENERATION_CSV = RAW_DIR / "DayAheadGenerationForecastWindAndSolar.csv"  # dataset DGWS

# Cleaned, gap-free half-hourly series used by every model
PRICES_CSV = PROCESSED_DIR / "prices_2016_2025.csv"
GENERATION_CSVS = {
    "Wind Offshore": PROCESSED_DIR / "wind_offshore_da_forecast_2016_2025.csv",
    "Wind Onshore": PROCESSED_DIR / "wind_onshore_da_forecast_2016_2025.csv",
    "Solar": PROCESSED_DIR / "solar_da_forecast_2016_2025.csv",
}

# Model outputs
ANN_FORECAST_CSV = RESULTS_DIR / "ann_forecast.csv"
ANN_MULTI_SEED_CSV = RESULTS_DIR / "ann_forecast_multiple_seeds.csv"
TRADES_CSVS = {
    "time": RESULTS_DIR / "trades_time_based.csv",
    "price": RESULTS_DIR / "trades_price_based.csv",
    "ann": RESULTS_DIR / "trades_ann_based.csv",
}

# ---------------------------------------------------------------------------
# Study period (all times are UTC, half-hourly)
# ---------------------------------------------------------------------------
START = pd.Timestamp("2016-09-19 00:00", tz="UTC")
END = pd.Timestamp("2025-01-20 00:00", tz="UTC")
PERIODS_PER_DAY = 48
# The three-strategy comparison starts where the multiple-seed ANN test set starts
COMPARISON_START = pd.Timestamp("2023-06-19 09:30", tz="UTC")

# ---------------------------------------------------------------------------
# Battery (a 33.46 MWh grid-scale lithium-ion unit)
# ---------------------------------------------------------------------------
BATTERY_CAPACITY_MWH = 33.46
DEGRADATION_PER_CYCLE = 0.025 / 365   # 2.5 % capacity loss per year at one cycle per day
ROUND_TRIP_EFFICIENCY = 0.9456
SOC_STEP = 0.125                      # state-of-charge change per half-hour trade (8 trades fill it)
CAPACITY_TO_POWER_RATIO = 2.231       # MWh of capacity per MW of grid connection
SLOTS_PER_DAY = 8                     # half-hours bought and sold per day (= 1 / SOC_STEP)

# Time-based strategy windows, inclusive UTC hours
IMPORT_HOURS = (1, 4)     # 01:00-04:59: buy
EXPORT_HOURS = (17, 20)   # 17:00-20:59: sell

# ---------------------------------------------------------------------------
# ANN price forecast
# ---------------------------------------------------------------------------
PRICE_FLOOR, PRICE_CAP = -35, 220   # prices outside this band are treated as outliers
FILL_VALUE = 73                     # value used for outliers and missing data
PRICE_LAG_DAYS = 6                  # features: the price 1..6 days earlier
TRAIN_RATIO = 0.8                   # first 80 % of rows train, last 20 % test
HIDDEN_LAYERS = (160, 160)          # neurons per hidden layer (from the Optuna search)
LEARNING_RATE = 0.00013504016404346537
EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
EARLY_STOP_PATIENCE = 3
EXOGENOUS_WEIGHT = 2.0              # multiplier on the wind forecasts (see README: no effect)
SEEDS = (42, 100, 7)                # for the multiple-seed run

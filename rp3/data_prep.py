"""Turn the raw Elexon downloads into complete half-hourly series, and load them.

Prices (MID):     drop the empty N2EX rows -> put on a complete half-hourly timeline
                  -> fill each gap with the mean of the values either side.
Wind/solar (DGWS): keep the day-ahead forecasts -> split by type -> put on the timeline
                  -> fill each gap with the mean of the same half-hour on the 7 days before.

Settlement periods are recomputed from the UTC time (00:00 UTC = period 1), so they
match the UTC hours the strategies use. Official BMRS periods follow UK local time,
see README "Known issues".
"""
import numpy as np
import pandas as pd

from rp3 import config

HALF_HOUR = pd.Timedelta(minutes=30)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_timeseries(path):
    """Read a project CSV with a UTC ``StartTime`` column, sorted by time."""
    df = pd.read_csv(path)
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)
    return df.sort_values("StartTime", ignore_index=True)


def load_prices(path=config.PRICES_CSV):
    return read_timeseries(path)


def load_generation(paths=None):
    """The three day-ahead forecasts as {"Wind Offshore": df, ...}."""
    return {name: read_timeseries(path) for name, path in (paths or config.GENERATION_CSVS).items()}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def full_timeline(start=config.START, end=config.END):
    return pd.date_range(start, end, freq=HALF_HOUR, name="StartTime")


def settlement_period(times):
    """Half-hour of the UTC day, 1..48 (00:00 -> 1, 23:30 -> 48)."""
    times = pd.DatetimeIndex(times)
    return times.hour * 2 + times.minute // 30 + 1


def on_full_timeline(df, start=config.START, end=config.END):
    """Reindex onto every half-hour from start to end; missing half-hours become NaN rows."""
    df = df.drop_duplicates("StartTime").set_index("StartTime")
    return df.reindex(full_timeline(start, end)).reset_index()


def fill_with_neighbour_mean(series):
    """Fill each gap with the mean of the last value before it and the next value after it."""
    return series.fillna((series.ffill() + series.bfill()) / 2)


def fill_with_same_period_last_week(series, decimals=3, periods_per_day=config.PERIODS_PER_DAY, days=7):
    """Fill each gap with the mean of the same half-hour on each of the previous ``days`` days.

    Gaps are filled in time order, so a filled value can be used for a later gap
    (as in the original). With no earlier data the gap is set to 0. Needs a
    complete timeline, so the same half-hour is exactly ``periods_per_day`` rows back.
    """
    values = series.to_numpy(dtype=float, copy=True)
    lags = periods_per_day * np.arange(1, days + 1)
    for i in np.flatnonzero(np.isnan(values)):
        window = values[i - lags[lags <= i]]
        window = window[~np.isnan(window)]
        values[i] = np.round(window.mean(), decimals) if len(window) else 0.0
    return pd.Series(values, index=series.index, name=series.name)


def find_gaps(times, step=HALF_HOUR):
    """Rows where the next time is not exactly ``step`` after the previous one."""
    times = pd.Series(pd.to_datetime(times, utc=True)).reset_index(drop=True)
    expected = times.shift() + step
    bad = times.index[1:][times.iloc[1:].ne(expected.iloc[1:]).to_numpy()]
    return pd.DataFrame({"csv_row": bad + 2, "expected": expected[bad], "actual": times[bad]})


# ---------------------------------------------------------------------------
# Building the processed datasets
# ---------------------------------------------------------------------------
def build_prices(raw_mid, start=config.START, end=config.END):
    """Market Index Data -> StartTime, SettlementDate, SettlementPeriod, Price, Volume."""
    df = raw_mid[raw_mid["DataProvider"] != "N2EXMIDP"].copy()
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)
    df = on_full_timeline(df[["StartTime", "Price", "Volume"]], start, end)
    for col in ["Price", "Volume"]:
        df[col] = fill_with_neighbour_mean(df[col])
    df.insert(1, "SettlementDate", df["StartTime"].dt.strftime("%d/%m/%Y"))
    df.insert(2, "SettlementPeriod", settlement_period(df["StartTime"]))
    return df


GENERATION_TYPES = {
    "Wind Offshore": "Wind generation",
    "Wind Onshore": "Wind generation",
    "Solar": "Solar generation",
}
# Decimal places for filled gaps. The original filled solar gaps to whole MW and wind to 3 d.p.
FILL_DECIMALS = {"Wind Offshore": 3, "Wind Onshore": 3, "Solar": 0}


def build_generation(raw_dgws, psr_type, start=config.START, end=config.END):
    """One day-ahead forecast series (e.g. "Solar") on the complete timeline."""
    df = raw_dgws[(raw_dgws["PsrType"] == psr_type) & (raw_dgws["ProcessType"] == "Day ahead")].copy()
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)
    df["SettlementPeriod"] = settlement_period(df["StartTime"])
    df = on_full_timeline(df, start, end)

    missing = df["PublishTime"].isna()
    # Forecasts are published at about 16:45 UTC the day before
    df.loc[missing, "PublishTime"] = (df.loc[missing, "StartTime"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%dT16:45:10Z")
    df["ProcessType"] = df["ProcessType"].fillna("Day ahead")
    df["BusinessType"] = df["BusinessType"].fillna(GENERATION_TYPES[psr_type])
    df["PsrType"] = df["PsrType"].fillna(psr_type)
    missing = df["SettlementDate"].isna()
    df.loc[missing, "SettlementDate"] = df.loc[missing, "StartTime"].dt.strftime("%d/%m/%Y")
    df["SettlementPeriod"] = settlement_period(df["StartTime"])
    df["Quantity"] = fill_with_same_period_last_week(df["Quantity"], FILL_DECIMALS[psr_type])
    return df[["PublishTime", "ProcessType", "BusinessType", "PsrType", "StartTime",
               "SettlementDate", "SettlementPeriod", "Quantity"]]

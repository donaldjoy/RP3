"""Turn the raw Elexon downloads into complete half-hourly series, and load them.

Prices (MID):     drop the empty N2EX rows -> put on a complete half-hourly timeline
                  -> fill each gap with the mean of the values either side.
Wind/solar (DGWS): keep the day-ahead forecasts -> split by type -> put on the timeline
                  -> fill each gap with the mean of the same half-hour on the 7 days before.

Times are stored in UTC. ``SettlementDate`` and ``SettlementPeriod`` follow the GB
market's definition: the UK local day, with period 1 starting at UK midnight (so there
are 46 periods on the day the clocks go forward and 50 on the day they go back).
Every gap that was filled is marked in a ``Filled`` column.
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


def uk_time(times):
    """UTC times as UK local time (GMT or BST)."""
    return pd.DatetimeIndex(times).tz_convert(config.TIMEZONE)


def settlement_date(times):
    """The UK local date each half-hour belongs to (the GB trading day)."""
    return uk_time(times).date


def settlement_period(times):
    """GB settlement period: half-hours since UK midnight, plus one (1..46/48/50)."""
    local = uk_time(times)
    return np.asarray((local - local.normalize()) // HALF_HOUR + 1, dtype=int)


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
    df["Filled"] = df["Price"].isna()
    for col in ["Price", "Volume"]:
        df[col] = fill_with_neighbour_mean(df[col])
    df.insert(1, "SettlementDate", pd.to_datetime(settlement_date(df["StartTime"])).strftime("%d/%m/%Y"))
    df.insert(2, "SettlementPeriod", settlement_period(df["StartTime"]))
    return df


GENERATION_TYPES = {
    "Wind Offshore": "Wind generation",
    "Wind Onshore": "Wind generation",
    "Solar": "Solar generation",
}


def build_generation(raw_dgws, psr_type, start=config.START, end=config.END):
    """One day-ahead forecast series (e.g. "Solar", in MW) on the complete timeline.

    If a half-hour appears more than once, the first row is kept (the download keeps
    the earliest publication first).
    """
    df = raw_dgws[(raw_dgws["PsrType"] == psr_type) & (raw_dgws["ProcessType"] == "Day ahead")].copy()
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)
    df = on_full_timeline(df, start, end)

    df["Filled"] = df["Quantity"].isna()
    df["ProcessType"] = "Day ahead"
    df["BusinessType"] = GENERATION_TYPES[psr_type]
    df["PsrType"] = psr_type
    df["SettlementDate"] = pd.to_datetime(settlement_date(df["StartTime"])).strftime("%d/%m/%Y")
    df["SettlementPeriod"] = settlement_period(df["StartTime"])
    df["Quantity"] = fill_with_same_period_last_week(df["Quantity"])
    return df[["PublishTime", "ProcessType", "BusinessType", "PsrType", "StartTime",
               "SettlementDate", "SettlementPeriod", "Quantity", "Filled"]]

import numpy as np
import pandas as pd

from rp3.data_prep import (build_generation, build_prices, fill_with_neighbour_mean,
                           fill_with_same_period_last_week, find_gaps, settlement_period)

T0 = pd.Timestamp("2024-03-01", tz="UTC")


def test_settlement_period_counts_half_hours_from_midnight_utc():
    times = pd.to_datetime(["2024-06-01 00:00", "2024-06-01 00:30", "2024-06-01 23:30"], utc=True)
    assert list(settlement_period(times)) == [1, 2, 48]


def test_neighbour_mean_fill():
    s = pd.Series([1.0, np.nan, np.nan, 5.0, np.nan])
    assert fill_with_neighbour_mean(s).tolist()[:4] == [1, 3, 3, 5]
    assert np.isnan(fill_with_neighbour_mean(s).iloc[-1])   # nothing after the last value


def test_same_period_fill_uses_the_previous_seven_days_and_chains():
    per_day = 2
    s = pd.Series([1.0, 10, 3.0, 20, np.nan, 30, np.nan, np.nan])
    out = fill_with_same_period_last_week(s, decimals=3, periods_per_day=per_day)
    assert out[4] == 2.0          # mean of 1 and 3
    assert out[6] == 2.0          # mean of 1, 3 and the filled 2
    assert out[7] == 20.0         # mean of 10, 20, 30
    assert fill_with_same_period_last_week(pd.Series([np.nan]))[0] == 0.0


def test_find_gaps_reports_csv_rows():
    times = pd.Series(T0 + pd.to_timedelta([0, 30, 90, 120], unit="min"))
    gaps = find_gaps(times)
    assert gaps["csv_row"].tolist() == [4]
    assert find_gaps(times.iloc[2:]).empty


def test_build_prices_drops_n2ex_and_fills_gaps():
    times = (T0 + pd.to_timedelta([0, 60], unit="min")).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = pd.DataFrame({
        "StartTime": list(times) * 2, "DataProvider": ["APXMIDP"] * 2 + ["N2EXMIDP"] * 2,
        "SettlementDate": "01/03/2024", "SettlementPeriod": [1, 3, 1, 3],
        "Price": [10.0, 20.0, 0.0, 0.0], "Volume": [1.0, 3.0, 0.0, 0.0],
    })
    out = build_prices(raw, T0, T0 + pd.Timedelta(hours=1))
    assert out.columns.tolist() == ["StartTime", "SettlementDate", "SettlementPeriod", "Price", "Volume"]
    assert out["Price"].tolist() == [10, 15, 20]
    assert out["SettlementPeriod"].tolist() == [1, 2, 3]


def test_build_generation_completes_the_timeline():
    times = T0 + pd.to_timedelta(np.arange(0, 4 * 30, 30), unit="min")
    raw = pd.DataFrame({
        "PublishTime": "x", "ProcessType": ["Day ahead", "Day ahead", "Intraday process", "Day ahead"],
        "BusinessType": "Solar generation", "PsrType": "Solar",
        "StartTime": times.strftime("%Y-%m-%dT%H:%M:%SZ"), "SettlementDate": "01/03/2024",
        "SettlementPeriod": [1, 2, 3, 4], "Quantity": [5.0, 6.0, 99.0, 8.0],
    })
    out = build_generation(raw, "Solar", T0, times[-1])
    assert len(out) == 4
    assert out["ProcessType"].eq("Day ahead").all()
    assert out["Quantity"].tolist() == [5, 6, 0, 8]     # no earlier day to copy from -> 0

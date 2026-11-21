import numpy as np
import pandas as pd
import pytest

from rp3 import config
from rp3.battery import (EXPORT, IDLE, IMPORT, Battery, ann_based_trades, price_ranked_actions, simulate,
                         time_based_actions, time_based_trades)
from rp3.data_prep import load_prices, read_timeseries

NO_DEGRADATION = Battery(degradation_per_cycle=0.0)


def test_soc_stays_between_empty_and_full():
    actions = [EXPORT] + [IMPORT] * 10 + [EXPORT] * 10
    out = simulate(actions, np.full(21, 50.0), NO_DEGRADATION)
    assert out["SoC"].between(0, 1).all()
    assert out["SoC"].max() == 1.0
    # selling when empty and buying when full are skipped
    assert out["Executed"].iloc[0] == IDLE
    assert (out["Executed"] != IDLE).sum() == 16


def test_cash_flows_use_power_and_efficiency():
    b = NO_DEGRADATION
    out = simulate([IMPORT, EXPORT], [10.0, 30.0], b)
    power = b.capacity_mwh / b.capacity_to_power_ratio
    assert out["Cost"].tolist() == pytest.approx([-power * 10, 0])
    assert out["Revenue"].tolist() == pytest.approx([0, power * 30 * b.round_trip_efficiency])
    assert out["Rolling Cashflow"].iloc[-1] == pytest.approx(power * (30 * b.round_trip_efficiency - 10))


def test_one_full_cycle_degrades_the_next_trade():
    b = Battery(degradation_per_cycle=0.1)
    actions = [IMPORT] * 8 + [EXPORT] * 8 + [IMPORT]   # 16 trades = one full cycle
    out = simulate(actions, np.ones(17), b)
    assert out["Total Cycles Achieved"].iloc[15] == 1
    assert out["Total Cycles Achieved"].isna().sum() == 16
    first, last = -out["Cost"].iloc[0], -out["Cost"].iloc[16]
    assert last == pytest.approx(first * 0.9)


def test_time_based_windows_are_inclusive_hours():
    times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
    actions = time_based_actions(times)
    hours = times.hour
    assert (actions[(hours >= 1) & (hours <= 4)] == IMPORT).all()
    assert (actions[(hours >= 17) & (hours <= 20)] == EXPORT).all()
    assert (actions == IMPORT).sum() == 8 and (actions == EXPORT).sum() == 8


def test_price_ranking_picks_cheapest_and_dearest_each_day_with_earliest_tie_first():
    prices = np.r_[np.arange(48.0), np.full(48, 5.0)]
    days = np.repeat([1, 2], 48)
    actions = price_ranked_actions(prices, days, n_slots=8)
    assert (actions[:8] == IMPORT).all() and (actions[40:48] == EXPORT).all()
    assert (actions[8:40] == IDLE).all()
    # all prices equal: the first 8 slots are "cheapest" and also "dearest"; selling wins
    assert (actions[48:56] == EXPORT).all() and (actions[56:] == IDLE).all()


def test_ann_trades_join_forecast_to_price_at_the_same_time():
    times = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    forecast = pd.DataFrame({"StartTime": times, "SettlementDate": "x", "SettlementPeriod": 1,
                             "PredictedPrice": [1.0, 2, 3, 4]})
    prices = pd.DataFrame({"StartTime": times, "Price": [10.0, 20, 30, 40]})
    out = ann_based_trades(forecast, prices)
    assert out["Price"].tolist() == [10, 20, 30, 40]


# ---------------------------------------------------------------------------
# Regression: the refactor reproduces the original results on the committed data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prices():
    return load_prices()


def test_time_based_total_matches_original(prices):
    out = time_based_trades(prices)
    assert out["Rolling Cashflow"].iloc[-1] == pytest.approx(9794682.132365199, abs=1e-4)
    assert out["Total Cycles Achieved"].max() == 3045


def test_report_ann_run_matches_original(prices):
    forecast = read_timeseries(config.RESULTS_DIR / "ann_forecast_2025-03-01.csv")
    out = ann_based_trades(forecast, prices)
    assert out["Rolling Cashflow"].iloc[-1] == pytest.approx(2134849.8453069306, abs=1e-4)

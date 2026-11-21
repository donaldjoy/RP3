import numpy as np
import pandas as pd
import pytest

from rp3.battery import (EXPORT, IDLE, IMPORT, Battery, ann_based_trades, comparison_start, price_based_trades,
                         price_ranked_actions, simulate, time_based_actions, time_based_trades)
from rp3.data_prep import load_prices

NO_DEGRADATION = Battery(degradation_per_cycle=0.0)


def test_soc_stays_between_empty_and_full():
    actions = [EXPORT] + [IMPORT] * 10 + [EXPORT] * 10
    out = simulate(actions, np.full(21, 50.0), NO_DEGRADATION)
    assert out["SoC"].between(0, 1).all()
    assert out["SoC"].max() == 1.0
    # selling when empty and buying when full are skipped
    assert out["Executed"].iloc[0] == IDLE
    assert (out["Executed"] != IDLE).sum() == 16


def test_each_trade_moves_the_energy_of_one_soc_step():
    b = NO_DEGRADATION
    out = simulate([IMPORT, EXPORT], [10.0, 30.0], b)
    energy = b.soc_step * b.capacity_mwh                      # 4.18 MWh
    assert out["Energy (MWh)"].tolist() == pytest.approx([energy, energy])
    assert out["Cost"].tolist() == pytest.approx([-energy * 10, 0])
    assert out["Revenue"].tolist() == pytest.approx([0, energy * 30 * b.round_trip_efficiency])
    # buying 8 steps adds up to exactly one full battery
    full = simulate([IMPORT] * 8, np.ones(8), b)
    assert full["Energy (MWh)"].sum() == pytest.approx(b.capacity_mwh)


def test_trade_size_must_fit_the_grid_connection():
    assert Battery().energy_per_trade_mwh / 0.5 <= Battery().power_mw
    with pytest.raises(ValueError):
        Battery(soc_step=0.5)                                  # 16.7 MWh in half an hour = 33 MW > 15 MW


def test_one_full_cycle_degrades_the_next_trade():
    b = Battery(degradation_per_cycle=0.1)
    actions = [IMPORT] * 8 + [EXPORT] * 8 + [IMPORT]           # 16 trades = one full cycle
    out = simulate(actions, np.ones(17), b)
    assert out["Total Cycles Achieved"].iloc[15] == 1
    assert out["Total Cycles Achieved"].isna().sum() == 16
    assert out["Energy (MWh)"].iloc[16] == pytest.approx(out["Energy (MWh)"].iloc[0] * 0.9)


def test_time_based_windows_follow_uk_local_time():
    winter = pd.date_range("2024-01-15", periods=48, freq="30min", tz="Europe/London").tz_convert("UTC")
    summer = pd.date_range("2024-07-15", periods=48, freq="30min", tz="Europe/London").tz_convert("UTC")
    for times in (winter, summer):
        actions = time_based_actions(times)
        hours = times.tz_convert("Europe/London").hour
        assert (actions[(hours >= 1) & (hours <= 4)] == IMPORT).all()
        assert (actions[(hours >= 17) & (hours <= 20)] == EXPORT).all()
        assert (actions == IMPORT).sum() == 8 and (actions == EXPORT).sum() == 8
    # 16:00 UTC in July is 17:00 BST: a sell slot
    assert time_based_actions(pd.DatetimeIndex(["2024-07-15 16:00"], tz="UTC"))[0] == EXPORT


def test_price_ranking_picks_cheapest_and_dearest_each_day_with_earliest_tie_first():
    prices = np.r_[np.arange(48.0), np.full(48, 5.0)]
    days = np.repeat([1, 2], 48)
    actions = price_ranked_actions(prices, days, n_slots=8)
    assert (actions[:8] == IMPORT).all() and (actions[40:48] == EXPORT).all()
    assert (actions[8:40] == IDLE).all()
    # all prices equal: the first 8 slots are "cheapest" and also "dearest"; selling wins
    assert (actions[48:56] == EXPORT).all() and (actions[56:] == IDLE).all()


def test_ann_trades_use_the_price_of_the_half_hour_that_was_forecast():
    times = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
    prices = pd.DataFrame({"StartTime": times, "SettlementDate": "01/01/2024",
                           "SettlementPeriod": range(1, 49), "Price": np.arange(48.0)})
    # a perfect forecast picks the same slots as perfect foresight
    forecast = pd.DataFrame({"StartTime": times, "PredictedPrice": np.arange(48.0)})
    out = ann_based_trades(forecast, prices)
    assert out["Price"].tolist() == list(np.arange(48.0))
    assert (out["Action"].to_numpy() == price_based_trades(prices)["Action"].to_numpy()).all()


def test_comparison_starts_at_the_next_uk_midnight():
    trades = pd.DataFrame({"StartTime": pd.to_datetime(["2023-06-19 09:30"], utc=True)})
    assert comparison_start(trades) == pd.Timestamp("2023-06-19 23:00", tz="UTC")   # 00:00 BST


def test_perfect_foresight_beats_fixed_hours_on_the_real_data():
    prices = load_prices()
    assert price_based_trades(prices)["Rolling Cashflow"].iloc[-1] > time_based_trades(prices)["Rolling Cashflow"].iloc[-1] > 0

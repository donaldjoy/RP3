import numpy as np
import pandas as pd
import pytest

from rp3 import config
from rp3.forecasting import (ACTUAL_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN, build_features, error_metrics, predict,
                             split_and_scale, train_model)

DAY = config.PERIODS_PER_DAY


@pytest.fixture
def synthetic():
    times = pd.date_range("2024-01-01", periods=20 * DAY, freq="30min", tz="UTC", name="StartTime")
    rng = np.random.default_rng(0)
    price = 50 + 20 * np.sin(np.arange(len(times)) * 2 * np.pi / DAY) + rng.normal(0, 2, len(times))
    price[10 * DAY] = 1000.0                                  # a spike
    prices = pd.DataFrame({"StartTime": times, "SettlementDate": "x", "SettlementPeriod": 1,
                           "Price": price, "Volume": 1.0})
    generation = {name: pd.DataFrame({"StartTime": times, "Quantity": rng.random(len(times)) * 100})
                  for name in ["Wind Offshore", "Wind Onshore", "Solar"]}
    return prices, generation


def test_each_row_is_the_half_hour_being_forecast(synthetic):
    prices, generation = synthetic
    f = build_features(prices, generation)
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert f.index[0] == prices["StartTime"][7 * DAY]          # first 7 days have no 7-day-old price
    t = f.index[50]
    by_time = prices.set_index("StartTime")["Price"].clip(config.PRICE_FLOOR, config.PRICE_CAP)
    assert f.loc[t, "price_2d_ago"] == by_time[t - pd.Timedelta(days=2)]
    assert f.loc[t, "price_7d_ago"] == by_time[t - pd.Timedelta(days=7)]
    assert f.loc[t, "SolarQuantity"] == generation["Solar"].set_index("StartTime").loc[t, "Quantity"]


def test_spikes_are_clipped_for_training_but_kept_for_scoring(synthetic):
    f = build_features(*synthetic)
    spike = f[ACTUAL_COLUMN].idxmax()
    assert f.loc[spike, ACTUAL_COLUMN] == 1000.0
    assert f.loc[spike, TARGET_COLUMN] == config.PRICE_CAP


def test_split_is_chronological_and_scaled_on_training_rows(synthetic):
    f = build_features(*synthetic)
    data = split_and_scale(f)
    n = len(f)
    assert len(data.X_fit) + len(data.X_val) + len(data.X_test) == n
    assert data.val_rows.stop == data.test_rows.start == int(n * config.TRAIN_RATIO)
    assert 0 <= data.X_fit.min() and data.X_fit.max() <= 1


def test_training_learns_the_daily_shape(synthetic):
    f = build_features(*synthetic)
    data = split_and_scale(f)
    model = train_model(data, hidden_layers=(16,), learning_rate=1e-2, epochs=50, patience=50, verbose=False)
    pred = predict(model, data)
    actual = f[ACTUAL_COLUMN].iloc[data.test_rows]
    naive = np.full(len(actual), f[ACTUAL_COLUMN].iloc[: data.test_rows.start].mean())
    assert error_metrics(actual, pred)["RMSE"] < 0.5 * error_metrics(actual, naive)["RMSE"]


def test_early_stopping_keeps_the_best_validation_epoch(synthetic):
    data = split_and_scale(build_features(*synthetic))
    model = train_model(data, hidden_layers=(16,), learning_rate=1e-2, epochs=50, patience=2, verbose=False)
    assert model is not None


def test_error_metrics():
    m = error_metrics([1, 2, 3], [1, 2, 6])
    assert m["MAE"] == 1.0 and m["RMSE"] == pytest.approx(np.sqrt(3))

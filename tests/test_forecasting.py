import numpy as np
import pandas as pd
import pytest

from rp3 import config
from rp3.forecasting import FEATURE_COLUMNS, build_features, error_metrics, split_and_scale

DAY = config.PERIODS_PER_DAY


@pytest.fixture
def synthetic():
    times = pd.date_range("2024-01-01", periods=10 * DAY, freq="30min", tz="UTC", name="StartTime")
    price = np.linspace(0, 200, len(times))         # rising, inside the -35..220 band
    price[5] = 1000.0                                # an outlier
    prices = pd.DataFrame({"StartTime": times, "SettlementDate": "x", "SettlementPeriod": 1,
                           "Price": price, "Volume": 1.0})
    generation = {name: pd.DataFrame({"StartTime": times, "Quantity": 2.0})
                  for name in ["Wind Offshore", "Wind Onshore", "Solar"]}
    return prices, generation


def test_features_lags_target_and_outliers(synthetic):
    f = build_features(*synthetic, exogenous_weight=3.0)
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert f["Price"].iloc[5] == config.FILL_VALUE
    row = 3 * DAY
    assert f["price_1d_ago"].iloc[row] == f["Price"].iloc[row - DAY]
    assert f["price_6d_ago"].iloc[row] == config.FILL_VALUE          # before the data starts
    assert f["ActualTomorrowPrice"].iloc[row] == f["Price"].iloc[row + DAY]
    assert f["OffshoreQuantity"].iloc[0] == 6.0 and f["SolarQuantity"].iloc[0] == 2.0


def test_scaling_is_fitted_on_training_rows_only(synthetic):
    data = split_and_scale(build_features(*synthetic))
    assert data.train_size == int(10 * DAY * config.TRAIN_RATIO)
    assert data.X_train.min() >= 0 and data.X_train.max() <= 1
    assert data.y_test.max() > 1                                      # test prices are higher than any training price


def test_exogenous_weight_has_no_effect_after_scaling(synthetic):
    """Documents the README issue: min-max scaling cancels a constant weight."""
    rng = np.random.default_rng(0)
    for g in synthetic[1].values():
        g["Quantity"] = rng.random(len(g))
    a = split_and_scale(build_features(*synthetic, exogenous_weight=1.0))
    b = split_and_scale(build_features(*synthetic, exogenous_weight=100.0))
    np.testing.assert_allclose(a.X_train, b.X_train, atol=1e-12)


def test_error_metrics():
    m = error_metrics([1, 2, 3], [1, 2, 6])
    assert m["MAE"] == 1.0 and m["RMSE"] == pytest.approx(np.sqrt(3))

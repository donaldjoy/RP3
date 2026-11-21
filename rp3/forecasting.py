"""Day-ahead price forecast with a feed-forward neural network (ANN).

For every half-hour ``t`` the network predicts the price one day later (``t + 48``
half-hours) from:

* the price 1..6 days before ``t`` (``price_1d_ago`` .. ``price_6d_ago``)
* the day of week, hour of day and month of ``t``
* the day-ahead wind (offshore, onshore) and solar generation forecasts for ``t``

Prices outside -35..220 GBP/MWh are replaced by 73 before training (inputs and
target), and any remaining gap is filled with 73. The first 80 % of rows train
the model, the last 20 % test it; features and target are min-max scaled on the
training rows only.

TensorFlow is imported inside the functions that need it, so the data side of
this module works without it installed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from rp3 import config

LAG_COLUMNS = [f"price_{i}d_ago" for i in range(1, config.PRICE_LAG_DAYS + 1)]
GENERATION_COLUMNS = {"Wind Offshore": "OffshoreQuantity", "Wind Onshore": "OnshoreQuantity", "Solar": "SolarQuantity"}
FEATURE_COLUMNS = LAG_COLUMNS + ["day_of_week", "hour_of_day", "month_of_year",
                                 "OffshoreQuantity", "OnshoreQuantity", "SolarQuantity"]
TARGET_COLUMN = "ActualTomorrowPrice"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(prices, generation, exogenous_weight=config.EXOGENOUS_WEIGHT):
    """One row per half-hour with every feature and the target, indexed by StartTime.

    ``prices`` and each ``generation`` frame are as returned by
    :func:`rp3.data_prep.load_prices` / :func:`rp3.data_prep.load_generation`.
    """
    df = prices.set_index("StartTime").sort_index()
    in_band = df["Price"].between(config.PRICE_FLOOR, config.PRICE_CAP)
    df["Price"] = df["Price"].where(in_band, config.FILL_VALUE)

    for name, column in GENERATION_COLUMNS.items():
        quantity = generation[name].set_index("StartTime")["Quantity"].rename(column)
        df = df.join(quantity, how="outer")
    # Note: min-max scaling cancels this weight out, see README "Known issues"
    df["OffshoreQuantity"] *= exogenous_weight
    df["OnshoreQuantity"] *= exogenous_weight

    df["day_of_week"] = df.index.dayofweek
    df["hour_of_day"] = df.index.hour
    df["month_of_year"] = df.index.month
    day = config.PERIODS_PER_DAY
    for i, column in enumerate(LAG_COLUMNS, 1):
        df[column] = df["Price"].shift(day * i)
    df[TARGET_COLUMN] = df["Price"].shift(-day)
    return df.fillna(config.FILL_VALUE)


@dataclass
class Dataset:
    """Chronological train/test split, scaled to 0..1 using the training rows."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    train_size: int


def split_and_scale(features, train_ratio=config.TRAIN_RATIO):
    X = features[FEATURE_COLUMNS].to_numpy()
    y = features[[TARGET_COLUMN]].to_numpy()
    n = int(len(X) * train_ratio)
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    return Dataset(
        X_train=scaler_X.fit_transform(X[:n]), X_test=scaler_X.transform(X[n:]),
        y_train=scaler_y.fit_transform(y[:n]), y_test=scaler_y.transform(y[n:]),
        scaler_X=scaler_X, scaler_y=scaler_y, train_size=n,
    )


def error_metrics(actual, predicted):
    error = np.asarray(actual, dtype=float).ravel() - np.asarray(predicted, dtype=float).ravel()
    return {"RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(np.abs(error)))}


# ---------------------------------------------------------------------------
# Model (needs TensorFlow)
# ---------------------------------------------------------------------------
def build_model(input_dim, hidden_layers=config.HIDDEN_LAYERS, learning_rate=config.LEARNING_RATE):
    from tensorflow import keras

    model = keras.Sequential([keras.Input(shape=(input_dim,))]
                             + [keras.layers.Dense(n, activation="relu") for n in hidden_layers]
                             + [keras.layers.Dense(1, activation="linear")])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model


def train_model(data, hidden_layers=config.HIDDEN_LAYERS, learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, seed=None, verbose=1):
    """Train with early stopping on the last 20 % of the training rows."""
    from tensorflow import keras

    if seed is not None:
        keras.utils.set_random_seed(seed)
    model = build_model(data.X_train.shape[1], hidden_layers, learning_rate)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                               min_delta=1e-5, restore_best_weights=True)
    model.fit(data.X_train, data.y_train, validation_split=config.VALIDATION_SPLIT, epochs=epochs,
              batch_size=batch_size, verbose=verbose, callbacks=[early_stop])
    return model


def predict(model, data):
    """Test-set predictions in GBP/MWh."""
    return data.scaler_y.inverse_transform(model.predict(data.X_test, verbose=0)).ravel()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_forecast(prices, generation, exogenous_weight=config.EXOGENOUS_WEIGHT,
                 batch_size=config.BATCH_SIZE, seed=None):
    """Train one model; return the test rows with a ``PredictedPrice`` column, and the metrics."""
    features = build_features(prices, generation, exogenous_weight)
    data = split_and_scale(features)
    model = train_model(data, batch_size=batch_size, seed=seed)
    test = features.iloc[data.train_size:].copy()
    test["PredictedPrice"] = predict(model, data)
    return test, error_metrics(test[TARGET_COLUMN], test["PredictedPrice"])


def run_multiple_seeds(prices, generation, seeds=config.SEEDS, exogenous_weight=100, batch_size=64):
    """Train the same model once per seed to see how much the result depends on the seed.

    The defaults (weight 100, batch 64) are the settings the original multiple-model run used.
    """
    features = build_features(prices, generation, exogenous_weight)
    data = split_and_scale(features)
    test = features.iloc[data.train_size:].copy()
    metrics = {}
    for seed in seeds:
        model = train_model(data, batch_size=batch_size, seed=seed)
        test[f"PredictedPrice_{seed}"] = predict(model, data)
        metrics[seed] = error_metrics(test[TARGET_COLUMN], test[f"PredictedPrice_{seed}"])
    return test, metrics


def tune_hyperparameters(prices, generation, n_trials=50, exogenous_weight=100):
    """Optuna search over batch size, depth, width and learning rate.

    As in the original, each trial is scored on the *test* set, so the chosen
    settings have seen the test data (see README "Known issues").
    """
    import optuna

    data = split_and_scale(build_features(prices, generation, exogenous_weight))

    def objective(trial):
        n_layers = trial.suggest_int("number_of_hidden_layers", 1, 10)
        n_neurons = trial.suggest_int("number_of_neurons_per_hidden_layer", 16, 256, step=16)
        model = train_model(
            data,
            hidden_layers=(n_neurons,) * n_layers,
            learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            verbose=0,
        )
        actual = data.scaler_y.inverse_transform(data.y_test)
        return error_metrics(actual, predict(model, data))["RMSE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study

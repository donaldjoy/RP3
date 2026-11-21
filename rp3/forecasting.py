"""Day-ahead price forecast with a feed-forward neural network (scikit-learn MLPRegressor).

Each row is one half-hour ``T`` to forecast. Its features are what is known on the
evening before ``T``'s trading day, when that day's trades are chosen:

* the price 2..7 days before ``T`` (``price_2d_ago`` .. ``price_7d_ago``); the price
  1 day before is not used because part of it lies after the decision time
* the day of week, hour (UK local) and month of ``T``
* NESO's day-ahead wind (offshore, onshore) and solar forecasts for ``T``, which are
  published at about 17:45 UK time the day before

Price spikes are clipped to -35..220 GBP/MWh in the inputs and in the training
target, so a few extreme half-hours don't dominate training. The forecast is always
*scored* against the real, unclipped price.

The first 80 % of rows are for training (the last 20 % of those for early stopping
and hyperparameter tuning), and the last 20 % are the test set, used once. Features
and target are min-max scaled on the training rows only.
"""
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from rp3 import config
from rp3.data_prep import uk_time

LAG_COLUMNS = [f"price_{d}d_ago" for d in config.PRICE_LAG_DAYS]
GENERATION_COLUMNS = {"Wind Offshore": "OffshoreQuantity", "Wind Onshore": "OnshoreQuantity", "Solar": "SolarQuantity"}
FEATURE_COLUMNS = LAG_COLUMNS + ["day_of_week", "hour_of_day", "month_of_year", *GENERATION_COLUMNS.values()]
TARGET_COLUMN = "PriceClipped"   # what the network learns
ACTUAL_COLUMN = "Price"          # what it is scored against


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(prices, generation):
    """One row per half-hour to forecast, with every feature, indexed by StartTime.

    ``prices`` and each ``generation`` frame are as returned by
    :func:`rp3.data_prep.load_prices` / :func:`rp3.data_prep.load_generation`.
    The first 7 days are dropped (they have no 7-day-old price).
    """
    df = prices.set_index("StartTime").sort_index()[["SettlementDate", "SettlementPeriod", "Price"]]
    df[TARGET_COLUMN] = df["Price"].clip(config.PRICE_FLOOR, config.PRICE_CAP)

    day = config.PERIODS_PER_DAY
    for d, column in zip(config.PRICE_LAG_DAYS, LAG_COLUMNS):
        df[column] = df[TARGET_COLUMN].shift(day * d)

    local = uk_time(df.index)
    df["day_of_week"] = local.dayofweek
    df["hour_of_day"] = local.hour
    df["month_of_year"] = local.month
    for name, column in GENERATION_COLUMNS.items():
        df[column] = generation[name].set_index("StartTime")["Quantity"]
    return df.dropna(subset=FEATURE_COLUMNS)


@dataclass
class Dataset:
    """Chronological fit / validation / test split, scaled to 0..1 on the training rows."""
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    scaler_y: MinMaxScaler
    val_rows: slice
    test_rows: slice


def split_and_scale(features, train_ratio=config.TRAIN_RATIO, validation_split=config.VALIDATION_SPLIT):
    X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = features[[TARGET_COLUMN]].to_numpy(dtype=float)
    n_train = int(len(X) * train_ratio)
    n_fit = n_train - int(n_train * validation_split)
    scaler_X = MinMaxScaler().fit(X[:n_train])
    scaler_y = MinMaxScaler().fit(y[:n_train])
    Xs, ys = scaler_X.transform(X), scaler_y.transform(y).ravel()
    return Dataset(X_fit=Xs[:n_fit], y_fit=ys[:n_fit], X_val=Xs[n_fit:n_train], y_val=ys[n_fit:n_train],
                   X_test=Xs[n_train:], scaler_y=scaler_y,
                   val_rows=slice(n_fit, n_train), test_rows=slice(n_train, None))


def error_metrics(actual, predicted):
    error = np.asarray(actual, dtype=float).ravel() - np.asarray(predicted, dtype=float).ravel()
    return {"RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(np.abs(error)))}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def train_model(data, hidden_layers=config.HIDDEN_LAYERS, learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, seed=config.SEED,
                patience=config.EARLY_STOP_PATIENCE, verbose=True):
    """Train epoch by epoch and keep the weights with the lowest validation error.

    Training stops when the validation error has not improved for ``patience`` epochs.
    The validation rows come after the fitting rows in time (scikit-learn's own early
    stopping would pick them at random, letting later data leak into training).
    """
    model = MLPRegressor(hidden_layer_sizes=tuple(hidden_layers), activation="relu", solver="adam",
                         learning_rate_init=learning_rate, batch_size=batch_size, random_state=seed)
    best_loss, best_model, waited = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.partial_fit(data.X_fit, data.y_fit)
        loss = float(np.mean((model.predict(data.X_val) - data.y_val) ** 2))
        if verbose:
            print(f"\r  epoch {epoch}: validation MSE {loss:.6f}", end="", flush=True)
        if loss < best_loss - 1e-6:
            best_loss, best_model, waited = loss, copy.deepcopy(model), 0
        else:
            waited += 1
            if waited >= patience:
                break
    if verbose:
        print()
    return best_model


def predict(model, data, X=None):
    """Predictions in GBP/MWh (for the test rows unless ``X`` is given)."""
    X = data.X_test if X is None else X
    return data.scaler_y.inverse_transform(model.predict(X).reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_forecast(prices, generation, seed=config.SEED, **settings):
    """Train one model; return the test rows with a ``PredictedPrice`` column, and the metrics."""
    features = build_features(prices, generation)
    data = split_and_scale(features)
    model = train_model(data, seed=seed, **settings)
    test = features.iloc[data.test_rows].copy()
    test["PredictedPrice"] = predict(model, data)
    return test, error_metrics(test[ACTUAL_COLUMN], test["PredictedPrice"])


def run_multiple_seeds(prices, generation, seeds=config.SEEDS):
    """Train the same model once per seed, to see how much the result depends on chance."""
    features = build_features(prices, generation)
    data = split_and_scale(features)
    test = features.iloc[data.test_rows].copy()
    metrics = {}
    for seed in seeds:
        model = train_model(data, seed=seed)
        test[f"PredictedPrice_{seed}"] = predict(model, data)
        metrics[seed] = error_metrics(test[ACTUAL_COLUMN], test[f"PredictedPrice_{seed}"])
    return test, metrics


def tune_hyperparameters(prices, generation, n_trials=50, seed=config.SEED):
    """Optuna search over depth, width, learning rate and batch size.

    Each trial is scored on the validation rows (against real prices), never on the
    test set, so the test error stays an honest estimate.
    """
    import optuna

    features = build_features(prices, generation)
    data = split_and_scale(features)
    actual = features[ACTUAL_COLUMN].to_numpy()[data.val_rows]

    def objective(trial):
        n_layers = trial.suggest_int("number_of_hidden_layers", 1, 4)
        n_neurons = trial.suggest_int("number_of_neurons_per_hidden_layer", 16, 256, step=16)
        model = train_model(
            data,
            hidden_layers=(n_neurons,) * n_layers,
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            seed=seed,
            verbose=False,
        )
        return error_metrics(actual, predict(model, data, data.X_val))["RMSE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study

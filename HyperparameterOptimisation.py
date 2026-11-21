import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from sklearn.metrics import mean_squared_error
import math

# Import your existing functions if they are in ANNForecast.py.
from ANNForecast import load_and_prepare_data, create_features_and_target, split_and_scale

def main():
    # Define file paths (adjust these as needed)
    price_path = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
    offshore_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Offshore DA Gen Forecast FullDataSet.csv"
    onshore_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast FullDataSet.csv"
    solar_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Solar DA Gen Forecast FullDataSet.csv"
    
    # Load and prepare the data.
    df = load_and_prepare_data(price_path, offshore_path, onshore_path, solar_path, wind_weight=100)
    X, y = create_features_and_target(df)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, train_size = split_and_scale(X, y)

    input_dim = X_train_scaled.shape[1]

    # Define the objective function using the test set for RMSE calculation.
    def objective(trial):
        # Suggest hyperparameters.
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_hidden_layers = trial.suggest_int("number_of_hidden_layers", 1, 10)
        n_neurons = trial.suggest_int("number_of_neurons_per_hidden_layer", 16, 256, step=16)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
        
        # Build the model dynamically.
        model = Sequential()
        model.add(Dense(n_neurons, activation='relu', input_shape=(input_dim,)))
        for _ in range(n_hidden_layers - 1):
            model.add(Dense(n_neurons, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        
        # Use early stopping to prevent overfitting.
        early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-5, restore_best_weights=True)
        
        # Train the model using a validation split from the training data.
        model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_split=0.2,
            epochs=100,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop]
        )
        
        # Predict on the test set.
        y_pred_scaled = model.predict(X_test_scaled)
        # Convert predictions and actual values back to original scale.
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = scaler_y.inverse_transform(y_test_scaled)
        
        # Calculate RMSE on the entire test set.
        test_rmse = math.sqrt(mean_squared_error(y_test_original, y_pred))
        return test_rmse

    # Create an Optuna study and run the optimisation.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best RMSE: {:.4f}".format(study.best_value))
    print("Best hyperparameters: ", study.best_params)

if __name__ == "__main__":
    main()

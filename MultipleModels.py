import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Import core functions from your ANNForecast.py module
from ANNForecast import (
    load_and_prepare_data,
    create_features_and_target,
    split_and_scale,
    build_and_train_model
)

# (Optional) If you want to reuse any evaluation or plotting functions from ANNForecastVisualisation.py,
# you can import them here. In this example we implement custom plotting to overlay multiple forecasts.
# from ANNForecastVisualisation import evaluate_forecast

def main():
    # Define file paths
    price_path = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
    offshore_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Offshore DA Gen Forecast FullDataSet.csv"
    onshore_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast FullDataSet.csv"
    solar_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Solar DA Gen Forecast FullDataSet.csv"
    output_csv = r"C:\Users\joydo\Documents\RP3\PythonCode\ANNForecastMultipleModels.csv"

    # Load and prepare the data (using the same data for all models)
    df = load_and_prepare_data(price_path, offshore_path, onshore_path, solar_path, wind_weight=100)
    X, y = create_features_and_target(df)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, train_size = split_and_scale(X, y)
    
    # Get the test set as a DataFrame (this includes ActualTomorrowPrice)
    df_test = df.iloc[train_size:].copy()

    # Define the list of seeds for the experiments
    seeds = [42, 100, 7]
    
    # Dictionaries to hold each model's predictions and metrics
    predictions = {}  # keys will be like "PredictedPrice_42", etc.
    metrics = {}      # keys will be the seed; values a dict with RMSE and MAE
    
    # Loop over each seed: set random seeds, train model, predict, and compute metrics.
    for seed in seeds:
        # Set random seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Build and train the model (all hyperparameters are identical)
        model, history = build_and_train_model(
            X_train_scaled, y_train_scaled,
            input_dim=X_train_scaled.shape[1],
            epochs=100, batch_size=64
        )
        
        # Generate predictions on the test set
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Compute RMSE and MAE
        actual = df_test['ActualTomorrowPrice'].values
        rmse = np.sqrt(np.mean((actual - y_pred) ** 2))
        mae = np.mean(np.abs(actual - y_pred))
        metrics[seed] = {'RMSE': rmse, 'MAE': mae}
        
        # Store predictions with a unique column name (based on the seed)
        predictions[f'PredictedPrice_{seed}'] = y_pred
        
        # Print individual model metrics
        print(f"Seed {seed} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    # Calculate and print the average MAE across models
    avg_mae = np.mean([metrics[s]['MAE'] for s in seeds])
    print(f"Average MAE across models: {avg_mae:.3f}")
    
    # Add each model's predictions as new columns to the df_test DataFrame
    for col_name, preds in predictions.items():
        df_test[col_name] = preds
    
    # Save the combined predictions and actual values to CSV
    df_test.to_csv(output_csv, index=True)
    print(f"Multiple models forecast saved to {output_csv}")
    
    # Plot the forecasts: actual values plus each model's predictions on one graph
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test['ActualTomorrowPrice'], label='Actual Tomorrow Price', color='black', linewidth=2)
    
    # Define a list of colors for the different models
    colors = ['red', 'blue', 'green']
    for i, seed in enumerate(seeds):
        col = f'PredictedPrice_{seed}'
        plt.plot(df_test.index, df_test[col], label=f'Predicted (seed {seed})', color=colors[i])
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Electricity Price Forecast vs Actual (Multiple Models)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

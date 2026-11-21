import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_and_prepare_data(price_path, offshore_path, onshore_path, solar_path, wind_weight=100):
    """Load and merge datasets; apply wind weight; filter out price outliers; add time features and create target."""
    df = pd.read_csv(price_path, parse_dates=['StartTime'])
    df.sort_values('StartTime', inplace=True)
    df.set_index('StartTime', inplace=True)
    print("Original dataset size:", df.shape)
    df['Price'] = df['Price'].where((df['Price'] >= -35) & (df['Price'] <= 220), 73)
   
    df_offshore = pd.read_csv(offshore_path, parse_dates=['StartTime'])
    df_offshore.sort_values('StartTime', inplace=True)
    df_offshore.set_index('StartTime', inplace=True)
    df_offshore.rename(columns={'Quantity': 'OffshoreQuantity'}, inplace=True)
    df = df.join(df_offshore[['OffshoreQuantity']], how='outer')
   
    df_onshore = pd.read_csv(onshore_path, parse_dates=['StartTime'])
    df_onshore.sort_values('StartTime', inplace=True)
    df_onshore.set_index('StartTime', inplace=True)
    df_onshore.rename(columns={'Quantity': 'OnshoreQuantity'}, inplace=True)
    df = df.join(df_onshore[['OnshoreQuantity']], how='outer')
   
    df_solar = pd.read_csv(solar_path, parse_dates=['StartTime'])
    df_solar.sort_values('StartTime', inplace=True)
    df_solar.set_index('StartTime', inplace=True)
    df_solar.rename(columns={'Quantity': 'SolarQuantity'}, inplace=True)
    df = df.join(df_solar[['SolarQuantity']], how='outer')
    print("After merging exogenous datasets:", df.shape)

    df['OffshoreQuantity'] *= wind_weight
    df['OnshoreQuantity'] *= wind_weight
   
    #df['OffshoreQuantity'] = df['OffshoreQuantity'].shift(-48)
    #df['OnshoreQuantity'] = df['OnshoreQuantity'].shift(-48)
    #df['SolarQuantity'] = df['SolarQuantity'].shift(-48)
    #print("After shifting exogenous datasets by 48:", df.shape)

    df['day_of_week'] = df.index.dayofweek
    df['hour_of_day'] = df.index.hour
    df['month_of_year'] = df.index.month  # Added MonthOfYear feature
   
    for i in range(1, 7):
        df[f'price_{i}d_ago'] = df['Price'].shift(48 * i)
   
    df['ActualTomorrowPrice'] = df['Price'].shift(-48)
    df.fillna(73, inplace=True)  # Use forward-fill if continuity is required
   
    return df

def create_features_and_target(df):
    """Extract feature matrix X and target vector y."""
    feature_cols = (
        [f'price_{i}d_ago' for i in range(1, 7)] +
        ['day_of_week', 'hour_of_day', 'month_of_year', 'OffshoreQuantity', 'OnshoreQuantity', 'SolarQuantity']
    )
    X = df[feature_cols].values
    y = df['ActualTomorrowPrice'].values.reshape(-1, 1)
    print("After creating features and dropping NaNs:", df.shape)
    return X, y

def split_and_scale(X, y, train_ratio=0.8):
    """Split into train/test sets and scale features and target."""
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
   
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
   
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled  = scaler_y.transform(y_test)
    print("Train size:", train_size)
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, train_size

def build_and_train_model(X_train_scaled, y_train_scaled, input_dim, epochs=100, batch_size=256):
    """Build and train the ANN model."""
    model = Sequential([
        Dense(160, activation='relu', input_shape=(input_dim,)),
        Dense(160, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.00013504016404346537), loss='mean_squared_error')
   
    early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-5, restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train_scaled,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[early_stop])
    return model, history

def forecast_and_save(model, X_test_scaled, scaler_y, df, train_size, output_csv):
    """Predict on the test set, compute metrics, and save forecast CSV."""
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
   
    df_test = df.iloc[train_size:].copy()
    df_test['PredictedPrice'] = y_pred
    # df_test already contains 'ActualTomorrowPrice'
   
    rmse = np.sqrt(mean_squared_error(df_test['ActualTomorrowPrice'], y_pred))
    mae = mean_absolute_error(df_test['ActualTomorrowPrice'], y_pred)
    print(f"High Precision Test Set RMSE: {rmse:.3f}")
    print(f"High Precision Test Set MAE:  {mae:.3f}")
   
    df_test.to_csv(output_csv, index=True)
    print(f"Forecast saved to {output_csv}")
    return df_test, rmse, mae

def main():
    """Run the complete ANN forecast pipeline."""
    price_path = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
    offshore_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Offshore DA Gen Forecast FullDataSet.csv"
    onshore_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast FullDataSet.csv"
    solar_path = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Solar DA Gen Forecast FullDataSet.csv"
    output_csv = r"C:\Users\joydo\Documents\RP3\PythonCode\ANNForecast.csv"
   
    df = load_and_prepare_data(price_path, offshore_path, onshore_path, solar_path, wind_weight=2.0)
    X, y = create_features_and_target(df)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, train_size = split_and_scale(X, y)
    model, history = build_and_train_model(X_train_scaled, y_train_scaled, input_dim=X_train_scaled.shape[1])
    forecast_and_save(model, X_test_scaled, scaler_y, df, train_size, output_csv)

if __name__ == "__main__":
    main()

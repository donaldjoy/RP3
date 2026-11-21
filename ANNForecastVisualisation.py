import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import re

# Configure global font sizes before creating any figures
plt.rcParams.update({
    'font.size': 15,          # Global font size
    'axes.labelsize': 25,     # Axis label font size
    'xtick.labelsize': 15,    # X-axis tick label font size
    'ytick.labelsize': 20,    # Y-axis tick label font size
    'legend.fontsize': 15     # Legend font size
})

def sanitize_filename(filename):
    """
    Replace characters that are not allowed in Windows filenames.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Define the folder where images will be saved
SAVE_FOLDER = r"C:\Users\joydo\Documents\RP3\PythonCode\Images"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def load_forecast(csv_file):
    """Load forecast CSV with datetime index."""
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df

def evaluate_forecast(df):
    """Compute and print RMSE and MAE using ActualTomorrowPrice."""
    y_true = df['ActualTomorrowPrice'].values
    y_pred = df['PredictedPrice'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("Evaluation Metrics:")
    print(f"Test Set RMSE: {rmse:.2f}")
    print(f"Test Set MAE:  {mae:.2f}")
    return rmse, mae

def plot_actual_vs_predicted(df):
    """Plot Actual vs Predicted Prices using ActualTomorrowPrice."""
    # Define the intended title (for file naming)
    intended_title = 'Electricity Price Forecast vs Actual (Test Set)'
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['ActualTomorrowPrice'], label='Actual Price', color='green')
    plt.plot(df.index, df['PredictedPrice'], label='Forecasted Price', color='red')
    # Title removed from display
    plt.xlabel('Time')
    plt.ylabel('Price (GBP/MWh)')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure with the intended title as filename
    fig.savefig(os.path.join(SAVE_FOLDER, sanitize_filename(intended_title) + ".png"), dpi=300)
    plt.show()

def plot_detailed_forecast(df, title, start_date, end_date):
    """Plot detailed forecast for a specified period using ActualTomorrowPrice."""
    df_period = df.loc[start_date:end_date]
    date_format = mdates.DateFormatter("%d-%m-%y %H:%M")
    hour_locator = mdates.HourLocator(interval=12)
    
    # Define the intended title (for file naming)
    intended_title = f'Forecast vs Actual Price {title}'
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df_period.index, df_period['ActualTomorrowPrice'], label='Actual Price', color='green')
    plt.plot(df_period.index, df_period['PredictedPrice'], label='Forecasted Price', color='red')
    # Title removed from display
    plt.xlabel('Time')
    plt.ylabel('Price (GBP/MWh)')
    ax = plt.gca()
    ax.xaxis.set_major_locator(hour_locator)
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure using the intended title as filename
    fig.savefig(os.path.join(SAVE_FOLDER, sanitize_filename(intended_title) + ".png"), dpi=300)
    plt.show()

def main():
    """Run evaluation and visualisation on ANNForecast.csv."""
    csv_file = r"C:\Users\joydo\Documents\RP3\PythonCode\ANNForecast.csv"
    df = load_forecast(csv_file)
    evaluate_forecast(df)
    plot_actual_vs_predicted(df)
    
    date_ranges = {
        "Winter (1st to 3rd Dec 2024)": ('2024-12-01', '2024-12-03'),
        "Summer (1st to 3rd Jun 2024)": ('2024-06-01', '2024-06-03')
    }
    for title, (start_date, end_date) in date_ranges.items():
        plot_detailed_forecast(df, title, start_date, end_date)

if __name__ == "__main__":
    main()

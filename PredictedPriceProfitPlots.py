import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import re

# Configure global font sizes before creating any figures
plt.rcParams.update({
    'font.size': 15,          # Global font size
    'axes.labelsize': 25,     # Axis label font size
    'xtick.labelsize': 20,    # X-axis tick label font size
    'ytick.labelsize': 20,    # Y-axis tick label font size
    'legend.fontsize': 15     # Legend font size
})

def sanitize_filename(filename):
    """
    Replace characters that are not allowed in Windows filenames.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def analyse_trades(price_based_csv, time_based_csv, predicted_price_based_trades):
    # Define the folder where images will be saved
    save_folder = r"C:\Users\joydo\Documents\RP3\PythonCode\Images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------------------------------------------------------------
    # 1) Read both CSVs and parse times
    # ------------------------------------------------------------------
    # Define the datetime range

    start_time = pd.Timestamp("2023-06-19 09:30:00", tz="UTC")
    
    # Read time-based trades
    df_time = pd.read_csv(time_based_csv)
    if not pd.api.types.is_datetime64_any_dtype(df_time['StartTime']):
        df_time['StartTime'] = pd.to_datetime(df_time['StartTime'], utc=True, errors='coerce')
    df_time.sort_values(by='StartTime', inplace=True, ignore_index=True)
    df_time = df_time[(df_time['StartTime'] >= start_time)]

    # Read price-based trades
    df_price = pd.read_csv(price_based_csv)
    if not pd.api.types.is_datetime64_any_dtype(df_price['StartTime']):
        df_price['StartTime'] = pd.to_datetime(df_price['StartTime'], utc=True, errors='coerce')
    df_price.sort_values(by='StartTime', inplace=True, ignore_index=True)
    df_price = df_price[(df_price['StartTime'] >= start_time)]

    # Read predicted price-based trades
    df_predicted_price = pd.read_csv(predicted_price_based_trades)
    if not pd.api.types.is_datetime64_any_dtype(df_predicted_price['StartTime']):
        df_predicted_price['StartTime'] = pd.to_datetime(df_predicted_price['StartTime'], utc=True, errors='coerce')
    df_predicted_price.sort_values(by='StartTime', inplace=True, ignore_index=True)
    df_predicted_price = df_predicted_price[(df_predicted_price['StartTime'] >= start_time)]

    # ------------------------------------------------------------------
    # 2) Plot Rolling Cashflow vs. StartTime for thrice datasets (in GBP millions)
    # ------------------------------------------------------------------
    title_fig1 = "Rolling Cashflow Over Time for Predicted Price"
    fig1 = plt.figure(figsize=(10, 5))
    # Divide rolling cashflow by 1e6 so the y-axis is in millions
    plt.plot(df_time['StartTime'], (df_time['Cost'] + df_time['Revenue']).cumsum()/1e6, label='Time Based', color='blue')
    plt.plot(df_price['StartTime'], (df_price['Cost'] + df_price['Revenue']).cumsum()/1e6, label='Price Based', color='orange')
    plt.plot(df_predicted_price['StartTime'], (df_predicted_price['Cost'] + df_predicted_price['Revenue']).cumsum()/1e6, label='ANN Based', color='red')
    plt.xlabel("Time")
    plt.ylabel("Profit (GBP Millions)")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    fig1.savefig(os.path.join(save_folder, sanitize_filename(title_fig1) + ".png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------------
    # 3) Total Profit per Year (bar chart) for thrice datasets (in GBP millions)
    #    Profit = sum(Cost + Revenue) for each year
    # ------------------------------------------------------------------
    # Process price-based trades
    df_price['year'] = df_price['StartTime'].dt.year
    df_price['PnL'] = df_price['Cost'] + df_price['Revenue']
    yearly_profit_price = df_price.groupby('year')['PnL'].sum() / 1e6  # Convert to millions

    # Process time-based trades
    df_time['year'] = df_time['StartTime'].dt.year
    df_time['PnL'] = df_time['Cost'] + df_time['Revenue']
    yearly_profit_time = df_time.groupby('year')['PnL'].sum() / 1e6  # Convert to millions

    # Process predicted price-based trades
    df_predicted_price['year'] = df_predicted_price['StartTime'].dt.year
    df_predicted_price['PnL'] = df_predicted_price['Cost'] + df_predicted_price['Revenue']
    yearly_profit__predicted_price = df_predicted_price.groupby('year')['PnL'].sum() / 1e6  # Convert to millions

    # Combine the two into one DataFrame for side-by-side bar chart
    profit_df = pd.DataFrame({
        'Time Based': yearly_profit_time,
        'Price Based': yearly_profit_price,
        'ANN Based': yearly_profit__predicted_price
    }).fillna(0)

    title_fig2 = "Total Profit per Year for Predicted Price"
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    profit_df.plot(kind='bar', ax=ax2)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Profit (GBP Millions)")
    plt.tight_layout()
    fig2.savefig(os.path.join(save_folder, sanitize_filename(title_fig2) + ".png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    time_based_trades = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - TimeBasedTrades.csv"
    price_based_trades = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - PriceBasedTrades.csv"
    predicted_price_based_trades = r"C:\Users\joydo\Documents\RP3\PythonCode\2023-2025 - PredictedPriceBasedTrades.csv"
    analyse_trades(price_based_trades, time_based_trades, predicted_price_based_trades)

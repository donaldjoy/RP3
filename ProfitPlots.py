import pandas as pd
import matplotlib.pyplot as plt
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

def analyse_trades(price_based_csv, time_based_csv):
    # Define the folder where images will be saved
    save_folder = r"C:\Users\joydo\Documents\RP3\PythonCode\Images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------------------------------------------------------------
    # 1) Read both CSVs and parse times
    # ------------------------------------------------------------------
    
    # Read time-based trades
    df_time = pd.read_csv(time_based_csv)
    if not pd.api.types.is_datetime64_any_dtype(df_time['StartTime']):
        df_time['StartTime'] = pd.to_datetime(df_time['StartTime'], utc=True, errors='coerce')
    df_time.sort_values(by='StartTime', inplace=True, ignore_index=True)

    # Read price-based trades
    df_price = pd.read_csv(price_based_csv)
    if not pd.api.types.is_datetime64_any_dtype(df_price['StartTime']):
        df_price['StartTime'] = pd.to_datetime(df_price['StartTime'], utc=True, errors='coerce')
    df_price.sort_values(by='StartTime', inplace=True, ignore_index=True)

    # ------------------------------------------------------------------
    # 2) Plot Rolling Cashflow vs. StartTime for both datasets (in GBP millions)
    # ------------------------------------------------------------------
    title_fig1 = "Rolling Cashflow Over Time"
    fig1 = plt.figure(figsize=(10, 5))
    # Divide rolling cashflow by 1e6 so the y-axis is in millions
    plt.plot(df_time['StartTime'], df_time['Rolling Cashflow'] / 1e6, label='Time Based', color='blue')
    plt.plot(df_price['StartTime'], df_price['Rolling Cashflow'] / 1e6, label='Price Based', color='orange')
    plt.xlabel("Time")
    plt.ylabel("Profit (GBP Millions)")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(save_folder, sanitize_filename(title_fig1) + ".png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------------
    # 3) Total Profit per Year (bar chart) for both datasets (in GBP millions)
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

    # Combine the two into one DataFrame for side-by-side bar chart
    profit_df = pd.DataFrame({
        'Time Based': yearly_profit_time,
        'Price Based': yearly_profit_price
    }).fillna(0)

    title_fig2 = "Total Profit per Year"
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    profit_df.plot(kind='bar', ax=ax2)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Profit (GBP Millions)")
    plt.tight_layout()
    fig2.savefig(os.path.join(save_folder, sanitize_filename(title_fig2) + ".png"), dpi=300)
    plt.show()

    # ------------------------------------------------------------------
    # 4) Settlement Period vs. Number of Imports/Exports (price-based only)
    # ------------------------------------------------------------------
    if 'Action' not in df_price.columns:
        print("No 'Action' column found in price-based trades. Cannot plot the import/export bar chart.")
        return

    # Define settlementPeriod for each row: 1 to 48 (e.g. 00:00 -> 1, 00:30 -> 2, etc.)
    df_price['settlementPeriod'] = df_price['StartTime'].apply(
        lambda dt: dt.hour * 2 + (dt.minute // 30) + 1
    )

    # Count occurrences for imports (Action == 1) and exports (Action == 2)
    import_counts = df_price[df_price['Action'] == 1].groupby('settlementPeriod').size()
    export_counts = df_price[df_price['Action'] == 2].groupby('settlementPeriod').size()

    # Ensure all periods 1..48 are present
    all_periods = range(1, 49)
    import_counts = import_counts.reindex(all_periods, fill_value=0)
    export_counts = export_counts.reindex(all_periods, fill_value=0)

    x = np.arange(1, 49)
    width = 0.4

    title_fig5 = "Count of Imports & Exports by Settlement Period"
    fig5 = plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, import_counts, width, color='green', label='Imports')
    plt.bar(x + width/2, export_counts, width, color='red', label='Exports')
    plt.xlabel("Settlement Period")
    plt.ylabel("Occurances")
    plt.xticks(np.arange(0, 49, 4), np.arange(0, 49, 4))
    plt.legend()
    plt.tight_layout()
    fig5.savefig(os.path.join(save_folder, sanitize_filename(title_fig5) + ".png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    time_based_trades = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - TimeBasedTrades.csv"
    price_based_trades = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - PriceBasedTrades.csv"
    analyse_trades(price_based_trades, time_based_trades)

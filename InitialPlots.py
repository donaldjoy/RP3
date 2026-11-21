import pandas as pd
import matplotlib.pyplot as plt
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

def plot_full_dataset(csv_path):
    # Define the folder where images will be saved
    save_folder = r"C:\Users\joydo\Documents\RP3\PythonCode\Images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------------------------
    # 1) Load the data
    # ------------------------------
    df = pd.read_csv(csv_path)
    if not pd.api.types.is_datetime64_any_dtype(df['StartTime']):
        df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True, errors='coerce')
    df.sort_values(by='StartTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------
    # 2) Plot Price vs. Time (top) and Volume vs. Time (bottom)
    # ------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    # Store the intended title and then do not display it on the plot
    title_fig1 = "Price (GBP/MWh) and Volume (MWh) from 2016 to 2025"
    
    # Top subplot: Price
    axes[0].plot(df['StartTime'], df['Price'], color='blue', label='Price')
    axes[0].set_ylabel("Price (GBP/MWh)")
    axes[0].legend(loc='upper right')

    # Bottom subplot: Volume
    axes[1].plot(df['StartTime'], df['Volume'], color='green', label='Volume')
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Volume (MWh)")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    # Save the figure using the title as the file name
    fig.savefig(os.path.join(save_folder, sanitize_filename(title_fig1) + ".png"), dpi=300)
    plt.show()

    # ------------------------------
    # 3) Average Price by Settlement Period with highlighted areas
    # ------------------------------
    if 'SettlementPeriod' not in df.columns:
        print("No 'SettlementPeriod' column found. Cannot plot the daily average curve.")
    else:
        daily_avg = df.groupby('SettlementPeriod')[['Price']].mean()

        fig2, ax1 = plt.subplots(figsize=(12, 6))
        title_fig2 = "Average Daily Price (GBP/MWh) by Settlement Period"
        # Note: We no longer add the title to the figure

        ax1.set_xlabel("Settlement Period")
        ax1.set_ylabel("Average Price (GBP/MWh)", color='blue')
        ax1.plot(daily_avg.index, daily_avg['Price'], color='blue', marker='o', label='Average Price')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        # Highlight settlement periods 14 to 20 and 33.5 to 39.5 in yellow
        ax1.axvspan(14, 20, color='yellow', alpha=1)
        ax1.axvspan(33.5, 39.5, color='yellow', alpha=1)

        plt.tight_layout()
        fig2.savefig(os.path.join(save_folder, sanitize_filename(title_fig2) + ".png"), dpi=300)
        plt.show()

    # --------------------------------------------------------
    # 5) Average Price vs. Settlement Period by day of week
    # --------------------------------------------------------
    # Add day-of-week column (Monday, Tuesday, ...)
    df['day_of_week'] = df['StartTime'].dt.day_name()
    grouped = df.groupby(['day_of_week', 'SettlementPeriod'])['Price'].mean().reset_index()
    pivoted = grouped.pivot(index='SettlementPeriod', columns='day_of_week', values='Price')
    # Reorder columns to Monday..Sunday
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivoted = pivoted[day_order]
    # Rename the legend title
    pivoted.columns.name = "Day of Week"

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    title_fig4 = "Average Price by Settlement Period for Each Day of Week"
    pivoted.plot(ax=ax4, marker='o')
    ax4.set_xlabel("Settlement Period")
    ax4.set_ylabel("Average Price (GBP/MWh)")

    plt.tight_layout()
    fig4.savefig(os.path.join(save_folder, sanitize_filename(title_fig4) + ".png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    csv_file = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
    plot_full_dataset(csv_file)

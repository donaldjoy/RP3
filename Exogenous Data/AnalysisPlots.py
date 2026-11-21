import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------- Global Configuration ---------------------
SAVE_FOLDER = r"C:\Users\joydo\Documents\RP3\PythonCode\Images"
CSV_FILES = [
    r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Offshore DA Gen Forecast FullDataSet.csv",
    r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast FullDataSet.csv",
    r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Solar DA Gen Forecast FullDataSet.csv"
]
LEGEND_LABELS = {
    "Wind Offshore DA Gen Forecast FullDataSet": "Wind Offshore Forecast",
    "Wind Onshore DA Gen Forecast FullDataSet": "Wind Onshore Forecast",
    "Solar DA Gen Forecast FullDataSet": "Solar Forecast"
}

plt.rcParams.update({
    'font.size': 15,          # Global font size
    'axes.labelsize': 25,     # Axis label font size
    'xtick.labelsize': 20,    # X-axis tick label font size
    'ytick.labelsize': 20,    # Y-axis tick label font size
    'legend.fontsize': 15     # Legend font size
})


# --------------------- Helper Functions ---------------------
def sanitise_filename(filename: str) -> str:
    """
    Replace characters that are not allowed in Windows filenames.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def create_save_folder(folder: str) -> None:
    """
    Create the save folder if it does not exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def read_and_prepare_data(file: str) -> dict:
    """
    Read a CSV file, convert date columns, and add derived columns.
    Returns a dictionary with:
        - 'df': the processed DataFrame.
        - 'category': the raw category name (from the filename).
        - 'label': the user-friendly legend label.
    """
    df = pd.read_csv(file)
    df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)
    df.sort_values(by='StartTime', inplace=True)
    
    # Extract category from filename
    category = os.path.basename(file).replace(".csv", "")
    label = LEGEND_LABELS.get(category, category)
    
    # Add extra columns for aggregations
    df['Year'] = df['StartTime'].dt.year
    df['SettlementPeriod'] = df['StartTime'].dt.hour * 2 + (df['StartTime'].dt.minute // 30) + 1
    df['Month'] = df['StartTime'].dt.month
    
    return {"df": df, "category": category, "label": label}


def create_line_plot(df: pd.DataFrame, label: str, title: str) -> None:
    """
    Create and save the original line plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['StartTime'], df['Quantity'], color='b', label=label)
    ax.set_xlabel("Year")
    ax.set_ylabel("Quantity (MWh)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    fig.savefig(os.path.join(SAVE_FOLDER, sanitise_filename(title) + ".png"), dpi=300)
    plt.show()


def aggregate_data(df: pd.DataFrame) -> dict:
    """
    Compute various aggregations from the DataFrame.
    Returns a dictionary with:
        - 'year': total Quantity per Year.
        - 'settlement': average Quantity per Settlement Period.
        - 'month': average Quantity per Month.
    """
    year_series = df.groupby('Year')['Quantity'].sum()
    settlement_avg = df.groupby('SettlementPeriod')['Quantity'].mean().reindex(range(1, 49))
    month_avg = df.groupby('Month')['Quantity'].mean().reindex(range(1, 13))
    
    return {"year": year_series, "settlement": settlement_avg, "month": month_avg}


def plot_aggregated_year_data(agg_year_data: dict) -> None:
    """
    Create and save a bar chart of aggregated year data.
    Excludes years 2016 and 2025.
    """
    df_agg = pd.DataFrame(agg_year_data).sort_index()
    df_agg = df_agg.drop([2016, 2025], axis=0, errors='ignore')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df_agg.plot(kind='bar', ax=ax)
    
    ax.set_title("")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Quantity (Million MWh)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}'))
    ax.legend(title="Category")
    ax.grid(True)
    fig.tight_layout()
    
    fig.savefig(os.path.join(SAVE_FOLDER, sanitise_filename("Aggregated DA Gen Forecast Quantity per Year") + ".png"), dpi=300)
    plt.show()


def plot_settlement_data(avg_settlement_data: dict) -> None:
    """
    Create and save a line plot of average quantity per settlement period.
    Uses friendly legend labels.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for raw_category, series in avg_settlement_data.items():
        friendly_label = LEGEND_LABELS.get(raw_category, raw_category)
        ax.plot(series.index, series.values, marker='o', label=friendly_label)
    
    ax.set_xlabel("Settlement Period")
    ax.set_ylabel("Average Quantity (MWh)")
    ax.legend(title="Category")
    ax.grid(True)
    fig.tight_layout()
    
    fig.savefig(os.path.join(SAVE_FOLDER, sanitise_filename("Average DA Gen Forecast Quantity per Settlement Period") + ".png"), dpi=300)
    plt.show()


def plot_month_data(avg_month_data: dict) -> None:
    """
    Create and save a bar chart of average quantity per month.
    Renames columns to use friendly legend labels.
    """
    avg_month_df = pd.DataFrame(avg_month_data)
    avg_month_df.index.name = "Month"
    avg_month_df.sort_index(inplace=True)
    avg_month_df.rename(columns=LEGEND_LABELS, inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_month_df.plot(kind='bar', ax=ax)
    
    ax.set_title("")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Quantity (MWh)")
    ax.legend(title="Category")
    ax.grid(True)
    fig.tight_layout()
    
    fig.savefig(os.path.join(SAVE_FOLDER, sanitise_filename("Average DA Gen Forecast Quantity per Month") + ".png"), dpi=300)
    plt.show()


# --------------------- Main Workflow ---------------------
def main():
    create_save_folder(SAVE_FOLDER)
    
    # Dictionaries for storing aggregated results
    agg_year_data = {}
    avg_settlement_data = {}
    avg_month_data = {}
    
    # Process each CSV file
    for file in CSV_FILES:
        data = read_and_prepare_data(file)
        df = data["df"]
        category = data["category"]
        label = data["label"]
        
        # Create the original line plot
        create_line_plot(df, label, title=category)
        
        # Compute aggregations
        agg = aggregate_data(df)
        agg_year_data[label] = agg["year"]
        avg_settlement_data[category] = agg["settlement"]
        avg_month_data[category] = agg["month"]
    
    # Create the aggregated plots
    plot_aggregated_year_data(agg_year_data)
    plot_settlement_data(avg_settlement_data)
    plot_month_data(avg_month_data)


if __name__ == "__main__":
    main()

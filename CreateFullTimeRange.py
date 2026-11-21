import pandas as pd
from datetime import datetime, timedelta
import pytz

def create_full_timeline_df():
    """
    Create a DataFrame of timestamps in 30-minute steps going forward
    from 2023-01-01T00:00:00Z to 2025-01-20T00:00:00Z, stored in memory only.
    """
    start = datetime(2023, 6, 19, 9, 30, 0, tzinfo=pytz.UTC)
    end = datetime(2025, 1, 19, 0, 0, 0, tzinfo=pytz.UTC)

    timestamps = []
    current_time = start
    while current_time <= end:
        iso_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        timestamps.append(iso_str)
        current_time += timedelta(minutes=30)

    df_full = pd.DataFrame({"StartTime": timestamps})
    return df_full

def merge_data_in_memory(df_full, partial_csv, merged_csv):
    """
    Merge the partial data onto the full timeline DataFrame in memory by matching on 'StartTime'.
    Finally, export the merged DataFrame to a new CSV.
    """
    # Read your existing partial data, parsing 'StartTime' as datetime
    df_partial = pd.read_csv(partial_csv, parse_dates=['StartTime'])
    
    # Convert df_full 'StartTime' to datetime
    df_full['StartTime'] = pd.to_datetime(df_full['StartTime'])
    
    # Merge the data on 'StartTime'
    df_merged = pd.merge(
        df_full,
        df_partial[['StartTime', 'SettlementDate', 'SettlementPeriod', 'PredictedPrice', 'ActualTomorrowPrice', 'Volume']],
        on='StartTime',
        how='left'
    )
    
    # Optionally, reformat 'StartTime' back to your desired string format if needed:
    df_merged['StartTime'] = df_merged['StartTime'].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Write the merged data to CSV
    df_merged.to_csv(merged_csv, index=False)
    print(f"Merged data CSV created: {merged_csv}")

if __name__ == "__main__":
    # 1) Generate the full timeline in memory
    df_full = create_full_timeline_df()

    # 2) Merge with partial data and output the final CSV
    partial_csv_path = r"C:\Users\joydo\Documents\RP3\PythonCode\MissingTimePriceANNForecast.csv"
    merged_csv_path  = r"C:\Users\joydo\Documents\RP3\PythonCode\MissingPriceANNForecast.csv"

    merge_data_in_memory(df_full, partial_csv_path, merged_csv_path)

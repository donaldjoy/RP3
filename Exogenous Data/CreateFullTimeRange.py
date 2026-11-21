import pandas as pd
from datetime import datetime, timedelta
import pytz

def create_full_timeline_df():
    """
    Create a DataFrame of timestamps in 30-minute steps from
    2016-09-19 00:00:00+00:00 to 2025-01-20 00:00:00+00:00.
    The timestamps are formatted as "YYYY-MM-DD HH:MM:SS+00:00".
    """
    start = datetime(2016, 9, 19, 0, 0, 0, tzinfo=pytz.UTC)
    end = datetime(2025, 1, 20, 0, 0, 0, tzinfo=pytz.UTC)

    timestamps = []
    current_time = start
    while current_time <= end:
        # Using isoformat with a space separator to get "YYYY-MM-DD HH:MM:SS+00:00"
        ts_str = current_time.isoformat(' ')
        timestamps.append(ts_str)
        current_time += timedelta(minutes=30)

    df_full = pd.DataFrame({"StartTime": timestamps})
    return df_full

def merge_data_in_memory(df_full, partial_csv, merged_csv):
    """
    Merge the partial data (with PublishTime, ProcessType, BusinessType, PsrType,
    SettlementDate, SettlementPeriod, Quantity) onto the full timeline DataFrame
    by matching on 'StartTime'. All StartTime values are converted to the format
    "YYYY-MM-DD HH:MM:SS+00:00". Finally, the output columns are reordered.
    """
    # Read the partial CSV
    df_partial = pd.read_csv(partial_csv)
        
    # Convert the StartTime column in df_partial to datetime (with UTC)
    # and then reformat to the desired string format.
    df_partial['StartTime'] = pd.to_datetime(df_partial['StartTime'], utc=True)\
                                  .apply(lambda dt: dt.isoformat(' '))
    
    # Remove any duplicate StartTime rows if present
    df_partial = df_partial.drop_duplicates(subset=['StartTime'])

    # Merge the two DataFrames on 'StartTime' (left join to keep all full timeline rows)
    df_merged = pd.merge(
        df_full,
        df_partial[['PublishTime', 'ProcessType', 'BusinessType', 'PsrType', 'StartTime', 'SettlementDate', 'SettlementPeriod', 'Quantity']],
        on='StartTime',
        how='left'
    )
    
    # Reorder columns to:
    # PublishTime, ProcessType, BusinessType, PsrType, StartTime, SettlementDate, SettlementPeriod, Quantity
    df_merged = df_merged[['PublishTime', 'ProcessType', 'BusinessType', 'PsrType', 'StartTime', 'SettlementDate', 'SettlementPeriod', 'Quantity']]
    
    # Write the merged DataFrame to CSV
    df_merged.to_csv(merged_csv, index=False)
    print(f"Merged data CSV created: {merged_csv}")

if __name__ == "__main__":
    # 1) Generate the full timeline DataFrame (common for all merges)
    df_full = create_full_timeline_df()

    # 2) Dictionary containing the partial and merged CSV file paths
    data_files = {
        "Solar DA Gen Forecast": {
            "partial": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Solar DA Gen Forecast.csv",
            "merged": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Solar DA Gen Forecast FullTimeRange.csv"
        },
        "Wind Offshore DA Gen Forecast": {
            "partial": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Offshore DA Gen Forecast.csv",
            "merged": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Offshore DA Gen Forecast FullTimeRange.csv"
        },
        "Wind Onshore DA Gen Forecast": {
            "partial": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast.csv",
            "merged": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast FullTimeRange.csv"
        }
    }

    # 3) Loop over the dictionary and merge each CSV with the full timeline
    for key, paths in data_files.items():
        print(f"Processing {key}...")
        merge_data_in_memory(df_full, paths["partial"], paths["merged"])

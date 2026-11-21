import pandas as pd

def process_full_dataset(input_csv, output_csv):
    """
    1) Read the CSV (with potential missing SettlementDate/Period/PredictedPrice/ActualTomorrowPrice/Volume).
    2) Derive SettlementDate and SettlementPeriod from StartTime.
    3) Fill missing Price and Volume by averaging the previous and next valid entries.
    4) Write a new CSV with all columns complete.
    """
    # Read the data
    df = pd.read_csv(input_csv)
    
    # Convert StartTime into a proper datetime (assuming UTC times ending with 'Z')
    # e.g., "2023-01-01T00:00:00Z" -> a tz-aware Timestamp
    # If your 'StartTime' is already a string without the trailing 'Z', adjust the format as needed.
    df['dt'] = pd.to_datetime(df['StartTime'], utc=True, format="%Y-%m-%dT%H:%M:%SZ")

    # Sort in ascending chronological order so that forward fill (ffill) is truly "earlier to later"
    df.sort_values(by='dt', inplace=True, ignore_index=True)

    # ----------------------------------------------------------------------
    # 1) Derive 'SettlementDate'
    #    We'll just take the date portion (in UTC) from the 'dt' column.
    #    This results in 'YYYY-MM-DD' format.
    # ----------------------------------------------------------------------
    df['SettlementDate'] = df['dt'].dt.strftime('%Y-%m-%d')

    # ----------------------------------------------------------------------
    # 2) Derive 'SettlementPeriod'
    #    For each half-hour, periods go 1..48 (one day).
    #    Formula: SettlementPeriod = hour*2 + (minute // 30) + 1
    #    (00:00 => period 1; 00:30 => period 2; 01:00 => 3; ... 23:30 => 48)
    # ----------------------------------------------------------------------
    df['SettlementPeriod'] = df['dt'].dt.hour * 2 + (df['dt'].dt.minute // 30) + 1

    # ----------------------------------------------------------------------
    # 3) Fill missing 'Price'/'Volume' data:
    #    The requirement: "look at the last data point and the next available data point
    #    and take the average for missing rows in between."
    #    In pandas: 
    #        - forward fill (ffill) gives "previous" valid
    #        - backward fill (bfill) gives "next" valid
    #
    #    We'll combine the two to get the average:
    #        fill_value = (ffilled + bfilled) / 2
    #
    #    If there's no valid value in front/behind (e.g. missing from the start
    #    or at the very end), the 'bfill' or 'ffill' might remain NaN,
    #    so the average will also be NaN. 
    # ----------------------------------------------------------------------
    for col in ['PredictedPrice','ActualTomorrowPrice', 'Volume']:
        if col in df.columns:
            ffilled = df[col].ffill()     # previous valid
            bfilled = df[col].bfill()     # next valid
            df[col] = df[col].where(df[col].notna(), (ffilled + bfilled)/2)
        else:
            # If for some reason your CSV doesn't have these columns,
            # you could create them or just skip. Here we skip quietly.
            pass

    # Remove the helper column if you like
    df.drop(columns=['dt'], inplace=True)

    # ----------------------------------------------------------------------
    # 4) Write out the final DataFrame
    # ----------------------------------------------------------------------
    df.to_csv(output_csv, index=False)
    print(f"Data with SettlementDate, SettlementPeriod, and filled 'PredictedPrice/ActualTomorrowPrice'/'Volume' saved to:\n{output_csv}")


if __name__ == "__main__":
    input_file = r"C:\Users\joydo\Documents\RP3\PythonCode\MissingPriceANNForecast.csv" # Full data with missing fields
    output_file = r"C:\Users\joydo\Documents\RP3\PythonCode\FullANNForecast.csv"

    process_full_dataset(input_file, output_file)

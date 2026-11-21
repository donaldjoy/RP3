import pandas as pd
from datetime import timedelta

# Define file paths
input_file = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast FullTimeRange.csv"
output_file = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast FullDataSet.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# --- Fill missing PublishTime values ---
mask_publish = df['PublishTime'].isna() | (df['PublishTime'] == '')
if mask_publish.any():
    start_times = pd.to_datetime(df.loc[mask_publish, 'StartTime'], utc=True)
    new_publish = (start_times - pd.Timedelta(days=1)).dt.strftime('%Y-%m-%dT16:45:10Z')
    df.loc[mask_publish, 'PublishTime'] = new_publish

# --- Fill missing ProcessType, BusinessType, and PsrType values ---
df['ProcessType'] = df['ProcessType'].fillna('Day ahead')
df['BusinessType'] = df['BusinessType'].fillna('Wind generation')
df['PsrType'] = df['PsrType'].fillna('Wind Onshore')

# --- Fill missing SettlementDate values ---
mask_settlement_date = df['SettlementDate'].isna() | (df['SettlementDate'] == '')
if mask_settlement_date.any():
    start_times = pd.to_datetime(df.loc[mask_settlement_date, 'StartTime'], utc=True)
    df.loc[mask_settlement_date, 'SettlementDate'] = start_times.dt.strftime('%d/%m/%Y')

# --- Fill missing SettlementPeriod values ---
mask_settlement_period = df['SettlementPeriod'].isna() | (df['SettlementPeriod'] == '')
if mask_settlement_period.any():
    start_times = pd.to_datetime(df.loc[mask_settlement_period, 'StartTime'], utc=True)
    settlement_period = start_times.dt.hour * 2 + (start_times.dt.minute // 30) + 1
    df.loc[mask_settlement_period, 'SettlementPeriod'] = settlement_period

# --- Fill missing Quantity values ---
mask_quantity = df['Quantity'].isna()

if mask_quantity.any():
    # Ensure StartTime is in datetime format for indexing
    df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)
    
    for idx in df.loc[mask_quantity].index:
        current_row = df.loc[idx]
        current_settlement_period = current_row['SettlementPeriod']
        current_start_time = current_row['StartTime']
        
        # Calculate the 7-day window prior to the current date
        start_window = current_start_time - timedelta(days=7)
        end_window = current_start_time - timedelta(days=1)

        # Filter data to get matching SettlementPeriod and dates within the window
        mask_window = (
            (df['StartTime'] >= start_window) & 
            (df['StartTime'] <= end_window) & 
            (df['SettlementPeriod'] == current_settlement_period) &
            (~df['Quantity'].isna())  # Exclude missing values from the average calculation
        )
        
        # Compute the average Quantity, rounded to the nearest whole number
        if not df.loc[mask_window, 'Quantity'].empty:
            avg_quantity = round(df.loc[mask_window, 'Quantity'].mean(), 3)
            # If the average is 0 or missing, explicitly set 0
            df.at[idx, 'Quantity'] = avg_quantity if avg_quantity != 0 else 0
        else:
            # If no previous data is available, default to 0
            df.at[idx, 'Quantity'] = 0

# Write the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print("CSV file has been successfully updated and saved as:")
print(output_file)

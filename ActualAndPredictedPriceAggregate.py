import pandas as pd

def aggregate_actual_predicted_prices(predicted_csv, actual_csv, output_csv):
    # Load predicted price CSV
    predicted_df = pd.read_csv(predicted_csv, parse_dates=['StartTime'])
    
    # Clean column names (handling unexpected whitespace or formatting issues)
    predicted_df.columns = predicted_df.columns.str.strip()
    
    # Extract relevant columns
    predicted_df = predicted_df[['StartTime', 'SettlementDate', 'SettlementPeriod', 'PredictedPrice']]
    
    # Load actual price CSV
    actual_df = pd.read_csv(actual_csv, parse_dates=['StartTime'])
    
    # Clean column names
    actual_df.columns = actual_df.columns.str.strip()
    
    # Extract only the necessary columns (Price)
    actual_df = actual_df[['StartTime', 'Price']]
    
    # Merge actual and predicted data on StartTime
    merged_df = predicted_df.merge(actual_df, on='StartTime', how='left')

    # Rename 'Price' to 'ActualTomorrowPrice'
    merged_df.rename(columns={'Price': 'ActualTomorrowPrice'}, inplace=True)
    
    # Save to CSV
    merged_df.to_csv(output_csv, index=False)
    
    print(f"Aggregated data saved to {output_csv}")

# File paths
predicted_csv = r"C:\Users\joydo\Documents\RP3\PythonCode\ANNForecast.csv"
actual_csv = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
output_csv = r"C:\Users\joydo\Documents\RP3\PythonCode\ActualAndPredictedPriceAggregate.csv"

# Run function
aggregate_actual_predicted_prices(predicted_csv, actual_csv, output_csv)

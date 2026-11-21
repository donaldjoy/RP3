import pandas as pd

# Dictionary mapping input files to output files
file_paths = {
    "Wind Onshore DA Gen Forecast": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Onshore DA Gen Forecast.csv",
    "Wind Offshore DA Gen Forecast": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Wind Offshore DA Gen Forecast.csv",
    "Solar DA Gen Forecast": r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\Solar DA Gen Forecast.csv"
}

# Loop through each file and process it
for name, input_file in file_paths.items():
    # Define output file name
    output_file = rf"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\{name} SettlementPeriod Corrected.csv"

    # Read CSV file
    df = pd.read_csv(input_file)

    # Convert StartTime to datetime, ensuring UTC handling
    df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)

    # Compute SettlementPeriod based on hour & minute
    df['SettlementPeriod'] = (df['StartTime'].dt.hour * 2) + (df['StartTime'].dt.minute // 30) + 1

    # Save the modified dataframe to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Processed and saved: {output_file}")

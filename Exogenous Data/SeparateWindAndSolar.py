import pandas as pd

# Read the CSV file
input_file = r"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\DayAheadGenerationForecastWindAndSolar.csv"
df = pd.read_csv(input_file)

# Define the categories to filter
categories = ["Wind Offshore", "Wind Onshore", "Solar"]

# Loop through each category and save the filtered rows to a new CSV file
for category in categories:
    # Filter rows based on the category AND ProcessType = "Day ahead"
    filtered_df = df[(df["PsrType"] == category) & (df["ProcessType"] == "Day ahead")]
    
    # Define the output file name (using raw string to handle backslashes)
    output_file = fr"C:\Users\joydo\Documents\RP3\PythonCode\Exogenous Data\{category} DA Gen Forecast.csv"
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Saved {output_file} with {len(filtered_df)} rows.")

print("All files have been saved successfully.")
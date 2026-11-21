import pandas as pd

def check_30min_increments(csv_path):
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Convert the 'StartTime' column to a datetime type
    df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)

    # A list to keep track of all errors found
    errors = []

    # Loop through pairs of consecutive rows
    for i in range(len(df) - 1):
        current_time = df.loc[i, 'StartTime']
        next_time = df.loc[i + 1, 'StartTime']
        
        # The next row should be 30 minutes behind the current row
        expected_time = current_time + pd.Timedelta(minutes=30)
        
        if next_time != expected_time:
            # Store the error details (which row, the expected vs. actual times)
            # Note that row numbers in the CSV file are typically 1-based,
            # so the next row is (i+1), but we print i+2 for human readability.
            errors.append({
                'row': i + 2,
                'expected': expected_time,
                'actual': next_time,
            })
            # Instead of stopping, we continue on to the next loop iteration
            # using the next_time as the new baseline in the following iteration.

    if errors:
        # Print out each error encountered
        for err in errors:
            print(
                f"Error contained in Row {err['row']} "
                f"- expected {err['expected']}, got {err['actual']}"
            )
    else:
        print("Error Free")

if __name__ == "__main__":
    csv_file = r"C:\Users\joydo\Documents\RP3\PythonCode\ANNForecast.csv"
    check_30min_increments(csv_file)


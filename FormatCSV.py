import pandas as pd

def remove_zero_prices(input_csv, output_csv):
    """
    Reads a CSV file, removes rows where price == 0, 
    and writes the remaining data to a new CSV file.
    """
    # Read the input CSV into a pandas DataFrame
    df = pd.read_csv(input_csv)
    
    # Filter out rows where 'price' is 0
    df_filtered = df[df['dataProvider'] != 'N2EXMIDP']
    
    # Write the filtered DataFrame to a new CSV
    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered data has been written to: {output_csv}")

if __name__ == "__main__":
    # Adjust the file paths as needed
    input_path = r"C:\Users\joydo\Documents\RP3\PythonCode\2023-2025.csv"
    output_path = r"C:\Users\joydo\Documents\RP3\PythonCode\2023-2025 Cleansed.csv"
    
    remove_zero_prices(input_path, output_path)

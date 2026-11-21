import pandas as pd

def combine_json_to_csv(json_file1, output_csv):
    """
    Reads a JSON files and writes out a CSV file.
    """
    # Read each JSON file into a pandas DataFrame
    df1 = pd.read_json(json_file1)

    # Write the combined DataFrame to a CSV file
    df1.to_csv(output_csv, index=False)
    print(f"Combined data written to {output_csv}.")


if __name__ == "__main__":
    json_path_1 = r"C:\Users\joydo\Documents\RP3\PythonCode\2023-2025.json"
    output_csv_path = r"C:\Users\joydo\Documents\RP3\PythonCode\2023-2025.csv"

    combine_json_to_csv(json_path_1, output_csv_path)

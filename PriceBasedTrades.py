import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Battery-related variables
# -------------------------------------------------------------------
MWh_Battery = 33.46                 # Initial capacity in MWh
Degradation_Per_Cycle = 0.025 / 365 # e.g. 2.5% per year => 0.00006849 per cycle
round_trip_efficiency_factor = 0.9456  # 94.56% efficiency

# SoC is dimensionless [0..1].
# We'll define a fixed increment for SoC changes when we import/export:
SoC_increment = 0.125  # Example chosen; adjust as needed

def add_price_based_trades(input_csv, output_csv):
    """
    Reads a CSV with 'StartTime' and 'Price' columns, plus any others.
    For each day (48 rows of half-hours per day):
      1) Finds the 8 cheapest slots -> "Import"
      2) Finds the 8 most expensive slots -> "Export"
      3) The rest remain idle
    Then applies the battery logic in chronological order over the entire dataset
    to compute SoC, Cost, Revenue, Rolling Cashflow, and partial-cycle counting.
    Writes the final DataFrame to 'output_csv'.
    """

    # 1) Load data
    df = pd.read_csv(input_csv)

    # Ensure 'StartTime' is a datetime
    if not pd.api.types.is_datetime64_any_dtype(df['StartTime']):
        df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')

    # Sort by time
    df.sort_values(by='StartTime', inplace=True, ignore_index=True)

    # Create a 'date' column so we can group by day
    df['date'] = df['StartTime'].dt.date

    # 2) For each day, pick 8 cheapest -> import, 8 costliest -> export
    # We'll store these in a new column, e.g. 'Action' with 0=idle,1=import,2=export
    df['Action'] = 0

    for day_val, df_day in df.groupby('date'):
        # Sort ascending by Price => pick first 8 for import
        df_day_sorted_asc = df_day.sort_values(by='Price', ascending=True)
        import_indices = df_day_sorted_asc.index[:8]  # first 8 rows => import
        df.loc[import_indices, 'Action'] = 1

        # Sort descending by Price => pick first 8 for export
        df_day_sorted_desc = df_day.sort_values(by='Price', ascending=False)
        export_indices = df_day_sorted_desc.index[:8]  # first 8 rows => export
        df.loc[export_indices, 'Action'] = 2

        # If there's overlap (same row among cheapest & costliest),
        # typically that means all prices are equal or there's a tie.
        # In that edge case, "export" might overwrite "import" or vice versa.
        # If you want a more careful approach, handle ties or overlap differently.

    # 3) Prepare columns for battery logic
    df['SoC'] = 0.0
    df['Cost'] = 0.0
    df['Revenue'] = 0.0
    df['Total Cycles Achieved'] = None  # partial-cycle logic
    # We'll fill Rolling Cashflow after we do the loop.

    # Tracking variables for partial cycles
    SoC = 0.0
    prev_SoC = 0.0
    partial_cycle_accum = 0.0
    cycle_count = 0

    # A helper to degrade battery capacity based on cycle_count
    def get_degraded_params(completed_cycles):
        # Each full cycle degrades capacity by (Degradation_Per_Cycle * MWh_Battery)
        effective_capacity = MWh_Battery * (1.0 - completed_cycles * Degradation_Per_Cycle)
        effective_capacity = max(effective_capacity, 0.0)
        # You mentioned dividing by 2.231, but adjust as needed:
        mw_conn = effective_capacity / 2.231
        return effective_capacity, mw_conn

    # 4) Main loop over all rows in chronological order
    for i in range(len(df)):
        effective_capacity, mw_connection = get_degraded_params(cycle_count)
        action = df.loc[i, 'Action']  # 0=idle,1=import,2=export
        Price = df.loc[i, 'Price']

        if action == 1:  # import
            # only import if SoC + increment <= 1
            if SoC + SoC_increment <= 1.0:
                df.loc[i, 'Cost'] = -mw_connection * Price
                SoC += SoC_increment

        elif action == 2:  # export
            # only export if SoC - increment >= 0
            if SoC - SoC_increment >= 0.0:
                df.loc[i, 'Revenue'] = mw_connection * Price * round_trip_efficiency_factor
                SoC -= SoC_increment

        df.loc[i, 'SoC'] = SoC

        # partial-cycle accumulation
        soc_change = abs(SoC - prev_SoC)
        partial_cycle_accum += soc_change

        # each time partial_cycle_accum >= 2 => 1 full cycle
        while partial_cycle_accum >= 2.0:
            cycle_count += 1
            partial_cycle_accum -= 2.0
            df.loc[i, 'Total Cycles Achieved'] = cycle_count

        prev_SoC = SoC

    # 5) Rolling Cashflow
    df['Rolling Cashflow'] = (df['Cost'] + df['Revenue']).cumsum()

    # 6) Reorder columns so SoC, Cost, Revenue, Rolling Cashflow, Total Cycles Achieved come last
    new_cols = ['SoC', 'Cost', 'Revenue', 'Rolling Cashflow', 'Total Cycles Achieved']
    existing_cols = [c for c in df.columns if c not in new_cols]
    df = df[existing_cols + new_cols]

    # 7) Save to CSV
    df.drop(columns=['date'], inplace=True)  # Remove 'date' before saving
    df.to_csv(output_csv, index=False)
    print(f"Price-Based Trades completed. Output: {output_csv}")
    print("Added columns: SoC, Cost, Revenue, Rolling Cashflow, Total Cycles Achieved")

# -----------------------------------------------
# Example direct usage
# -----------------------------------------------
if __name__ == "__main__":
    input_file = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
    output_file = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - PriceBasedTrades.csv"
    add_price_based_trades(input_file, output_file)
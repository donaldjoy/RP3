import pandas as pd

# -------------------------------------------------------------------
# Battery-related variables
# -------------------------------------------------------------------
MWh_Battery = 33.46                 # Initial capacity in MWh
Degradation_Per_Cycle = 0.025 / 365 # e.g. 2.5% per year => 0.00006849 per cycle, etc.
round_trip_efficiency_factor = 0.9456  # 94.56% efficiency

# SoC ranges from 0 to 1 dimensionlessly.
# We'll still define SoC_increment as before, but partial cycles are tracked differently.
SoC_increment = 0.125  # Ask Ben about this, I think it should be 0.25

def add_time_based_trades(input_csv, output_csv):
    # Read CSV
    df = pd.read_csv(input_csv)
    
    # Convert 'StartTime' to datetime, if needed
    if not pd.api.types.is_datetime64_any_dtype(df['StartTime']):
        df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    
    # Sort by time so everything is chronological
    df.sort_values(by='StartTime', inplace=True, ignore_index=True)

    # Define time-based import/export windows
    import_intervals = [(1, 4)]    # 01:00..04:59 => "Import"
    export_intervals = [(17, 20)]  # 17:00..20:59 => "Export"
    
    # Prepare columns
    df['SoC'] = 0.0
    df['Cost'] = 0.0
    df['Revenue'] = 0.0
    df['Total Cycles Achieved'] = None  # Will update only on rows where at least one full cycle completes

    # Tracking variables
    SoC = 0.0
    cycle_count = 0           # Count of total full cycles completed so far
    partial_cycle_accum = 0.0 # Accumulate SoC changes here for partial cycles
    prev_SoC = SoC            # We’ll measure changes from this

    # Function to compute the effective capacity & MW connection
    # after 'completed_cycles' full cycles
    def get_degraded_params(completed_cycles):
        # Each cycle degrades capacity by (Degradation_Per_Cycle * MWh_Battery)
        effective_capacity = MWh_Battery * (1.0 - completed_cycles * Degradation_Per_Cycle)
        effective_capacity = max(effective_capacity, 0.0)  # clamp to 0
        mw_conn = effective_capacity / 2.231 # Ask Ben about this?
        return effective_capacity, mw_conn

    # Main loop
    for i in range(len(df)):
        hour = df.loc[i, 'StartTime'].hour

        # Look up the current degraded capacity / MW connection 
        # based on how many full cycles we've done so far
        effective_capacity, mw_connection = get_degraded_params(cycle_count)

        # Decide whether to import or export
        if any(start_h <= hour <= end_h for (start_h, end_h) in import_intervals):
            if SoC + SoC_increment <= 1.0:
                Price = df.loc[i, 'Price']
                df.loc[i, 'Cost'] = -mw_connection * Price  # buying power => negative cost
                SoC += SoC_increment
        elif any(start_h <= hour <= end_h for (start_h, end_h) in export_intervals):
            if SoC - SoC_increment >= 0.0:
                Price = df.loc[i, 'Price']
                df.loc[i, 'Revenue'] = mw_connection * Price * round_trip_efficiency_factor
                SoC -= SoC_increment
        
        # Update SoC in the DataFrame
        df.loc[i, 'SoC'] = SoC

        # ------------------------------------------------------
        # Partial-Cycle Accumulation
        # ------------------------------------------------------
        # Add the absolute SoC change (since last row) to partial_cycle_accum
        soc_change = abs(SoC - prev_SoC)
        partial_cycle_accum += soc_change
        
        # Check if we've completed >=1 full cycle's worth (which is 2.0 SoC "distance")
        while partial_cycle_accum >= 2.0:
            cycle_count += 1
            partial_cycle_accum -= 2.0
            # Mark that we’ve reached a new total cycle count on this row
            df.loc[i, 'Total Cycles Achieved'] = cycle_count
        
        # Update prev_SoC for next iteration
        prev_SoC = SoC

    # Rolling Cashflow
    df['Rolling Cashflow'] = (df['Cost'] + df['Revenue']).cumsum()

    # Reorder columns so that SoC, Cost, Revenue, Rolling Cashflow, Total Cycles Achieved
    # come last, preserving the original columns' order
    new_cols = ['SoC', 'Cost', 'Revenue', 'Rolling Cashflow', 'Total Cycles Achieved']
    existing_cols = [c for c in df.columns if c not in new_cols]
    df = df[existing_cols + new_cols]

    # Save
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV: {output_csv}")
    print("Added columns: SoC, Cost, Revenue, Rolling Cashflow, Total Cycles Achieved")

if __name__ == "__main__":
    input_file = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - FullDataSet.csv"
    output_file = r"C:\Users\joydo\Documents\RP3\PythonCode\2016-2025 - TimeBasedTrades.csv"
    add_time_based_trades(input_file, output_file)

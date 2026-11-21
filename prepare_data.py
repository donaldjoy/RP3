"""Step 2: build the clean half-hourly datasets in data/processed from data/raw.

    python prepare_data.py              # prices and wind/solar forecasts
    python prepare_data.py --check      # only report gaps in the processed files

The prices need data/raw/market_index_data_raw.csv from ``python download_data.py``.
If it is missing, the committed data/processed/prices_2016_2025.csv is kept as it is.
"""
import argparse

import pandas as pd

from rp3 import config
from rp3.data_prep import build_generation, build_prices, find_gaps, read_timeseries


def save(df, path):
    """Write numbers as the committed files do: "2819" not "2819.0", "103.2" not "103.19999999999999"."""
    out = df.copy()
    for col in out.select_dtypes("float").columns:
        values = out[col].round(10)
        whole = values.notna() & (values % 1 == 0)
        out[col] = values.astype(object)
        out.loc[whole, col] = values[whole].astype("int64")
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def check(path):
    gaps = find_gaps(read_timeseries(path)["StartTime"])
    if gaps.empty:
        print(f"OK   {path.name}: no gaps")
    else:
        print(f"GAPS {path.name}: {len(gaps)}")
        print(gaps.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check the processed files")
    args = parser.parse_args()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.check:
        if config.RAW_PRICES_CSV.exists():
            save(build_prices(pd.read_csv(config.RAW_PRICES_CSV)), config.PRICES_CSV)
        else:
            print(f"No {config.RAW_PRICES_CSV.name}; keeping the committed {config.PRICES_CSV.name}")

        raw = pd.read_csv(config.RAW_GENERATION_CSV)
        for psr_type, path in config.GENERATION_CSVS.items():
            save(build_generation(raw, psr_type), path)

    for path in [config.PRICES_CSV, *config.GENERATION_CSVS.values()]:
        check(path)


if __name__ == "__main__":
    main()

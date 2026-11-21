"""Step 1: download the raw price and wind/solar forecast data from Elexon BMRS.

    python download_data.py                # both datasets, 2016-09-19 to 2025-01-20
    python download_data.py --only prices  # just the Market Index Data
    python download_data.py --out data/raw/check  # write somewhere else

Takes a few minutes (about 440 one-week requests per dataset). The repository already
contains the raw generation file, so this is only needed to rebuild from scratch.
"""
import argparse
from pathlib import Path

from rp3 import config
from rp3.elexon import download_market_index, download_wind_solar_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["prices", "generation"])
    parser.add_argument("--out", type=Path, default=config.RAW_DIR, help="output folder")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "prices"):
        path = args.out / config.RAW_PRICES_CSV.name
        download_market_index(config.START, config.END).to_csv(path, index=False)
        print(f"Saved {path}")
    if args.only in (None, "generation"):
        # The forecast for the first day is published the day before
        path = args.out / config.RAW_GENERATION_CSV.name
        download_wind_solar_forecast(config.START, config.END).to_csv(path, index=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

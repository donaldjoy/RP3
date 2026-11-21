"""Step 4: run the three battery trading strategies and save every trade to results/.

    python run_backtests.py
    python run_backtests.py --forecast results/ann_forecast.csv

Strategies (see rp3/battery.py):
  time   buy 01:00-04:59, sell 17:00-20:59 (UK time) every day
  price  buy the 8 cheapest and sell the 8 dearest half-hours of each day, using the
         real prices (perfect foresight, an upper benchmark)
  ann    the same, but choosing the half-hours from the ANN forecast
"""
import argparse
from pathlib import Path

import pandas as pd

from rp3 import config
from rp3.battery import ann_based_trades, comparison_start, price_based_trades, time_based_trades, yearly_profit
from rp3.data_prep import load_prices, read_timeseries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forecast", type=Path, default=config.ANN_FORECAST_CSV,
                        help="ANN forecast CSV with StartTime and PredictedPrice")
    args = parser.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)

    prices = load_prices()
    trades = {
        "time": time_based_trades(prices),
        "price": price_based_trades(prices),
        "ann": ann_based_trades(read_timeseries(args.forecast), prices),
    }
    for key, df in trades.items():
        df.to_csv(config.TRADES_CSVS[key], index=False)
        print(f"Saved {config.TRADES_CSVS[key]}")

    # Summary: whole period, and the ANN test period where all three can be compared
    start = comparison_start(trades["ann"])
    summary = pd.DataFrame({
        "Profit, full period (GBP m)": {k: df["Rolling Cashflow"].iloc[-1] / 1e6 for k, df in trades.items()},
        f"Profit from {start.tz_convert(config.TIMEZONE):%Y-%m-%d} (GBP m)":
            {k: yearly_profit(df, start).sum() / 1e6 for k, df in trades.items()},
        "Full cycles": {k: int(df["Total Cycles Achieved"].max()) for k, df in trades.items()},
    })
    print()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()

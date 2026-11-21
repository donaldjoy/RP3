"""Step 5: draw every report figure into images/.

    python make_plots.py               # all figures
    python make_plots.py market ann    # only some groups
    python make_plots.py --show        # also open each figure

Groups: market (prices and volume), generation (wind/solar forecasts),
ann (forecast vs actual), trades (profit; run run_backtests.py first).
"""
import argparse
from pathlib import Path

import matplotlib

from rp3 import config, plotting
from rp3.data_prep import load_generation, load_prices, read_timeseries
from rp3.forecasting import error_metrics

GROUPS = ["market", "generation", "ann", "trades"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", choices=GROUPS, default=GROUPS)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", type=Path, default=config.IMAGES_DIR, help="folder for the PNGs")
    args = parser.parse_args()
    plotting.OUTPUT_DIR = args.out
    if args.show:
        plotting.SHOW = True
    else:
        matplotlib.use("Agg")

    if "market" in args.groups:
        prices = load_prices()
        plotting.price_and_volume(prices)
        plotting.average_price_by_period(prices)
        plotting.average_price_by_weekday(prices)
        plotting.price_by_month(prices)
        plotting.daily_price_vs_volume(prices)
        plotting.total_volume_per_year(prices)

    if "generation" in args.groups:
        generation = load_generation()
        plotting.generation_series(generation)
        plotting.generation_summaries(generation)

    if "ann" in args.groups:
        forecast = read_timeseries(config.ANN_FORECAST_CSV)
        m = error_metrics(forecast["ActualTomorrowPrice"], forecast["PredictedPrice"])
        print(f"ANN test set: RMSE {m['RMSE']:.2f}, MAE {m['MAE']:.2f}")
        plotting.forecast_vs_actual(forecast)
        plotting.forecast_windows(forecast)
        if config.ANN_MULTI_SEED_CSV.exists():
            plotting.multiple_seeds(read_timeseries(config.ANN_MULTI_SEED_CSV))

    if "trades" in args.groups:
        missing = [p for p in config.TRADES_CSVS.values() if not p.exists()]
        if missing:
            raise SystemExit(f"Run `python run_backtests.py` first (missing {missing[0].name})")
        trades = {k: read_timeseries(p) for k, p in config.TRADES_CSVS.items()}
        perfect_and_time = {k: trades[k] for k in ("time", "price")}
        plotting.cashflow(perfect_and_time)
        plotting.profit_per_year(perfect_and_time)
        plotting.import_export_counts(trades["price"])
        plotting.cashflow(trades, "Rolling Cashflow Over Time for Predicted Price", start=config.COMPARISON_START)
        plotting.profit_per_year(trades, "Total Profit per Year for Predicted Price", start=config.COMPARISON_START)

    print(f"Figures saved to {args.out}")


if __name__ == "__main__":
    main()

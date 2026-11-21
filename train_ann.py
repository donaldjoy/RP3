"""Step 3: train the ANN day-ahead price forecast (scikit-learn, any recent Python).

    python train_ann.py                 # one model -> results/ann_forecast.csv
    python train_ann.py --seed 7        # the same with another random seed
    python train_ann.py --seeds         # one model per seed (42, 100, 7) -> results/ann_forecast_multiple_seeds.csv
    python train_ann.py --tune 50       # Optuna search on the validation rows (needs optuna)

The settings (layers, learning rate, epochs, ...) are in rp3/config.py; copy the
result of --tune there. Runs are repeatable: the default seed is config.SEED.
"""
import argparse

from rp3 import config
from rp3.data_prep import load_generation, load_prices
from rp3.forecasting import run_forecast, run_multiple_seeds, tune_hyperparameters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seeds", action="store_true", help="train once per seed in config.SEEDS")
    mode.add_argument("--tune", type=int, metavar="TRIALS", help="run an Optuna search with this many trials")
    parser.add_argument("--seed", type=int, default=config.SEED, help="random seed for a single run")
    args = parser.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)

    prices, generation = load_prices(), load_generation()

    if args.tune:
        study = tune_hyperparameters(prices, generation, n_trials=args.tune)
        print(f"Best validation RMSE: {study.best_value:.3f}")
        print("Best settings (copy into rp3/config.py):", study.best_params)
    elif args.seeds:
        forecast, metrics = run_multiple_seeds(prices, generation)
        for seed, m in metrics.items():
            print(f"Seed {seed}: RMSE {m['RMSE']:.3f}, MAE {m['MAE']:.3f}")
        print(f"Average MAE: {sum(m['MAE'] for m in metrics.values()) / len(metrics):.3f}")
        forecast.to_csv(config.ANN_MULTI_SEED_CSV)
        print(f"Saved {config.ANN_MULTI_SEED_CSV}")
    else:
        forecast, metrics = run_forecast(prices, generation, seed=args.seed)
        print(f"Test set RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}")
        forecast.to_csv(config.ANN_FORECAST_CSV)
        print(f"Saved {config.ANN_FORECAST_CSV}")


if __name__ == "__main__":
    main()

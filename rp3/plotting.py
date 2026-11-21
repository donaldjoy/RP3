"""Every figure in the report. Each function draws one figure and saves it to images/.

Figures have no on-plot title (the report captions them); the title is used as the
file name instead, e.g. "Total Profit per Year" -> images/Total Profit per Year.png.
"""
import re
from contextlib import contextmanager

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from rp3 import config
from rp3.data_prep import settlement_date, settlement_period, uk_time

STYLE = {"font.size": 15, "axes.labelsize": 25, "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 15}
FORECAST_STYLE = {**STYLE, "xtick.labelsize": 15}   # date labels need more room
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
STRATEGY_LABELS = {"time": "Time Based", "price": "Price Based", "ann": "ANN Based"}
STRATEGY_COLOURS = {"time": "blue", "price": "orange", "ann": "red"}
GENERATION_LABELS = {"Wind Offshore": "Wind Offshore Forecast", "Wind Onshore": "Wind Onshore Forecast",
                     "Solar": "Solar Forecast"}
SHOW = False                 # set True (or pass --show to make_plots.py) to open each figure as well
OUTPUT_DIR = config.IMAGES_DIR


def save(fig, title):
    """Save as "<title>.png" in OUTPUT_DIR (characters Windows does not allow become "_")."""
    folder = OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / (re.sub(r'[<>:"/\\|?*]', "_", title) + ".png"), dpi=300)
    if SHOW:
        plt.show()
    plt.close(fig)


@contextmanager
def figure(title, figsize=(12, 6), style=STYLE, **subplot_kw):
    """``with figure("Title") as (fig, ax): ...`` draws, then saves on exit."""
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize, **subplot_kw)
        yield fig, ax
        save(fig, title)


def millions(ax, decimals=0):
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x / 1e6:,.{decimals}f}"))


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def price_and_volume(prices):
    with figure("Price (GBP/MWh) and Volume (MWh) from 2016 to 2025", figsize=(12, 8), nrows=2,
                sharex=True, gridspec_kw={"height_ratios": [4, 1]}) as (fig, (top, bottom)):
        top.plot(prices["StartTime"], prices["Price"], color="blue", label="Price")
        top.set_ylabel("Price (GBP/MWh)")
        top.legend(loc="upper right")
        bottom.plot(prices["StartTime"], prices["Volume"], color="green", label="Volume")
        bottom.set_xlabel("Time")
        bottom.set_ylabel("Volume (MWh)")
        bottom.legend(loc="upper right")


def average_price_by_period(prices, highlight=((14, 20), (33.5, 39.5))):
    """Average daily price curve; the shaded bands mark the cheap-morning / dear-evening spans."""
    curve = prices.groupby("SettlementPeriod")["Price"].mean().reindex(range(1, 49))
    with figure("Average Daily Price (GBP/MWh) by Settlement Period") as (fig, ax):
        ax.plot(curve.index, curve.values, color="blue", marker="o", label="Average Price")
        ax.set_xlabel("Settlement Period")
        ax.set_ylabel("Average Price (GBP/MWh)", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.legend(loc="upper left")
        for lo, hi in highlight:
            ax.axvspan(lo, hi, color="yellow", alpha=1)


def average_price_by_weekday(prices):
    table = prices.pivot_table(index="SettlementPeriod", columns=uk_time(prices["StartTime"]).day_name(),
                               values="Price", aggfunc="mean")[DAYS].reindex(range(1, 49))
    table.columns.name = "Day of Week"
    with figure("Average Price by Settlement Period for Each Day of Week") as (fig, ax):
        table.plot(ax=ax, marker="o")
        ax.set_xlabel("Settlement Period")
        ax.set_ylabel("Average Price (GBP/MWh)")


def price_by_month(prices):
    df = pd.DataFrame({"Price": prices["Price"],
                       "month": pd.Categorical(uk_time(prices["StartTime"]).month_name(), MONTHS, ordered=True)})
    df = df.sort_values("month", kind="stable")   # seaborn colours months in order of appearance
    with figure("Price vs. Month") as (fig, ax):
        sns.violinplot(data=df, x="month", y="Price", hue="month", order=MONTHS, palette="coolwarm",
                       inner="quartile", legend=False, dodge=False, ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Price (GBP/MWh)")
        ax.tick_params(axis="x", rotation=45)


def daily_price_vs_volume(prices, year=2024):
    df = prices[uk_time(prices["StartTime"]).year == year]
    daily = df.groupby(settlement_date(df["StartTime"])).agg(Volume=("Volume", "sum"), Price=("Price", "mean"))
    with figure(f"Daily Average Price vs Daily Volume ({year})", figsize=(10, 6)) as (fig, ax):
        ax.scatter(daily["Price"], daily["Volume"], alpha=0.5, c="blue", edgecolor="k")
        ax.set_xlabel("Average Daily Price (GBP/MWh)")
        ax.set_ylabel("Daily Volume (MWh)")
        ax.grid(True)


def total_volume_per_year(prices, skip_years=(2016, 2025)):
    """Partial first and last years are left out."""
    totals = prices.groupby(uk_time(prices["StartTime"]).year)["Volume"].sum().drop(list(skip_years), errors="ignore")
    with figure("Total Volume vs Year", figsize=(10, 6)) as (fig, ax):
        totals.plot(kind="bar", color="orange", edgecolor="black", ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Volume (Million MWh)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        millions(ax)


# ---------------------------------------------------------------------------
# Wind and solar forecasts
# ---------------------------------------------------------------------------
def generation_series(generation):
    for name, df in generation.items():
        with figure(f"{name} DA Gen Forecast") as (fig, ax):
            ax.plot(df["StartTime"], df["Quantity"], color="b", label=GENERATION_LABELS[name])
            ax.set_xlabel("Year")
            ax.set_ylabel("Forecast output (MW)")
            ax.legend()
            ax.grid(True)


def _generation_table(generation, by, how):
    """Forecast MW grouped by UK year, month or settlement period."""
    columns = {}
    for name, df in generation.items():
        local = uk_time(df["StartTime"])
        key = {"year": local.year, "month": local.month, "period": settlement_period(df["StartTime"])}[by]
        columns[GENERATION_LABELS[name]] = df["Quantity"].groupby(np.asarray(key)).agg(how)
    return pd.DataFrame(columns).sort_index()


def generation_summaries(generation, skip_years=(2016, 2025)):
    # Each value is average MW over half an hour, so energy = MW x 0.5 h; shown in TWh
    yearly = _generation_table(generation, "year", "sum").drop(list(skip_years), errors="ignore") * 0.5
    with figure("Aggregated DA Gen Forecast Quantity per Year") as (fig, ax):
        yearly.plot(kind="bar", ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Forecast energy (TWh)")
        millions(ax)
        ax.legend(title="Category")
        ax.grid(True)

    by_period = _generation_table(generation, "period", "mean").reindex(range(1, 49))
    with figure("Average DA Gen Forecast Quantity per Settlement Period") as (fig, ax):
        by_period.plot(ax=ax, marker="o")
        ax.set_xlabel("Settlement Period")
        ax.set_ylabel("Average forecast (MW)")
        ax.legend(title="Category")
        ax.grid(True)

    by_month = _generation_table(generation, "month", "mean").reindex(range(1, 13))
    by_month.index.name = "Month"
    with figure("Average DA Gen Forecast Quantity per Month") as (fig, ax):
        by_month.plot(kind="bar", ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average forecast (MW)")
        ax.legend(title="Category")
        ax.grid(True)


# ---------------------------------------------------------------------------
# ANN forecast
# ---------------------------------------------------------------------------
def forecast_vs_actual(forecast):
    """``forecast`` has StartTime (the half-hour forecast), Price (real) and PredictedPrice."""
    with figure("Electricity Price Forecast vs Actual (Test Set)", style=FORECAST_STYLE) as (fig, ax):
        ax.plot(forecast["StartTime"], forecast["Price"], label="Actual Price", color="green")
        ax.plot(forecast["StartTime"], forecast["PredictedPrice"], label="Forecasted Price", color="red")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (GBP/MWh)")
        ax.legend()


FORECAST_WINDOWS = {
    "Winter (1st to 3rd Dec 2024)": ("2024-12-01", "2024-12-03"),
    "Summer (1st to 3rd Jun 2024)": ("2024-06-01", "2024-06-03"),
}


def forecast_windows(forecast, windows=FORECAST_WINDOWS):
    """Zoom on a few UK days (both end days included); times shown in UK local time."""
    days = pd.Series(settlement_date(forecast["StartTime"]), index=forecast.index)
    for name, (start, end) in windows.items():
        df = forecast[(days >= pd.Timestamp(start).date()) & (days <= pd.Timestamp(end).date())]
        df = df.assign(StartTime=uk_time(df["StartTime"]).tz_localize(None))
        with figure(f"Forecast vs Actual Price {name}", style=FORECAST_STYLE) as (fig, ax):
            ax.plot(df["StartTime"], df["Price"], label="Actual Price", color="green")
            ax.plot(df["StartTime"], df["PredictedPrice"], label="Forecasted Price", color="red")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (GBP/MWh)")
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%y %H:%M"))
            ax.tick_params(axis="x", rotation=45)
            ax.legend()


def multiple_seeds(forecast, seeds=config.SEEDS):
    with figure("Electricity Price Forecast vs Actual (Multiple Models)", style=FORECAST_STYLE) as (fig, ax):
        ax.plot(forecast["StartTime"], forecast["Price"], label="Actual Price",
                color="black", linewidth=2)
        for seed, colour in zip(seeds, ["red", "blue", "green"]):
            ax.plot(forecast["StartTime"], forecast[f"PredictedPrice_{seed}"], label=f"Predicted (seed {seed})",
                    color=colour)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (GBP/MWh)")
        ax.legend()


# ---------------------------------------------------------------------------
# Trading results
# ---------------------------------------------------------------------------
def _from(trades, start):
    return {k: (df if start is None else df[df["StartTime"] >= start]) for k, df in trades.items()}


def cashflow(trades, title="Rolling Cashflow Over Time", start=None):
    """Cumulative profit of each strategy; ``trades`` is {"time": df, "price": df, ...}.

    With ``start`` every curve restarts from zero at that time.
    """
    with figure(title, figsize=(10, 5)) as (fig, ax):
        for key, df in _from(trades, start).items():
            ax.plot(df["StartTime"], (df["Cost"] + df["Revenue"]).cumsum() / 1e6,
                    label=STRATEGY_LABELS[key], color=STRATEGY_COLOURS[key])
        ax.set_xlabel("Time")
        ax.set_ylabel("Profit (GBP Millions)")
        ax.legend()
        if start is not None:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def profit_per_year(trades, title="Total Profit per Year", start=None):
    from rp3.battery import yearly_profit

    table = pd.DataFrame({STRATEGY_LABELS[k]: yearly_profit(df) / 1e6
                          for k, df in _from(trades, start).items()}).fillna(0)
    with figure(title, figsize=(8, 5)) as (fig, ax):
        table.plot(kind="bar", ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Profit (GBP Millions)")


def import_export_counts(trades):
    """How often each settlement period was actually used to buy and to sell.

    Trades skipped because the battery was full or empty are not counted. Periods 49
    and 50 (only on the day the clocks go back) are left out.
    """
    from rp3.battery import EXPORT, IMPORT

    period = settlement_period(trades["StartTime"])
    counts = pd.crosstab(period, trades["Executed"]).reindex(index=range(1, 49), columns=[IMPORT, EXPORT],
                                                          fill_value=0)
    x, width = np.arange(1, 49), 0.4
    with figure("Count of Imports & Exports by Settlement Period") as (fig, ax):
        ax.bar(x - width / 2, counts[IMPORT], width, color="green", label="Imports")
        ax.bar(x + width / 2, counts[EXPORT], width, color="red", label="Exports")
        ax.set_xlabel("Settlement Period")
        ax.set_ylabel("Occurrences")
        ax.set_xticks(np.arange(0, 49, 4))
        ax.legend()

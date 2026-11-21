"""Battery arbitrage model and the three trading strategies.

A strategy is just a choice of *action* for every half-hour:
``IDLE`` (0), ``IMPORT`` (1, buy and charge) or ``EXPORT`` (2, sell and discharge).
:func:`simulate` then plays those actions through the battery in time order.

Battery rules (unchanged from the original RP3 model):

* State of charge (SoC) runs from 0 to 1 in steps of ``soc_step`` (0.125, so eight
  half-hour trades fill or empty it). A trade that would leave 0..1 is skipped.
* One full cycle is a SoC distance of 2 (fully charge and fully discharge).
  After ``n`` full cycles the capacity is ``capacity * (1 - n * degradation)``.
* The grid connection is ``capacity / capacity_to_power_ratio`` MW. A trade buys or
  sells that many MW at the half-hour's price (GBP/MWh). Revenue on export is
  reduced by the round-trip efficiency.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rp3 import config

IDLE, IMPORT, EXPORT = 0, 1, 2
RESULT_COLUMNS = ["Action", "Executed", "SoC", "Cost", "Revenue", "Rolling Cashflow", "Total Cycles Achieved"]


@dataclass(frozen=True)
class Battery:
    capacity_mwh: float = config.BATTERY_CAPACITY_MWH
    degradation_per_cycle: float = config.DEGRADATION_PER_CYCLE
    round_trip_efficiency: float = config.ROUND_TRIP_EFFICIENCY
    soc_step: float = config.SOC_STEP
    capacity_to_power_ratio: float = config.CAPACITY_TO_POWER_RATIO

    @property
    def steps_to_full(self):
        return int(round(1 / self.soc_step))


# ---------------------------------------------------------------------------
# Strategies: each returns one action per row
# ---------------------------------------------------------------------------
def time_based_actions(start_times, import_hours=config.IMPORT_HOURS, export_hours=config.EXPORT_HOURS):
    """Buy in a fixed night window and sell in a fixed evening window (inclusive UTC hours)."""
    hour = pd.DatetimeIndex(start_times).hour
    actions = np.full(len(hour), IDLE, dtype=np.int8)
    actions[(hour >= import_hours[0]) & (hour <= import_hours[1])] = IMPORT
    actions[(hour >= export_hours[0]) & (hour <= export_hours[1])] = EXPORT
    return actions


def price_ranked_actions(prices, days, n_slots=config.SLOTS_PER_DAY):
    """Each day, buy in the ``n_slots`` cheapest half-hours and sell in the ``n_slots`` dearest.

    ``prices`` is what the decision is based on: the real price (perfect foresight)
    or a forecast. On a day with fewer than ``2 * n_slots`` rows a slot can be in
    both sets, and selling wins (as in the original code).
    """
    ranks = pd.DataFrame({"price": np.asarray(prices), "day": np.asarray(days)}).groupby("day")["price"]
    cheapest = ranks.rank(method="first", ascending=True) <= n_slots
    dearest = ranks.rank(method="first", ascending=False) <= n_slots
    return np.where(dearest, EXPORT, np.where(cheapest, IMPORT, IDLE)).astype(np.int8)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _walk_soc(actions, steps_to_full):
    """SoC level (in whole steps) after each row, and which trades actually happened.

    This is the only part that has to be a loop: whether a trade fits depends on
    every earlier trade. It runs on plain integers, so it is fast.
    """
    level = 0
    levels = np.empty(len(actions), dtype=np.int64)
    executed = np.zeros(len(actions), dtype=np.int8)
    for i, action in enumerate(np.asarray(actions).tolist()):
        if action == IMPORT and level < steps_to_full:
            level += 1
            executed[i] = IMPORT
        elif action == EXPORT and level > 0:
            level -= 1
            executed[i] = EXPORT
        levels[i] = level
    return levels, executed


def simulate(actions, trade_prices, battery=Battery()):
    """Play the actions through the battery and return the trade results per row."""
    trade_prices = np.asarray(trade_prices, dtype=float)
    levels, executed = _walk_soc(actions, battery.steps_to_full)

    # Every executed trade moves the SoC by one step; 2 / step trades = one full cycle
    trades_per_cycle = 2 * battery.steps_to_full
    cycles_after = np.cumsum(executed != IDLE) // trades_per_cycle
    cycles_before = np.concatenate(([0], cycles_after[:-1]))

    capacity = np.maximum(battery.capacity_mwh * (1.0 - cycles_before * battery.degradation_per_cycle), 0.0)
    power_mw = capacity / battery.capacity_to_power_ratio
    cost = np.where(executed == IMPORT, -power_mw * trade_prices, 0.0)
    revenue = np.where(executed == EXPORT, power_mw * trade_prices * battery.round_trip_efficiency, 0.0)

    return pd.DataFrame({
        "Action": np.asarray(actions),
        "Executed": executed,
        "SoC": levels * battery.soc_step,
        "Cost": cost,
        "Revenue": revenue,
        "Rolling Cashflow": np.cumsum(cost + revenue),
        # Filled only on the rows where a new full cycle completes
        "Total Cycles Achieved": np.where(cycles_after > cycles_before, cycles_after, np.nan),
    })


def run_strategy(df, actions, price_column, battery=Battery()):
    """Return ``df`` (sorted by time) with the simulation results appended as columns."""
    out = df.drop(columns=RESULT_COLUMNS, errors="ignore").reset_index(drop=True)
    results = simulate(actions, out[price_column], battery)
    return pd.concat([out, results], axis=1)


# ---------------------------------------------------------------------------
# The three strategies compared in the report
# ---------------------------------------------------------------------------
def time_based_trades(prices, battery=Battery()):
    prices = prices.sort_values("StartTime", ignore_index=True)
    return run_strategy(prices, time_based_actions(prices["StartTime"]), "Price", battery)


def price_based_trades(prices, battery=Battery()):
    """Perfect-foresight benchmark: picks each day's slots from the real prices."""
    prices = prices.sort_values("StartTime", ignore_index=True)
    actions = price_ranked_actions(prices["Price"], prices["StartTime"].dt.date)
    return run_strategy(prices, actions, "Price", battery)


def ann_based_trades(forecast, prices, battery=Battery()):
    """Pick each day's slots from the ANN forecast, then trade at the real price.

    As in the original code the forecast row at time ``t`` is joined to the real
    price at the same ``t``. The forecast at ``t`` is a forecast of the price at
    ``t + 1 day``, so this is one day out of step; see README "Known issues".
    """
    table = forecast[["StartTime", "SettlementDate", "SettlementPeriod", "PredictedPrice"]].merge(
        prices[["StartTime", "Price"]], on="StartTime", how="left"
    ).sort_values("StartTime", ignore_index=True)
    actions = price_ranked_actions(table["PredictedPrice"], table["StartTime"].dt.date)
    return run_strategy(table, actions, "Price", battery)


def yearly_profit(trades, start=None):
    """Total profit (Cost + Revenue) per calendar year, optionally from ``start`` on."""
    if start is not None:
        trades = trades[trades["StartTime"] >= start]
    pnl = trades["Cost"] + trades["Revenue"]
    return pnl.groupby(trades["StartTime"].dt.year).sum()

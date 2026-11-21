"""Battery arbitrage model and the three trading strategies.

A strategy is just a choice of *action* for every half-hour:
``IDLE`` (0), ``IMPORT`` (1, buy and charge) or ``EXPORT`` (2, sell and discharge).
:func:`simulate` then plays those actions through the battery in time order.

Battery rules:

* The battery charges and discharges at its full grid connection, ``capacity /
  capacity_to_power_ratio`` (15 MW when new). In one half-hour that moves up to
  ``power x 0.5 h`` = 7.5 MWh, or only what still fits if it is nearly full or empty,
  so a full charge takes 5 half-hours (4 x 7.5 MWh + 3.46 MWh).
* Energy is bought or sold at that half-hour's price (GBP/MWh); the revenue from a
  sale is reduced by the round-trip efficiency. A trade with nothing to move is skipped.
* One full cycle is a state-of-charge distance of 2 (fully charge and fully
  discharge). After ``n`` full cycles the capacity is
  ``capacity * (1 - n * degradation)``, and the power falls with it.

Days and hours are UK local time, as in the GB market.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rp3 import config
from rp3.data_prep import settlement_date, uk_time

IDLE, IMPORT, EXPORT = 0, 1, 2
RESULT_COLUMNS = ["Action", "Executed", "SoC", "Energy (MWh)", "Cost", "Revenue", "Rolling Cashflow",
                  "Total Cycles Achieved"]
HOURS_PER_TRADE = 0.5      # one settlement period
MIN_TRADE_MWH = 1e-6       # smaller moves count as "nothing to move"


@dataclass(frozen=True)
class Battery:
    capacity_mwh: float = config.BATTERY_CAPACITY_MWH
    degradation_per_cycle: float = config.DEGRADATION_PER_CYCLE
    round_trip_efficiency: float = config.ROUND_TRIP_EFFICIENCY
    capacity_to_power_ratio: float = config.CAPACITY_TO_POWER_RATIO

    @property
    def power_mw(self):
        """Grid connection while new (15 MW)."""
        return self.capacity_mwh / self.capacity_to_power_ratio

    @property
    def energy_per_trade_mwh(self):
        """Most energy one half-hour can move while new (7.5 MWh)."""
        return self.power_mw * HOURS_PER_TRADE

    @property
    def slots_to_full(self):
        """Half-hours needed to fill (or empty) the battery: 5."""
        return math.ceil(self.capacity_mwh / self.energy_per_trade_mwh - 1e-9)


# ---------------------------------------------------------------------------
# Strategies: each returns one action per row
# ---------------------------------------------------------------------------
def time_based_actions(start_times, import_hours=config.IMPORT_HOURS, export_hours=config.EXPORT_HOURS):
    """Buy in a fixed night window and sell in a fixed evening window (inclusive UK local hours).

    The battery fills (or empties) in the first 5 half-hours of a window; later
    trades in the window are skipped.
    """
    hour = uk_time(start_times).hour
    actions = np.full(len(hour), IDLE, dtype=np.int8)
    actions[(hour >= import_hours[0]) & (hour <= import_hours[1])] = IMPORT
    actions[(hour >= export_hours[0]) & (hour <= export_hours[1])] = EXPORT
    return actions


def price_ranked_actions(prices, days, n_slots=Battery().slots_to_full):
    """Each day, buy in the ``n_slots`` cheapest half-hours and sell in the ``n_slots`` dearest.

    ``prices`` is what the decision is based on: the real price (perfect foresight)
    or a forecast. ``n_slots`` defaults to the half-hours needed to fill the battery.
    Ties go to the earlier half-hour. On a day with fewer than ``2 * n_slots`` rows
    a slot can be in both sets, and selling wins.
    """
    ranks = pd.DataFrame({"price": np.asarray(prices), "day": np.asarray(days)}).groupby("day")["price"]
    cheapest = ranks.rank(method="first", ascending=True) <= n_slots
    dearest = ranks.rank(method="first", ascending=False) <= n_slots
    return np.where(dearest, EXPORT, np.where(cheapest, IMPORT, IDLE)).astype(np.int8)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(actions, trade_prices, battery=Battery()):
    """Play the actions through the battery and return the trade results per row.

    This has to be a loop: how much a trade can move depends on the energy already
    stored and on the degradation from every earlier trade. It runs on plain floats,
    so the 146,000 half-hours take well under a second.
    """
    actions = np.asarray(actions)
    trade_prices = np.asarray(trade_prices, dtype=float)
    n = len(actions)
    energy = np.zeros(n)          # MWh moved: + charge, - discharge
    soc = np.zeros(n)
    cycles_done = np.zeros(n, dtype=np.int64)

    stored, distance, cycles = 0.0, 0.0, 0     # MWh stored, SoC distance travelled, full cycles
    for i, action in enumerate(actions.tolist()):
        capacity = max(battery.capacity_mwh * (1.0 - cycles * battery.degradation_per_cycle), 0.0)
        most = capacity / battery.capacity_to_power_ratio * HOURS_PER_TRADE
        if action == IMPORT:
            moved = min(most, capacity - stored)
        elif action == EXPORT:
            moved = -min(most, stored)
        else:
            moved = 0.0
        if abs(moved) < MIN_TRADE_MWH:
            moved = 0.0
        stored = min(max(stored + moved, 0.0), capacity)
        if capacity > 0:
            distance += abs(moved) / capacity
        cycles = int(distance / 2 + 1e-9)
        energy[i], soc[i], cycles_done[i] = moved, (stored / capacity if capacity else 0.0), cycles

    executed = np.where(energy > 0, IMPORT, np.where(energy < 0, EXPORT, IDLE)).astype(np.int8)
    cost = np.where(energy > 0, -energy * trade_prices, 0.0)
    revenue = np.where(energy < 0, -energy * trade_prices * battery.round_trip_efficiency, 0.0)
    cycles_before = np.concatenate(([0], cycles_done[:-1]))
    return pd.DataFrame({
        "Action": actions,
        "Executed": executed,
        "SoC": soc,
        "Energy (MWh)": np.abs(energy),
        "Cost": cost,
        "Revenue": revenue,
        "Rolling Cashflow": np.cumsum(cost + revenue),
        # Filled only on the rows where a new full cycle completes
        "Total Cycles Achieved": np.where(cycles_done > cycles_before, cycles_done, np.nan),
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
    """Perfect-foresight benchmark: picks each UK day's slots from the real prices."""
    prices = prices.sort_values("StartTime", ignore_index=True)
    actions = price_ranked_actions(prices["Price"], settlement_date(prices["StartTime"]), battery.slots_to_full)
    return run_strategy(prices, actions, "Price", battery)


def ann_based_trades(forecast, prices, battery=Battery()):
    """Pick each UK day's slots from the ANN forecast, then trade at the real price.

    Each forecast row's ``StartTime`` is the half-hour it predicts, so it is joined to
    the real price of that same half-hour.
    """
    table = forecast[["StartTime", "PredictedPrice"]].merge(
        prices[["StartTime", "SettlementDate", "SettlementPeriod", "Price"]], on="StartTime", how="inner"
    ).sort_values("StartTime", ignore_index=True)
    actions = price_ranked_actions(table["PredictedPrice"], settlement_date(table["StartTime"]), battery.slots_to_full)
    return run_strategy(table, actions, "Price", battery)


def comparison_start(ann_trades):
    """First UK midnight on or after the first ANN forecast: where the three strategies are compared."""
    first = uk_time(ann_trades["StartTime"])[0]
    midnight = first.normalize()
    return (midnight if midnight == first else midnight + pd.DateOffset(days=1)).tz_convert("UTC")


def yearly_profit(trades, start=None):
    """Total profit (Cost + Revenue) per calendar year, optionally from ``start`` on."""
    if start is not None:
        trades = trades[trades["StartTime"] >= start]
    pnl = trades["Cost"] + trades["Revenue"]
    return pnl.groupby(uk_time(trades["StartTime"]).year).sum()

"""Download the two public Elexon BMRS datasets this project is built on.

Both come from the Elexon Insights Solution API: https://bmrs.elexon.co.uk/api-documentation

* MID  - Market Index Data: half-hourly GB short-term wholesale price (GBP/MWh) and
         traded volume (MWh). Two providers report it; only APX (``APXMIDP``) has
         values, N2EX (``N2EXMIDP``) reports zeros and is dropped.
         https://bmrs.elexon.co.uk/market-index-prices
* DGWS - Day-ahead generation forecast for wind (offshore, onshore) and solar, in MW,
         published by NESO at about 17:45 UK time the day before.
         https://bmrs.elexon.co.uk/day-ahead-generation-forecast-wind-and-solar

The API returns at most 7 days per request, so the range is fetched week by week.
The files are written with the same columns as the CSV export of the BMRS website.

API key: the Insights API is open and needs no key today. If Elexon ever asks for one,
put it in ``.env`` at the repository root (git-ignored, see ``.env.example``):

    ELEXON_API_KEY=...

It is read the first time a request is made and is never printed.
"""
import os
import time
from functools import lru_cache

import pandas as pd
import requests
from dotenv import load_dotenv

from rp3 import config

API = "https://data.elexon.co.uk/bmrs/api/v1"
MID_URL = f"{API}/datasets/MID"
DGWS_URL = f"{API}/forecast/generation/wind-and-solar/day-ahead"
CHUNK = pd.Timedelta(days=7)


class ElexonError(Exception):
    """A request failed. The message never contains the API key."""


@lru_cache(maxsize=1)
def api_key(env_file=config.ENV_FILE):
    """ELEXON_API_KEY from .env, or None (the API is open, so it is optional)."""
    load_dotenv(env_file, override=True)
    return os.environ.get("ELEXON_API_KEY") or None


def _weekly_chunks(start, end):
    """Yield (from, to) pairs covering [start, end] in windows the API accepts (both ends inclusive)."""
    t = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while t <= end:
        stop = min(t + CHUNK - pd.Timedelta(minutes=30), end)
        yield t, stop
        t = stop + pd.Timedelta(minutes=30)


def _get(url, params, retries=3):
    if api_key():
        params = {**params, "apiKey": api_key()}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()["data"]
        except (requests.RequestException, KeyError) as error:
            if attempt == retries - 1:
                # Report the URL without its query string, so a key can never be printed
                raise ElexonError(f"{url} failed after {retries} attempts: {type(error).__name__}") from None
            time.sleep(2 ** attempt)


def _fetch(url, start, end, extra_params=None, pause=0.1):
    rows = []
    chunks = list(_weekly_chunks(start, end))
    for i, (a, b) in enumerate(chunks, 1):
        params = {"from": a.strftime("%Y-%m-%dT%H:%MZ"), "to": b.strftime("%Y-%m-%dT%H:%MZ"),
                  "format": "json", **(extra_params or {})}
        rows.extend(_get(url, params))
        print(f"\r  {url.rsplit('/', 1)[-1]}: week {i}/{len(chunks)}", end="", flush=True)
        time.sleep(pause)
    print()
    return pd.DataFrame(rows)


def _bmrs_dates(df):
    """BMRS website CSVs show settlement dates as dd/mm/yyyy."""
    return pd.to_datetime(df["settlementDate"]).dt.strftime("%d/%m/%Y")


def download_market_index(start, end):
    """Market Index Data (MID) for both providers, as the BMRS CSV export shows it."""
    df = _fetch(MID_URL, start, end)
    df = df.sort_values(["startTime", "dataProvider"]).drop_duplicates(["startTime", "dataProvider"])
    return pd.DataFrame({
        "StartTime": df["startTime"],
        "DataProvider": df["dataProvider"],
        "SettlementDate": _bmrs_dates(df),
        "SettlementPeriod": df["settlementPeriod"],
        "Price": df["price"],
        "Volume": df["volume"],
    }).reset_index(drop=True)


def download_wind_solar_forecast(start, end):
    """Day-ahead wind and solar generation forecast (DGWS).

    If a half-hour was forecast more than once, only the first publication is kept:
    that is the forecast that existed when the next day's trades were chosen.
    """
    df = _fetch(DGWS_URL, start, end, {"processType": "day ahead"})
    df = (df.sort_values(["startTime", "psrType", "publishTime"])
            .drop_duplicates(["startTime", "psrType"]))
    return pd.DataFrame({
        "PublishTime": df["publishTime"],
        "ProcessType": df["processType"],
        "BusinessType": df["businessType"],
        "PsrType": df["psrType"],
        "StartTime": df["startTime"],
        "SettlementDate": _bmrs_dates(df),
        "SettlementPeriod": df["settlementPeriod"],
        "Quantity": df["quantity"],
    }).reset_index(drop=True)

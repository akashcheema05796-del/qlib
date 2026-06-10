# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Deribit DVOL implied-volatility index collector for Qlib.

Downloads the Deribit DVOL index (annualised % implied-volatility) via the
public Deribit v2 REST API — no API key required.  Writes Qlib-compatible CSV
files suitable for use with ``RegimeDataHandler`` and the crypto derivatives
backtest workflow.

Key features
------------
- Supports BTC-DVOL and ETH-DVOL (and any future currency Deribit adds)
- Uses ``resolution="1D"`` for daily OHLC bars directly from Deribit
- Automatic pagination via the ``continuation`` token returned by the API
- Retry with exponential back-off on transient HTTP errors
- DVOL values stored as-is in annualised % units (e.g. 80.0 = 80 % ann. vol)
- Output format: Qlib flat-file layout
  ``<output_dir>/features/<instrument>/dvol.csv``

Usage
-----
::

    python dvol_collector.py \\
        --currencies BTC ETH \\
        --start_date 2021-03-01 \\
        --output_dir ~/.qlib/qlib_data/crypto_deribit

Notes
-----
BTC-DVOL and ETH-DVOL are both available from approximately 2021-03-01.

The instrument name written to ``instruments/dvol.txt`` uses the convention
``<currency.lower()>usdt`` (e.g. "btcusdt", "ethusdt") to align with the
OHLCV and funding-rate datasets.

Dependencies
------------
    pip install requests pandas fire
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Deribit public endpoint
# ---------------------------------------------------------------------------

_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_RETRY_DELAYS = [2, 4, 8, 16]  # seconds

# Default currencies — both available from ~2021-03
DEFAULT_CURRENCIES = ["BTC", "ETH"]


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------

def _to_ms(dt: str) -> int:
    """Convert 'YYYY-MM-DD' string to Unix milliseconds (UTC midnight)."""
    return int(datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_dvol_page(
    currency: str,
    start_ms: int,
    end_ms: int,
    resolution: str = "1D",
) -> dict:
    """Single request to the Deribit DVOL endpoint.

    Parameters
    ----------
    currency : str
        ``"BTC"`` or ``"ETH"`` (case-insensitive; uppercased internally).
    start_ms : int
        Window start in Unix milliseconds.
    end_ms : int
        Window end in Unix milliseconds.
    resolution : str
        ``"1D"`` for daily bars (default).  ``"3600"`` for hourly.

    Returns
    -------
    dict
        The full API response ``{"id": ..., "jsonrpc": ..., "result": {...}}``.
        ``result["data"]`` is a list of ``[timestamp_ms, open, high, low, close]``
        rows.  ``result["continuation"]`` is the next start timestamp (ms) or
        ``null`` when all data has been returned.

    Raises
    ------
    requests.RequestException
        If all retry attempts are exhausted.
    """
    params = {
        "currency": currency.upper(),
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "resolution": resolution,
    }
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            logger.warning(
                "Retry %d for %s-DVOL after %ds...", attempt, currency.upper(), delay
            )
            time.sleep(delay)
        try:
            resp = requests.get(_DVOL_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
            if attempt == len(_RETRY_DELAYS):
                raise
    return {}


def fetch_dvol(
    currency: str,
    start_date: str,
    end_date: str,
    resolution: str = "1D",
) -> pd.DataFrame:
    """Fetch full DVOL history for *currency* by paginating through the API.

    Parameters
    ----------
    currency : str
        Deribit currency code: ``"BTC"`` or ``"ETH"``.
    start_date, end_date : str
        Inclusive date range, ``"YYYY-MM-DD"`` format.
    resolution : str
        ``"1D"`` (default) for daily OHLC bars.

    Returns
    -------
    pd.DataFrame
        Columns: ``dvol_open``, ``dvol_high``, ``dvol_low``, ``dvol_close``;
        DatetimeIndex (UTC, tz-naive, daily).

    Notes
    -----
    Values are stored in annualised % units exactly as returned by Deribit
    (e.g. 80.0 means 80 % annualised implied volatility).  Division by 100
    is left to the downstream feature layer.
    """
    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date) + 86_400_000 - 1  # inclusive end

    all_rows: list = []
    cursor_ms = start_ms

    while True:
        response = _fetch_dvol_page(currency, cursor_ms, end_ms, resolution)
        if not response:
            break

        result = response.get("result", {})
        data = result.get("data", [])

        if not data:
            break

        for bar in data:
            # Each bar: [timestamp_ms, open, high, low, close]
            ts_ms = int(bar[0])
            all_rows.append({
                "datetime": pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_localize(None),
                "dvol_open": float(bar[1]),
                "dvol_high": float(bar[2]),
                "dvol_low": float(bar[3]),
                "dvol_close": float(bar[4]),
            })

        # Pagination: continuation is the next start timestamp (ms) or null
        continuation = result.get("continuation")
        if continuation is None:
            break
        if continuation >= end_ms:
            break
        cursor_ms = continuation

    if not all_rows:
        logger.warning(
            "No DVOL data returned for %s [%s, %s]", currency.upper(), start_date, end_date
        )
        return pd.DataFrame(
            columns=["datetime", "dvol_open", "dvol_high", "dvol_low", "dvol_close"]
        ).set_index("datetime")

    df = pd.DataFrame(all_rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Qlib flat-file writers
# ---------------------------------------------------------------------------

def _currency_to_qlib_name(currency: str) -> str:
    """Convert 'BTC' → 'btcusdt' to match OHLCV instrument names."""
    return f"{currency.lower()}usdt"


def write_qlib_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write a Qlib-compatible DVOL CSV.

    Format::

        date,dvol_open,dvol_high,dvol_low,dvol_close
        2021-03-01,82.500000,85.200000,79.100000,83.400000

    Values are in annualised % units exactly as returned by Deribit.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.index.name = "date"
    df.to_csv(out_path, float_format="%.6f")
    logger.info("Wrote %d daily DVOL bars → %s", len(df), out_path)


def write_instruments(instrument_rows: list, inst_path: Path) -> None:
    """Write ``instruments/dvol.txt`` listing available instruments.

    Format: tab-separated ``<instrument>  <start_date>  <end_date>`` per line.
    """
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inst_path, "w") as fh:
        for row in instrument_rows:
            fh.write(
                f"{row['instrument']}\t{row['start_datetime']}\t{row['end_datetime']}\n"
            )
    logger.info("Instruments list → %s", inst_path)


def write_calendar(dates: pd.DatetimeIndex, cal_path: Path) -> None:
    """Write / update a Qlib trading-calendar file.

    Merges *dates* with any existing entries; output is one ``YYYY-MM-DD``
    per line, sorted ascending.
    """
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    existing: set = set()
    if cal_path.exists():
        with open(cal_path) as fh:
            existing = {line.strip() for line in fh if line.strip()}

    new_dates = {d.strftime("%Y-%m-%d") for d in dates} | existing
    with open(cal_path, "w") as fh:
        for d in sorted(new_dates):
            fh.write(d + "\n")
    logger.info("Calendar written: %d dates → %s", len(new_dates), cal_path)


# ---------------------------------------------------------------------------
# Top-level collection entry point
# ---------------------------------------------------------------------------

def collect(
    currencies: Optional[List[str]] = None,
    start_date: str = "2021-03-01",
    end_date: Optional[str] = None,
    output_dir: str = "~/.qlib/qlib_data/crypto_deribit",
    write_calendar_file: bool = True,
) -> Path:
    """Collect Deribit DVOL index data and write Qlib flat files.

    Parameters
    ----------
    currencies : list of str, optional
        Deribit currency codes.  Defaults to ``["BTC", "ETH"]``.
        BTC-DVOL and ETH-DVOL are both available from approximately 2021-03-01.
    start_date : str
        First date to collect (inclusive), ``"YYYY-MM-DD"``.
    end_date : str, optional
        Last date to collect (inclusive).  Defaults to today (UTC).
    output_dir : str
        Root directory for Qlib CSV output.
        Layout::

            <output_dir>/
              features/<instrument>/dvol.csv
              instruments/dvol.txt
              calendars/day.txt
    write_calendar_file : bool
        If True, write / update ``calendars/day.txt`` under ``output_dir``.

    Returns
    -------
    Path
        Resolved ``output_dir`` path (suitable for ``qlib.init(provider_uri=...)``.
    """
    if currencies is None:
        currencies = DEFAULT_CURRENCIES

    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    out_root = Path(output_dir).expanduser().resolve()
    features_dir = out_root / "features"
    calendars_dir = out_root / "calendars"

    all_dates: set = set()
    instrument_rows: list = []

    for currency in currencies:
        logger.info(
            "Collecting %s-DVOL  [%s, %s]", currency.upper(), start_date, end_date
        )
        df = fetch_dvol(currency, start_date, end_date)

        if df.empty:
            logger.warning("Skipping %s-DVOL — no data.", currency.upper())
            continue

        qlib_name = _currency_to_qlib_name(currency)
        sym_dir = features_dir / qlib_name
        write_qlib_csv(df, sym_dir / "dvol.csv")

        all_dates.update(df.index.tolist())
        instrument_rows.append({
            "instrument": qlib_name,
            "start_datetime": df.index.min().strftime("%Y-%m-%d"),
            "end_datetime": df.index.max().strftime("%Y-%m-%d"),
        })

    # Write instruments/dvol.txt
    inst_path = out_root / "instruments" / "dvol.txt"
    write_instruments(instrument_rows, inst_path)

    # Write / update calendar
    if write_calendar_file and all_dates:
        sorted_dates = pd.DatetimeIndex(sorted(all_dates))
        write_calendar(sorted_dates, calendars_dir / "day.txt")

    logger.info("DVOL collection complete. Provider URI: %s", out_root)
    return out_root


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import fire
        fire.Fire(collect)
    except ImportError:
        import argparse

        parser = argparse.ArgumentParser(
            description="Deribit DVOL implied-volatility index collector for Qlib"
        )
        parser.add_argument(
            "--currencies", nargs="+", default=DEFAULT_CURRENCIES,
            help="Deribit currency codes (default: BTC ETH)",
        )
        parser.add_argument("--start_date", default="2021-03-01", help="Start date YYYY-MM-DD")
        parser.add_argument("--end_date", default=None, help="End date YYYY-MM-DD (default: today)")
        parser.add_argument(
            "--output_dir",
            default="~/.qlib/qlib_data/crypto_deribit",
            help="Root output directory for Qlib flat files",
        )
        parser.add_argument(
            "--no_calendar", action="store_true",
            help="Skip writing/updating calendars/day.txt",
        )
        args = parser.parse_args()
        collect(
            currencies=args.currencies,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            write_calendar_file=not args.no_calendar,
        )

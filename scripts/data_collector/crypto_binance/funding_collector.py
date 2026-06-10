# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Binance perpetual-futures funding rate collector for Qlib.

Downloads funding rate history from the Binance USDT-margined futures REST API —
no API key required.  Resamples the 8-hourly payments to daily aggregates and
writes Qlib-compatible CSV files.

Key features
------------
- Supports any Binance USDT-margined perpetual symbol (BTCUSDT, ETHUSDT, …)
- Automatic pagination (Binance caps each request at 1 000 records)
- Retry with exponential back-off on transient HTTP errors
- Output format: Qlib flat-file layout
  ``<output_dir>/features/<symbol>/funding.csv``

Daily aggregation
-----------------
Binance pays funding three times a day (00:00, 08:00, 16:00 UTC).  Each
daily row is the **sum** of those payments::

    funding_daily = sum(rates for that UTC day)   # fraction, not %
    funding_ann   = funding_daily * 365
    n_payments    = number of 8-h payments observed (usually 3)

Usage
-----
::

    python funding_collector.py \\
        --symbols BTCUSDT ETHUSDT \\
        --start_date 2019-09-01 \\
        --output_dir ~/.qlib/qlib_data/crypto_binance/funding

Dependencies
------------
    pip install requests pandas fire
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Binance USDT-M futures funding rate endpoint
# ---------------------------------------------------------------------------

_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_MAX_RECORDS_PER_REQUEST = 1000
_RETRY_DELAYS = [2, 4, 8, 16]  # seconds

# Default symbols — BTCUSDT data starts ~2019-09, ETHUSDT ~2020-01
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------

def _to_ms(dt: str) -> int:
    """Convert 'YYYY-MM-DD' string to Unix milliseconds (UTC midnight)."""
    return int(datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_funding_page(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = _MAX_RECORDS_PER_REQUEST,
) -> list:
    """Single paginated request to Binance funding-rate endpoint.

    Returns the raw JSON list of funding records::

        [{"fundingTime": 1569888000000,
          "fundingRate": "0.00010000",
          "symbol": "BTCUSDT"}, ...]

    Retries up to ``len(_RETRY_DELAYS)`` times on transient errors.
    """
    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            logger.warning("Retry %d for %s after %ds...", attempt, symbol, delay)
            time.sleep(delay)
        try:
            resp = requests.get(_FUNDING_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
            if attempt == len(_RETRY_DELAYS):
                raise
    return []


def fetch_funding_rates(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch full funding-rate history for *symbol* by paginating through the API.

    Parameters
    ----------
    symbol : str
        Binance USDT-M perpetual symbol, e.g. ``"BTCUSDT"``.
    start_date, end_date : str
        Inclusive date range, ``"YYYY-MM-DD"`` format.

    Returns
    -------
    pd.DataFrame
        Columns: ``fundingRate`` (float); DatetimeIndex (UTC, 8-h frequency).
    """
    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date) + 86_400_000 - 1  # inclusive end

    all_rows: list = []
    cursor = start_ms

    while cursor <= end_ms:
        page = _fetch_funding_page(symbol, cursor, end_ms)
        if not page:
            break
        for record in page:
            ts_ms = int(record["fundingTime"])
            all_rows.append({
                "datetime": pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_localize(None),
                "fundingRate": float(record["fundingRate"]),
            })
        last_ts_ms = int(page[-1]["fundingTime"])
        if len(page) < _MAX_RECORDS_PER_REQUEST:
            break
        # Advance cursor by 1 ms to avoid re-fetching the last record
        cursor = last_ts_ms + 1

    if not all_rows:
        logger.warning("No funding data returned for %s [%s, %s]", symbol, start_date, end_date)
        return pd.DataFrame(columns=["datetime", "fundingRate"]).set_index("datetime")

    df = pd.DataFrame(all_rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Daily resampler
# ---------------------------------------------------------------------------

def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 8-hourly funding payments to daily rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw 8-hourly funding rates; index is a tz-naive DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns: ``funding_daily``, ``funding_ann``, ``n_payments``;
        index is a daily DatetimeIndex.

    Notes
    -----
    ``funding_daily``
        Sum of the (up to three) 8-h payments that fall in the UTC calendar day.
    ``funding_ann``
        ``funding_daily * 365`` — a rough annualised rate.
    ``n_payments``
        Number of 8-h settlements observed (usually 3, sometimes 2 on the first
        calendar day of a new listing).
    """
    daily = df["fundingRate"].resample("D").agg(
        funding_daily="sum",
        n_payments="count",
    )
    daily["funding_ann"] = daily["funding_daily"] * 365
    # Reorder columns to the canonical output order
    daily = daily[["funding_daily", "funding_ann", "n_payments"]]
    # Drop days with no payments (e.g. future dates that crept in)
    daily = daily[daily["n_payments"] > 0]
    return daily


# ---------------------------------------------------------------------------
# Qlib flat-file writers
# ---------------------------------------------------------------------------

def _symbol_to_qlib_name(symbol: str) -> str:
    """Convert 'BTCUSDT' → 'btcusdt' (Qlib lowercase convention)."""
    return symbol.lower()


def write_qlib_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write a Qlib-compatible funding CSV.

    Format::

        date,funding_daily,funding_ann,n_payments,factor
        2021-01-01,0.000300,0.109500,3,1.000000

    ``factor`` is always 1.0 (no splits/dividends for funding data; kept for
    format consistency with other Qlib collectors).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.index.name = "date"
    df["factor"] = 1.0
    df.to_csv(out_path, float_format="%.6f")
    logger.info("Wrote %d days of funding → %s", len(df), out_path)


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
    symbols: Optional[List[str]] = None,
    start_date: str = "2019-09-01",
    end_date: Optional[str] = None,
    output_dir: str = "~/.qlib/qlib_data/crypto_binance/funding",
    write_calendar_file: bool = True,
) -> Path:
    """Collect Binance perpetual-futures funding rates and write Qlib flat files.

    Parameters
    ----------
    symbols : list of str, optional
        Binance USDT-M perpetual symbols.
        Defaults to ``["BTCUSDT", "ETHUSDT"]``.
    start_date : str
        First date to collect (inclusive), ``"YYYY-MM-DD"``.
        BTCUSDT data starts ~2019-09, ETHUSDT ~2020-01.
    end_date : str, optional
        Last date to collect (inclusive).  Defaults to today (UTC).
    output_dir : str
        Root directory for Qlib CSV output.
        Layout::

            <output_dir>/
              features/<instrument>/funding.csv
              instruments/funding.txt
              calendars/day.txt
    write_calendar_file : bool
        If True, write / update ``calendars/day.txt`` under ``output_dir``.

    Returns
    -------
    Path
        Resolved ``output_dir`` path (suitable for ``qlib.init(provider_uri=...)``.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    out_root = Path(output_dir).expanduser().resolve()
    instruments_dir = out_root / "instruments"
    features_dir = out_root / "features"
    calendars_dir = out_root / "calendars"

    all_dates: set = set()
    instrument_rows: list = []

    for symbol in symbols:
        logger.info("Collecting funding rates: %s  [%s, %s]", symbol, start_date, end_date)
        raw_df = fetch_funding_rates(symbol, start_date, end_date)

        if raw_df.empty:
            logger.warning("Skipping %s — no funding data.", symbol)
            continue

        daily_df = resample_to_daily(raw_df)

        if daily_df.empty:
            logger.warning("Skipping %s — resampling produced no rows.", symbol)
            continue

        qlib_name = _symbol_to_qlib_name(symbol)
        sym_dir = features_dir / qlib_name
        write_qlib_csv(daily_df, sym_dir / "funding.csv")

        all_dates.update(daily_df.index.tolist())
        instrument_rows.append({
            "instrument": qlib_name,
            "start_datetime": daily_df.index.min().strftime("%Y-%m-%d"),
            "end_datetime": daily_df.index.max().strftime("%Y-%m-%d"),
        })

    # Write instruments/funding.txt
    instruments_dir.mkdir(parents=True, exist_ok=True)
    inst_path = instruments_dir / "funding.txt"
    with open(inst_path, "w") as fh:
        for row in instrument_rows:
            fh.write(f"{row['instrument']}\t{row['start_datetime']}\t{row['end_datetime']}\n")
    logger.info("Instruments list → %s", inst_path)

    # Write / update calendar
    if write_calendar_file and all_dates:
        sorted_dates = pd.DatetimeIndex(sorted(all_dates))
        write_calendar(sorted_dates, calendars_dir / "day.txt")

    logger.info("Funding collection complete. Provider URI: %s", out_root)
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
            description="Binance perpetual-futures funding rate collector for Qlib"
        )
        parser.add_argument(
            "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
            help="Binance USDT-M symbols (default: BTCUSDT ETHUSDT)",
        )
        parser.add_argument("--start_date", default="2019-09-01", help="Start date YYYY-MM-DD")
        parser.add_argument("--end_date", default=None, help="End date YYYY-MM-DD (default: today)")
        parser.add_argument(
            "--output_dir",
            default="~/.qlib/qlib_data/crypto_binance/funding",
            help="Root output directory for Qlib flat files",
        )
        parser.add_argument(
            "--no_calendar", action="store_true",
            help="Skip writing/updating calendars/day.txt",
        )
        args = parser.parse_args()
        collect(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            write_calendar_file=not args.no_calendar,
        )

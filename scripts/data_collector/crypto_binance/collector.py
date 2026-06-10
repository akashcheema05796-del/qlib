# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Binance OHLCV data collector for Qlib.

Downloads public kline (candlestick) data from Binance via the REST API —
no API key required.  Writes Qlib-compatible CSV files suitable for use
with ``RegimeDataHandler`` and the crypto derivatives backtest workflow.

Key features
------------
- Supports any Binance symbol (BTCUSDT, ETHUSDT, …)
- Intervals: 1d, 4h, 1h (passed as ``--freq``)
- Pagination handled automatically (Binance caps each request at 1 000 bars)
- Retry with exponential back-off on transient HTTP errors
- Output format: Qlib flat-file layout ``<output_dir>/<symbol>.csv``

Usage
-----
::

    # Collect daily BTC + ETH from 2018 to present
    python collector.py --symbols BTCUSDT ETHUSDT --freq 1d \\
        --start_date 2018-01-01 --end_date 2024-12-31 \\
        --output_dir ~/.qlib/qlib_data/crypto_binance/1d

    # Then initialise Qlib with this data (add to your workflow script)
    import qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/crypto_binance/1d",
              region="us")   # use "us" or custom region — calendar is "D"

Dependencies
------------
    pip install requests pandas fire
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Binance public klines endpoint
# ---------------------------------------------------------------------------

_KLINES_URL = "https://api.binance.com/api/v3/klines"
_MAX_BARS_PER_REQUEST = 1000
_RETRY_DELAYS = [2, 4, 8, 16]  # seconds

# Map from human-readable freq to Binance interval string
_FREQ_MAP = {
    "1d": "1d",
    "4h": "4h",
    "1h": "1h",
    "1w": "1w",
    "15m": "15m",
}

# Qlib feature column order (matches RegimeDataHandler expectations)
_COLUMNS = ["open", "high", "low", "close", "volume"]

# Default symbols to collect when none are specified
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------

def _to_ms(dt: str) -> int:
    """Convert 'YYYY-MM-DD' string to Unix milliseconds (UTC midnight)."""
    return int(datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_klines_page(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = _MAX_BARS_PER_REQUEST,
) -> list:
    """Single paginated request to Binance klines.  Returns raw JSON list."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            logger.warning("Retry %d for %s after %ds...", attempt, symbol, delay)
            time.sleep(delay)
        try:
            resp = requests.get(_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
            if attempt == len(_RETRY_DELAYS):
                raise
    return []


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch full OHLCV history for *symbol* by paginating through the API.

    Parameters
    ----------
    symbol : str
        Binance trading pair, e.g. ``"BTCUSDT"``.
    interval : str
        Binance interval string: ``"1d"``, ``"4h"``, ``"1h"``, etc.
    start_date, end_date : str
        Inclusive date range, ``"YYYY-MM-DD"`` format.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume; DatetimeIndex (UTC).
    """
    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date) + 86_400_000 - 1  # inclusive end

    all_rows = []
    cursor = start_ms

    while cursor < end_ms:
        page = _fetch_klines_page(symbol, interval, cursor, end_ms)
        if not page:
            break
        for bar in page:
            # Binance klines: [open_time, open, high, low, close, volume, ...]
            ts_ms = int(bar[0])
            all_rows.append({
                "datetime": pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_localize(None),
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": float(bar[5]),
            })
        last_bar_open_ms = int(page[-1][0])
        if len(page) < _MAX_BARS_PER_REQUEST:
            break
        cursor = last_bar_open_ms + 1

    if not all_rows:
        logger.warning("No data returned for %s [%s, %s]", symbol, start_date, end_date)
        return pd.DataFrame(columns=["datetime"] + _COLUMNS)

    df = pd.DataFrame(all_rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Qlib flat-file writer
# ---------------------------------------------------------------------------

def _symbol_to_qlib_name(symbol: str) -> str:
    """Convert 'BTCUSDT' → 'btcusdt' (Qlib convention: lowercase)."""
    return symbol.lower()


def write_qlib_csv(df: pd.DataFrame, out_path: Path):
    """Write a Qlib-compatible CSV.

    Format:
    ::

        date,open,high,low,close,volume,factor
        2020-01-01,7195.24,7255.0,...,1.0

    ``factor`` is always 1.0 for crypto (no splits/dividends).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.index.name = "date"
    df["factor"] = 1.0
    df.to_csv(out_path, float_format="%.6f")
    logger.info("Wrote %d bars → %s", len(df), out_path)


# ---------------------------------------------------------------------------
# Qlib calendar writer
# ---------------------------------------------------------------------------

def write_calendar(dates: pd.DatetimeIndex, cal_path: Path):
    """Write a Qlib trading-calendar file containing all observed dates.

    Qlib calendar format: one date per line, ``YYYY-MM-DD``.
    """
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    existing: set = set()
    if cal_path.exists():
        with open(cal_path) as f:
            existing = {line.strip() for line in f if line.strip()}

    new_dates = {d.strftime("%Y-%m-%d") for d in dates} | existing
    with open(cal_path, "w") as f:
        for d in sorted(new_dates):
            f.write(d + "\n")
    logger.info("Calendar written: %d dates → %s", len(new_dates), cal_path)


# ---------------------------------------------------------------------------
# Top-level collection entry point
# ---------------------------------------------------------------------------

def collect(
    symbols: Optional[List[str]] = None,
    freq: str = "1d",
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    output_dir: str = "~/.qlib/qlib_data/crypto_binance/1d",
    write_calendar_file: bool = True,
):
    """Collect Binance OHLCV data and write Qlib flat files.

    Parameters
    ----------
    symbols : list of str
        Binance trading pairs.  Defaults to ``["BTCUSDT", "ETHUSDT"]``.
    freq : str
        Binance interval: ``"1d"`` (default), ``"4h"``, ``"1h"``, etc.
    start_date : str
        First bar date (inclusive), ``"YYYY-MM-DD"``.
    end_date : str, optional
        Last bar date (inclusive).  Defaults to today.
    output_dir : str
        Root directory for Qlib CSV output.
    write_calendar_file : bool
        If True, write / update ``calendars/day.txt`` under ``output_dir``.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    interval = _FREQ_MAP.get(freq)
    if interval is None:
        raise ValueError(f"Unsupported freq '{freq}'. Choose from: {list(_FREQ_MAP)}")

    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    out_root = Path(output_dir).expanduser().resolve()
    instruments_dir = out_root / "instruments"
    features_dir = out_root / "features"
    calendars_dir = out_root / "calendars"

    all_dates: set = set()
    instrument_rows = []

    for symbol in symbols:
        logger.info("Collecting %s  interval=%s  [%s, %s]", symbol, interval, start_date, end_date)
        df = fetch_ohlcv(symbol, interval, start_date, end_date)

        if df.empty:
            logger.warning("Skipping %s — no data.", symbol)
            continue

        qlib_name = _symbol_to_qlib_name(symbol)
        sym_dir = features_dir / qlib_name
        write_qlib_csv(df, sym_dir / "1d.csv")

        all_dates.update(df.index.tolist())
        instrument_rows.append({
            "instrument": qlib_name,
            "start_datetime": df.index.min().strftime("%Y-%m-%d"),
            "end_datetime": df.index.max().strftime("%Y-%m-%d"),
        })

    # Write instruments/all.txt
    instruments_dir.mkdir(parents=True, exist_ok=True)
    inst_path = instruments_dir / "all.txt"
    with open(inst_path, "w") as f:
        for row in instrument_rows:
            f.write(f"{row['instrument']}\t{row['start_datetime']}\t{row['end_datetime']}\n")
    logger.info("Instruments list → %s", inst_path)

    # Write calendar
    if write_calendar_file and all_dates:
        sorted_dates = pd.DatetimeIndex(sorted(all_dates))
        write_calendar(sorted_dates, calendars_dir / "day.txt")

    logger.info("Collection complete. Provider URI: %s", out_root)
    return out_root


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import fire
        fire.Fire(collect)
    except ImportError:
        # Minimal fallback without fire
        import argparse

        parser = argparse.ArgumentParser(description="Binance OHLCV collector for Qlib")
        parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
        parser.add_argument("--freq", default="1d")
        parser.add_argument("--start_date", default="2018-01-01")
        parser.add_argument("--end_date", default=None)
        parser.add_argument("--output_dir", default="~/.qlib/qlib_data/crypto_binance/1d")
        parser.add_argument("--no_calendar", action="store_true")
        args = parser.parse_args()
        collect(
            symbols=args.symbols,
            freq=args.freq,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            write_calendar_file=not args.no_calendar,
        )

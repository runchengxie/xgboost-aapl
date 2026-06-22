"""Alpaca Markets data source — minute and daily bars.

Supports:
- Minute bars (1Min, 5Min, 15Min, 30Min, 1Hour)
- Daily bars
- Multi-symbol fetching
- Local parquet cache (parameterised by symbol, timeframe, date range)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _get_credentials() -> tuple[str, str, str, str]:
    """Return (api_key, secret_key, base_url, data_url) from environment."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets/v2")
    if not api_key or not secret_key:
        sys.exit(
            "❌  ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.\n"
            "    export ALPACA_API_KEY='***'\n"
            "    export ALPACA_SECRET_KEY='***'"
        )
    return api_key, secret_key, base_url, data_url


# ---------------------------------------------------------------------------
# Bar fetching
# ---------------------------------------------------------------------------


def _fetch_bars(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Day",
    limit: int = 10000,
    feed: str = "iex",
    adjustment: str = "raw",
) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca Data API v2.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume,
    trade_count, vwap.  Sorted chronologically.
    """
    api_key, secret_key, _, data_url = _get_credentials()

    url = f"{data_url}/stocks/{symbol}/bars"
    params: dict[str, Any] = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": limit,
        "adjustment": adjustment,
        "feed": feed,
        "sort": "asc",
    }

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "accept": "application/json",
    }

    all_bars: list[dict[str, Any]] = []
    page_count = 0
    max_pages = 50  # safety limit

    while url and page_count < max_pages:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        bars = data.get("bars", [])
        if not bars:
            break

        all_bars.extend(bars)
        page_count += 1

        # Pagination: use page_token from response
        next_token = data.get("next_page_token")
        if next_token:
            params["page_token"] = next_token
        else:
            break

        # Rate limit: Alpaca free tier allows 200 req/min
        time.sleep(0.3)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["timestamp"] = pd.to_datetime(df["t"])
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "n": "trade_count", "vw": "vwap",
    })
    return cast(
        pd.DataFrame,
        df[["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]],
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(symbol: str, timeframe: str, start: str, end: str) -> str:
    """Normalised cache filename."""
    tf_clean = timeframe.replace(" ", "").replace("/", "_")
    return f"{symbol}_{tf_clean}_{start}_{end}.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_alpaca_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1Day",
    cache_dir: str | Path = ".cache/alpaca",
    feed: str = "iex",
) -> pd.DataFrame:
    """Fetch bars for multiple symbols from Alpaca, caching locally.

    Parameters
    ----------
    symbols : list[str]
        Stock symbols, e.g. ['AAPL', 'MSFT'].
    start_date : str
        ISO date, e.g. '2024-01-01'.
    end_date : str
        ISO date, e.g. '2025-01-01'.
    timeframe : str
        Bar granularity: '1Min', '5Min', '15Min', '1Hour', '1Day'.
    cache_dir : path
        Directory for cached parquet files.
    feed : str
        Data feed: 'iex' (free) or 'sip' (paid).

    Returns
    -------
    DataFrame with columns: symbol, timestamp, open, high, low, close, volume,
    trade_count, vwap.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        cache_file = cache_path / _cache_key(symbol, timeframe, start_date, end_date)

        if cache_file.exists():
            print(f"[alpaca:cache] {symbol} {timeframe} — loaded from {cache_file}")
            df = pd.read_parquet(cache_file)
        else:
            print(f"[alpaca:fetch] {symbol} {timeframe} {start_date} → {end_date}")
            df = _fetch_bars(symbol, start_date, end_date, timeframe, feed=feed)
            if df.empty:
                print(f"  Warning: no data for {symbol}")
                continue
            df.to_parquet(cache_file)
            print(f"[alpaca:save] → {cache_file} ({len(df)} rows)")

        df["symbol"] = symbol
        frames.append(df)

    if not frames:
        sys.exit("No data returned for any symbol.")

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["symbol", "timestamp"], inplace=True)
    return result.reset_index(drop=True)


def alpaca_daily_to_ts_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Alpaca daily DataFrame to a format compatible with the existing
    TuShare-based pipeline.

    Maps Alpaca columns to the legacy names expected by build_features():
      timestamp → trade_date
      volume   → vol
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df["trade_date"] = pd.to_datetime(df["timestamp"])
    if "volume" in df.columns:
        df["vol"] = df["volume"]
    return cast(pd.DataFrame, df)

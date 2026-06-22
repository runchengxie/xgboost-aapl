"""Feature engineering — self-contained, no pandas_ta dependency."""

from __future__ import annotations

from typing import cast

import pandas as pd


def compute_sma(series: pd.Series, length: int) -> pd.Series:  # pyright: ignore[reportReturnType]
    """Simple Moving Average."""
    return cast(pd.Series, series.rolling(window=length, min_periods=length).mean())


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:  # pyright: ignore[reportReturnType]
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return cast(pd.Series, 100.0 - (100.0 / (1.0 + rs)))


def compute_macd_hist(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram = MACD line - signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = cast(pd.Series, ema_fast - ema_slow)
    signal_line = cast(pd.Series, macd_line.ewm(span=signal, adjust=False).mean())
    return cast(pd.Series, macd_line - signal_line)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical-indicator columns to a DataFrame with OHLCV columns.

    Expects columns: close, vol.
    Returns a new DataFrame (does not mutate input).
    """
    df = df.copy()

    close_series = cast(pd.Series, df["close"])
    vol_series = cast(pd.Series, df["vol"])

    # Simple Moving Averages & their day-to-day changes
    for win in (5, 10, 20):
        sma_col = f"SMA{win}"
        df[sma_col] = compute_sma(close_series, length=win)
        df[f"{sma_col}_diff"] = df[sma_col].pct_change()

    # RSI
    df["RSI_14"] = compute_rsi(close_series, length=14)

    # MACD histogram
    df["MACD_hist"] = compute_macd_hist(close_series)

    # Volume signals
    df["Volume_SMA5"] = compute_sma(vol_series, length=5)
    df["Volume_SMA5_ratio"] = vol_series / df["Volume_SMA5"].replace(0, float("nan"))

    return df


# Public list of feature column names used for model input
FEATURE_COLUMNS = [
    "SMA5_diff",
    "SMA10_diff",
    "SMA20_diff",
    "RSI_14",
    "MACD_hist",
    "Volume_SMA5_ratio",
    "vol",
]

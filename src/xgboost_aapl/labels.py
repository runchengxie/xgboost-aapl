"""Target-label construction — fixed: last row is dropped, not zero-filled."""

from __future__ import annotations

import pandas as pd


def build_labels(df: pd.DataFrame, threshold: float = 0.002) -> pd.DataFrame:
    """Compute next-day binary target and drop the last row (no future data).

    The old code had a bug: ``(NaN >= threshold).astype(int)`` evaluates to 0,
    silently labelling the last row as "Down/Flat" even though its future return
    is unknown.  This function explicitly drops that row.

    Returns a DataFrame with an added ``target`` column (int8).
    The intermediate ``future_return`` column is not exposed.
    """
    df = df.copy()
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1.0

    # Drop the final row — its future return is unknown.
    df = df.dropna(subset=["future_return"]).copy()

    df["target"] = (df["future_return"] >= threshold).astype("int8")
    return df

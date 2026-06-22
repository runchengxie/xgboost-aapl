"""Data loading with parameterised local cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import tushare as ts

from .config import Settings


def load_data(settings: Settings) -> pd.DataFrame:
    """Fetch daily bars from TuShare, caching locally.

    Returns a DataFrame sorted chronologically by trade_date.
    """
    token = os.getenv("TUSHARE_API_KEY")
    if not token:
        sys.exit(
            "❌  TUSHARE_API_KEY environment variable is not set.\n"
            "    export TUSHARE_API_KEY='your_token'"
        )

    cache = Path(settings.cache_file)

    if cache.exists():
        print(f"♻️  加载缓存数据: {cache}")
        df = pd.read_parquet(cache)
    else:
        ts.set_token(token)
        pro = ts.pro_api()
        print(f"📥  从 TuShare 获取 {settings.symbol} 日线数据 …")
        df = pro.us_daily(
            ts_code=settings.symbol,
            start_date=settings.start_date,
            end_date=settings.end_date,
        )
        if df.empty:
            sys.exit(
                f"未获取到数据，请检查 symbol ({settings.symbol}) "
                f"和日期范围 ({settings.start_date}–{settings.end_date})。"
            )
        print(f"💾  保存缓存: {cache}")
        df.to_parquet(cache)

    df.sort_values("trade_date", inplace=True)
    return df.reset_index(drop=True)

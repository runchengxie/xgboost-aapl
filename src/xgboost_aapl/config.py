"""Configuration via dataclass — no global mutable state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    """Immutable settings for a single experiment run.

    All values have sensible defaults and can be overridden via CLI or env.
    """

    # --- Data ---
    symbol: str = "AAPL"
    start_date: str = ""  # YYYYMMDD; empty → compute from lookback_days
    end_date: str = ""    # YYYYMMDD; empty → today
    lookback_days: int = 5 * 365

    # --- Cache ---
    cache_dir: Path = Path(".cache")

    # --- Model ---
    test_size: float = 0.2
    up_threshold: float = 0.002   # +0.2 %
    random_state: int = 42

    # --- XGBoost hyper-params ---
    xgb_params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "learning_rate": 0.01,
        "max_depth": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    })

    # --- Cross-validation ---
    cv_splits: int = 5

    def __post_init__(self) -> None:
        if not self.start_date:
            start = datetime.now() - timedelta(days=self.lookback_days)
            object.__setattr__(self, "start_date", start.strftime("%Y%m%d"))
        if not self.end_date:
            object.__setattr__(self, "end_date", datetime.now().strftime("%Y%m%d"))

    @property
    def cache_file(self) -> Path:
        """Parameterised cache path: cache/{symbol}_{start}_{end}.parquet"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.symbol}_{self.start_date}_{self.end_date}.parquet"

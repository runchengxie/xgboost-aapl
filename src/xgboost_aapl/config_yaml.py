"""YAML configuration file support.

Allows loading Settings from a YAML file instead of command-line args.
Example YAML:

    symbol: AAPL
    start_date: ""
    end_date: ""
    lookback_days: 1825
    up_threshold: 0.002
    test_size: 0.2
    cv_splits: 5
    purge_days: 20
    embargo_days: 1
    model_type: xgboost
    xgb_params:
      n_estimators: 200
      learning_rate: 0.01
      max_depth: 3
      subsample: 0.7
      colsample_bytree: 0.7
      reg_alpha: 1.0
      reg_lambda: 1.0
      objective: binary:logistic
      eval_metric: logloss
      random_state: 42
    run_backtest: false
    compare_models: false
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def merge_with_defaults(
    yaml_cfg: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge YAML config into defaults.  YAML values take precedence.

    Returns a new merged dict.
    """
    merged = dict(defaults)
    for key, value in yaml_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_with_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged

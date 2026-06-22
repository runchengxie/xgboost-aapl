"""XGBoost model training with time-series cross-validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    cv_splits: int = 5,
) -> tuple[XGBClassifier, dict[str, float]]:
    """Fit XGBoost with time-series CV, return model and CV stats."""
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    cv_scores = cross_val_score(
        XGBClassifier(**params),
        X_train,
        y_train,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
    )

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    cv_stats = {
        "mean": float(np.mean(cv_scores)),
        "std": float(np.std(cv_scores)),
        "scores": [float(s) for s in cv_scores],
    }
    return model, cv_stats

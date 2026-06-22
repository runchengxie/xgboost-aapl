"""Evaluation: classification report, baselines, ROC AUC, confusion matrix, IC/ICIR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def majority_baseline(y_true: pd.Series) -> dict[str, Any]:
    """Always predict the most frequent class."""
    majority_class = int(y_true.mode().iloc[0])
    preds = np.full(len(y_true), majority_class)
    return {
        "name": "Majority",
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def persistence_baseline(
    y_true: pd.Series, prev_direction: pd.Series
) -> dict[str, Any]:
    """Predict that tomorrow's direction equals today's direction.

    *prev_direction* should be a same-length Series where each row indicates the
    most recent known direction (0 or 1) at the time the forecast would have
    been made.
    """
    preds = prev_direction.values
    return {
        "name": "Persistence (yesterday's direction)",
        "accuracy": float(accuracy_score(y_true, preds)),
    }


# ---------------------------------------------------------------------------
# IC / ICIR
# ---------------------------------------------------------------------------


def compute_ic_icir(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> dict[str, float]:
    """Compute Rank IC (Spearman) and ICIR.

    IC is the rank correlation between predicted probabilities and actual
    returns.  ICIR = mean(IC) / std(IC) is measured over a single period by
    splitting the test set into rolling sub-windows.

    Returns dict with keys: rank_ic, icir, ic_p_value.
    """
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    if mask.sum() < 10:
        return {"rank_ic": float("nan"), "icir": float("nan"), "ic_p_value": float("nan")}

    pred = predictions[mask]
    ret = actual_returns[mask]

    # Overall rank IC
    result = spearmanr(pred, ret)
    ic = result.correlation  # type: ignore[assignment]
    p_value = result.pvalue   # type: ignore[assignment]

    # ICIR: rolling sub-windows (quarterly ≈ 63 days, 1-month step)
    window = min(63, len(pred) // 3)
    step = max(21, window // 3)
    sub_ics: list[float] = []
    for start in range(0, len(pred) - window + 1, step):
        sub_result = spearmanr(
            pred[start : start + window], ret[start : start + window]
        )
        sub_ics.append(float(sub_result.correlation))  # type: ignore[arg-type]

    if len(sub_ics) >= 2:
        ic_mean = float(np.mean(sub_ics))
        ic_std = float(np.std(sub_ics, ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
    else:
        icir = float("nan")

    return {
        "rank_ic": float(ic),
        "icir": float(icir),
        "ic_p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    prev_direction_test: pd.Series | None = None,
    actual_returns_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a dictionary with all evaluation metrics."""
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    y_pred_train = (prob_train >= 0.5).astype(int)
    y_pred_test = (prob_test >= 0.5).astype(int)

    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc = float(accuracy_score(y_test, y_pred_test))

    roc_auc = float(roc_auc_score(y_test, prob_test))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    clf_report = classification_report(
        y_test, y_pred_test,
        target_names=["Down/Flat", "Up \u22650.2%"],
        digits=3,
    )

    # Baselines
    baselines = [majority_baseline(y_test)]
    if prev_direction_test is not None:
        baselines.append(persistence_baseline(y_test, prev_direction_test))

    # IC / ICIR
    ic_info: dict[str, float] = {}
    if actual_returns_test is not None:
        ic_info = compute_ic_icir(prob_test, actual_returns_test)

    # Feature importance
    importance: dict[str, float] = {}
    if feature_names and hasattr(model, "feature_importances_"):
        importance = dict(
            sorted(
                zip(feature_names, model.feature_importances_, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "roc_auc": roc_auc,
        "overfitting_gap": train_acc - test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": clf_report,
        "baselines": baselines,
        "ic": ic_info,
        "feature_importance": importance,
        "class_distribution_test": {
            "down_flat": int((y_test == 0).sum()),
            "up": int((y_test == 1).sum()),
            "up_pct": float((y_test == 1).mean()),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print an evaluation report."""
    print("\n" + "=" * 60)
    print("\U0001f4ca  EVALUATION SUMMARY")
    print("=" * 60)

    print(f"  Train Accuracy:        {report['train_accuracy']:.3f}")
    print(f"  Test Accuracy:         {report['test_accuracy']:.3f}")
    print(f"  ROC AUC:               {report['roc_auc']:.3f}")
    print(f"  Overfitting Gap:       {report['overfitting_gap']:.3f}")
    print(f"  Precision:             {report['precision']:.3f}")
    print(f"  Recall:                {report['recall']:.3f}")
    print(f"  F1 Score:              {report['f1']:.3f}")

    # IC / ICIR
    ic = report.get("ic", {})
    if ic and not np.isnan(ic.get("rank_ic", float("nan"))):
        print(f"  Rank IC:               {ic['rank_ic']:.3f}")
        print(f"  ICIR (rolling):        {ic['icir']:.3f}")
        print(f"  IC p-value:            {ic['ic_p_value']:.4f}")

    # Overfitting diagnosis
    gap = report["overfitting_gap"]
    if gap < 0.05:
        print("\n  \u2705  Low overfitting — good generalisation.")
    elif gap < 0.10:
        print("\n  \u26a0\ufe0f   Moderate overfitting — consider tuning regularisation.")
    else:
        print("\n  \u274c  High overfitting — model may be memorising training data.")

    # IC diagnosis
    if ic:
        rank_ic = ic.get("rank_ic", 0)
        icir = ic.get("icir", 0)
        if not np.isnan(rank_ic):
            if rank_ic > 0.05:
                print("  \u2705  Rank IC > 0.05 — meaningful predictive signal.")
            elif rank_ic > 0:
                print("  \u26a0\ufe0f   Rank IC > 0 but weak — marginal signal.")
            else:
                print("  \u274c  Rank IC \u2264 0 — no directional signal.")
        if not np.isnan(icir):
            if icir > 1.0:
                print("  \u2705  ICIR > 1.0 — stable signal.")
            elif icir > 0.3:
                print("  \u26a0\ufe0f   ICIR > 0.3 — modest stability.")
            else:
                print("  \u274c  ICIR < 0.3 — signal unstable across sub-periods.")

    # Baselines
    print("\n── Baselines ──")
    for b in report["baselines"]:
        name = b["name"]
        acc = b["accuracy"]
        delta = report["test_accuracy"] - acc
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<40}: {acc:.3f}  (model {sign}{delta:.3f})")

    # Class distribution
    cd = report["class_distribution_test"]
    print("\n── Test Class Distribution ──")
    print(f"  Down/Flat: {cd['down_flat']} samples ({1 - cd['up_pct']:.1%})")
    print(f"  Up \u22650.2%:  {cd['up']} samples ({cd['up_pct']:.1%})")

    # Classification report
    print("\n── Classification Report (Test) ──")
    print(report["classification_report"])

    # Confusion matrix
    cm = report["confusion_matrix"]
    print("── Confusion Matrix ──")
    print("           Pred Down   Pred Up")
    print(f"  True Down    {cm['tn']:>5}      {cm['fp']:>5}")
    print(f"  True Up      {cm['fn']:>5}      {cm['tp']:>5}")

    # Feature importance
    if report["feature_importance"]:
        print("\n── Feature Importance ──")
        for feat, imp in report["feature_importance"].items():
            print(f"  {feat:<25}: {imp:.3f}")

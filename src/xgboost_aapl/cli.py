"""Command-line entry point for the XGBoost stock prediction pipeline.

Supports:
- Purged time-series CV with embargo
- Threshold optimisation on validation set
- Multi-model comparison (XGBoost, LR, RF, LightGBM)
- Walk-forward backtest with TCA
- Factor IC analysis
- YAML configuration files
"""

from __future__ import annotations

import argparse
from typing import Any, cast

import numpy as np
import pandas as pd

from .backtest import walk_forward
from .config import Settings
from .config_yaml import load_yaml_config
from .data import load_data
from .features import FEATURE_COLUMNS, build_features
from .labels import build_labels
from .metrics import (
    compute_factor_correlation,
    compute_factor_ic,
    evaluate,
    print_report,
)
from .model import compare_models, train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost stock movement prediction")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--lookback-days", type=int, default=5 * 365)
    parser.add_argument("--threshold", type=float, default=0.002)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--purge-days", type=int, default=20,
        help="Days to purge from end of each training fold (default 20 = max rolling window)",
    )
    parser.add_argument(
        "--embargo-days", type=int, default=1,
        help="Gap days between train and test in CV (default 1)",
    )
    parser.add_argument(
        "--optimize-threshold", action="store_true",
        help="Grid-search optimal classification threshold on validation set",
    )
    parser.add_argument(
        "--compare-models", action="store_true",
        help="Compare XGBoost, LogisticRegression, RandomForest, LightGBM",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run walk-forward backtest with TCA",
    )
    parser.add_argument(
        "--backtest-cost-bps", type=float, default=5.0,
        help="Round-trip transaction cost in basis points (default 5)",
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data-source", type=str, default="tushare",
        choices=["tushare", "alpaca"],
        help="Data provider: tushare or alpaca",
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated symbols for multi-symbol mode (e.g. AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1Day",
        choices=["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"],
        help="Bar granularity for Alpaca (default: 1Day)",
    )
    return parser.parse_args(argv)


def _optimize_threshold(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, metric: str = "f1"
) -> tuple[float, float]:
    """Grid-search classification threshold on validation set."""
    prob = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import f1_score, precision_score

    scorer = {"f1": f1_score, "precision": precision_score}[metric]

    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.02):
        pred = (prob >= t).astype(int)
        s = scorer(y_val, pred)
        if s > best_score:
            best_score = s
            best_thresh = t

    return float(best_thresh), float(best_score)


def _print_model_comparison(results: list[dict[str, Any]]) -> None:
    print("\n-- Model Comparison (CV accuracy) --")
    print(f"  {'Model':<20} {'CV Mean':>8}  {'CV Std':>8}")
    for r in results:
        print(f"  {r['model_name']:<20} {r['cv_mean']:>8.3f}  {r['cv_std']:>8.3f}")


def _print_backtest(bt: dict[str, Any]) -> None:
    if "error" in bt:
        print(f"\n[backtest] {bt['error']}")
        return
    print("\n-- Walk-Forward Backtest (TCA: {:.1f} bps round-trip) --".format(
        bt.get("cost_bps_used", 5.0)
    ))
    print(f"  Trades:          {bt['n_trades']}")
    print(f"  Total Return:    {bt['total_return']:.4f}")
    print(f"  Annual Return:   {bt['annual_return']:.4f}")
    print(f"  Annual Vol:      {bt['annual_vol']:.4f}")
    print(f"  Sharpe:          {bt['sharpe']:.3f}")
    print(f"  Max Drawdown:    {bt['max_drawdown']:.4f}")
    print(f"  Win Rate:        {bt['win_rate']:.3f}")
    print(f"  Profit Factor:   {bt['profit_factor']:.3f}")
    print(f"  Turnover/yr:     {bt['turnover']:.1f}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- YAML config (args take precedence) ---
    yaml_overrides: dict[str, Any] = {}
    if args.config:
        yaml_overrides = load_yaml_config(args.config)
        print(f"[config] Loaded YAML: {args.config}")

    # --- Settings ---
    symbols = [args.symbol]
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    settings = Settings(
        data_source=args.data_source,
        symbol=symbols[0],
        symbols=symbols,
        start_date=args.start_date or yaml_overrides.get("start_date", ""),
        end_date=args.end_date or yaml_overrides.get("end_date", ""),
        lookback_days=args.lookback_days,
        up_threshold=args.threshold,
        test_size=args.test_size,
        timeframe=args.timeframe,
    )
    purge_days = args.purge_days
    embargo_days = args.embargo_days
    if not args.purge_days and "purge_days" in yaml_overrides:
        purge_days = yaml_overrides["purge_days"]
    if not args.embargo_days and "embargo_days" in yaml_overrides:
        embargo_days = yaml_overrides["embargo_days"]

    print(f"Experiment: {', '.join(symbols)}")
    print(f"  Data source: {settings.data_source}")
    if settings.data_source == "alpaca":
        print(f"  Timeframe  : {settings.timeframe}")
    print(f"  Date range : {settings.start_date} - {settings.end_date}")
    print(f"  Threshold  : {settings.up_threshold:.3f}")
    print(f"  Test size  : {settings.test_size:.0%}")
    print(f"  CV purge   : {purge_days} days")
    print(f"  CV embargo : {embargo_days} day(s)")

    # --- 1. Load data ---
    if settings.data_source == "alpaca":
        from .data_alpaca import alpaca_daily_to_ts_format, load_alpaca_data

        df = load_alpaca_data(
            symbols=settings.symbols,
            start_date=settings.start_date_iso,
            end_date=settings.end_date_iso,
            timeframe=settings.timeframe,
        )
        df = alpaca_daily_to_ts_format(df)
        print(f"  Raw rows   : {len(df)} ({df['symbol'].nunique()} symbols)")
    else:
        df = load_data(settings)
        print(f"  Raw rows   : {len(df)}")

    # --- 2. Features ---
    df = build_features(df)
    print("[features] Features built.")

    # --- 3. Labels ---
    df = build_labels(df, threshold=settings.up_threshold)

    # Select columns and drop NaN
    keep_cols = [*FEATURE_COLUMNS, "target", "future_return"]
    if "trade_date" in df.columns:
        keep_cols.append("trade_date")
    if "close" in df.columns:
        keep_cols.append("close")
    df = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)

    # --- 4. Train-test split ---
    split_idx = int(len(df) * (1 - settings.test_size))
    X_train = df.iloc[:split_idx][FEATURE_COLUMNS]
    X_test = df.iloc[split_idx:][FEATURE_COLUMNS]
    y_train = df.iloc[:split_idx]["target"]
    y_test = df.iloc[split_idx:]["target"]
    actual_returns_test = df.iloc[split_idx:]["future_return"].values

    print(f"[split] Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # --- 5. Train (with optional model comparison) ---
    if args.compare_models:
        print("[train] Comparing models (purged TS-CV) ...")
        results = compare_models(
            X_train, y_train,
            params=settings.xgb_params,
            cv_splits=settings.cv_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
        _print_model_comparison(results)
        # Use best model
        best = results[0]
        model = best["model"]
        cv_stats = {
            "mean": best["cv_mean"],
            "std": best["cv_std"],
            "scores": [],
            "n_folds": settings.cv_splits,
        }
        print(f"\n  Using best model: {best['model_name']}")
    else:
        print(
            f"[train] Training XGBoost "
            f"(purged TS-CV, purge={purge_days}d, embargo={embargo_days}d) ..."
        )
        model, cv_stats = train_model(
            X_train, y_train,
            params=settings.xgb_params,
            cv_splits=settings.cv_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
        print(f"  CV folds     : {cv_stats['n_folds']}")
        print(f"  CV Accuracy  : {cv_stats['mean']:.3f} +/- {cv_stats['std'] * 2:.3f}")

    # --- Threshold optimisation ---
    pred_threshold = 0.5
    if args.optimize_threshold:
        print("[tune] Optimising classification threshold ...")
        # Split train into train2/val (chronological)
        val_split = int(len(X_train) * 0.8)
        X_train2, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_train2, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

        model.fit(X_train2, y_train2)
        pred_threshold, best_f1 = _optimize_threshold(model, X_val, y_val)
        print(f"  Optimal threshold: {pred_threshold:.2f} (val F1={best_f1:.3f})")
        # Re-fit on full training data
        model.fit(X_train, y_train)

    # --- 6. Evaluate ---
    print("[eval] Evaluating ...")

    prev_direction_test = df.iloc[split_idx:]["target"].shift(1).fillna(0).astype(int)

    report = evaluate(
        model, X_train, X_test, y_train, y_test,
        feature_names=FEATURE_COLUMNS,
        prev_direction_test=prev_direction_test,
        actual_returns_test=actual_returns_test,
    )

    # Factor IC analysis
    df_eval = cast(pd.DataFrame, df)
    _, corr_warnings = compute_factor_correlation(df_eval, FEATURE_COLUMNS)
    factor_ic = compute_factor_ic(df_eval, FEATURE_COLUMNS)
    report["factor_ic"] = factor_ic
    report["factor_correlation_warnings"] = corr_warnings

    print_report(report)

    # --- 7. Walk-forward backtest (optional) ---
    if args.backtest:
        print("\n[backtest] Running walk-forward backtest ...")
        from xgboost import XGBClassifier
        bt = walk_forward(
            df_eval,
            feature_cols=FEATURE_COLUMNS,
            model_class=XGBClassifier,
            model_params=settings.xgb_params,
            threshold=pred_threshold,
            retrain_freq="M",
            cost_bps=args.backtest_cost_bps,
            purge_days=purge_days,
        )
        _print_backtest(bt)


if __name__ == "__main__":
    main()

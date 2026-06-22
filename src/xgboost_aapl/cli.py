"""Command-line entry point for the XGBoost stock prediction pipeline."""

from __future__ import annotations

import argparse

from .config import Settings
from .data import load_data
from .features import FEATURE_COLUMNS, build_features
from .labels import build_labels
from .metrics import evaluate, print_report
from .model import train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost AAPL stock movement prediction",
    )
    parser.add_argument(
        "--symbol", default="AAPL",
        help="Stock symbol (default: AAPL)",
    )
    parser.add_argument(
        "--start-date", default="",
        help="Start date YYYYMMDD (default: 5 years ago)",
    )
    parser.add_argument(
        "--end-date", default="",
        help="End date YYYYMMDD (default: today)",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=5 * 365,
        help="Lookback window in days (default: 1825 = 5 years)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.002,
        help="Up threshold for label (default: 0.002 = 0.2%%)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    settings = Settings(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        up_threshold=args.threshold,
        test_size=args.test_size,
    )

    print(f"\U0001f4ca  Experiment: {settings.symbol}")
    print(f"    Date range: {settings.start_date} - {settings.end_date}")
    print(f"    Threshold : {settings.up_threshold:.3f}")
    print(f"    Test size : {settings.test_size:.0%}")

    # 1. Load data
    df = load_data(settings)
    print(f"    Raw rows  : {len(df)}")

    # 2. Features
    df = build_features(df)
    print("🛠️   特征工程完成。")

    # 3. Labels (target bug fixed — last row dropped explicitly)
    df = build_labels(df, threshold=settings.up_threshold)

    # Select feature columns and drop rows with NaN (early rows where rolling
    # windows aren't ready yet).  The final row was already dropped in
    # build_labels, so we only need to drop leading NaN features.
    # Keep future_return for IC computation, drop it after split.
    cols = [*FEATURE_COLUMNS, "target", "future_return"]
    df = df[cols].dropna().reset_index(drop=True)

    # 4. Train-test split (chronological, no shuffle)
    split_idx = int(len(df) * (1 - settings.test_size))
    X_train = df.iloc[:split_idx][FEATURE_COLUMNS]
    X_test = df.iloc[split_idx:][FEATURE_COLUMNS]
    y_train = df.iloc[:split_idx]["target"]
    y_test = df.iloc[split_idx:]["target"]
    actual_returns_test = df.iloc[split_idx:]["future_return"].values

    print(f"✂️   训练集: {len(X_train)} 行, 测试集: {len(X_test)} 行")

    # 5. Train
    print("🚂  训练 XGBoost（时间序列交叉验证）…")
    model, cv_stats = train_model(
        X_train, y_train,
        params=settings.xgb_params,
        cv_splits=settings.cv_splits,
    )
    print(
        f"    CV Accuracy: {cv_stats['mean']:.3f} "
        f"± {cv_stats['std'] * 2:.3f}"
    )

    # 6. Evaluate
    print("🔍  评估模型…")

    # Build persistence baseline: yesterday's actual direction
    prev_direction_test = df.iloc[split_idx:]["target"].shift(1).fillna(0).astype(int)

    report = evaluate(
        model,
        X_train, X_test,
        y_train, y_test,
        feature_names=FEATURE_COLUMNS,
        prev_direction_test=prev_direction_test,
        actual_returns_test=actual_returns_test,
    )
    print_report(report)


if __name__ == "__main__":
    main()

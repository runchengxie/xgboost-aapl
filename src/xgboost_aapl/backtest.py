"""Walk-forward backtest with transaction cost analysis.

Implements a purged walk-forward framework: at each retrain date, the model
is refit on all prior data and used to generate signals for the next period.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_class: Any,
    model_params: dict[str, Any],
    threshold: float = 0.5,
    retrain_freq: str = "M",
    cost_bps: float = 5.0,
    purge_days: int = 20,
) -> dict[str, Any]:
    """Purged walk-forward backtest.

    Parameters
    ----------
    df : DataFrame
        Must include 'trade_date' (datetime), feature columns, 'target' (0/1),
        'future_return' (float), and a 'close' column for return calculation.
    feature_cols : list[str]
        Feature column names.
    model_class : callable
        Model constructor, e.g. ``lambda: XGBClassifier(**params)``.
    model_params : dict
        Parameters passed to model_class.
    threshold : float
        Probability threshold for generating buy signals.
    retrain_freq : str
        Pandas offset alias for retrain frequency ('M' = monthly, 'W' = weekly).
    cost_bps : float
        Round-trip transaction cost in basis points (e.g. 5 = 0.05%).
    purge_days : int
        Days to purge from end of training window before each retrain.

    Returns
    -------
    dict with keys: total_return, annual_return, annual_vol, sharpe,
    max_drawdown, win_rate, profit_factor, n_trades, turnover, equity_curve,
    monthly_returns.
    """
    if "trade_date" not in df.columns:
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df.index)

    df = df.sort_values("trade_date").reset_index(drop=True)

    # Build period boundaries
    periods = pd.date_range(
        start=df["trade_date"].min(),
        end=df["trade_date"].max(),
        freq=retrain_freq,
    )

    trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve: list[float] = [1.0]
    dates: list[pd.Timestamp] = [df["trade_date"].iloc[0]]

    for period_start in periods:
        # Train: all data before period_start, minus purge
        train_mask = df["trade_date"] < period_start
        train_idx = df[train_mask].index
        if purge_days > 0 and len(train_idx) > purge_days:
            train_idx = train_idx[:-purge_days]

        # Test: data in the next period
        period_end = period_start + pd.tseries.frequencies.to_offset(retrain_freq)
        test_mask = (df["trade_date"] >= period_start) & (df["trade_date"] < period_end)

        if train_mask.sum() < 100 or test_mask.sum() < 5:
            continue

        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, "target"]
        X_test = df.loc[test_mask, feature_cols]
        returns_test = df.loc[test_mask, "future_return"]
        test_dates = df.loc[test_mask, "trade_date"]

        # Fit
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Predict
        prob = model.predict_proba(X_test)[:, 1]
        signal = (prob >= threshold).astype(int)

        # Simulate trades
        for i in range(len(signal)):
            if signal[i] == 1:
                ret = returns_test.iloc[i]
                cost = cost_bps / 10000.0
                net_ret = ret - cost
                trades.append({
                    "date": test_dates.iloc[i],
                    "return": float(net_ret),
                })
                equity *= (1.0 + net_ret)
            equity_curve.append(equity)
            dates.append(test_dates.iloc[i])

    return _compute_backtest_metrics(trades, equity_curve, dates)


def _compute_backtest_metrics(
    trades: list[dict[str, Any]],
    equity_curve: list[float],
    dates: list[pd.Timestamp],
) -> dict[str, Any]:
    n_trades = len(trades)
    if n_trades == 0:
        return {"error": "No trades executed", "n_trades": 0}

    returns = np.array([t["return"] for t in trades])
    total_return = np.prod(1.0 + returns) - 1.0

    # Annualised
    n_years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1.0 + total_return) ** (1.0 / max(n_years, 0.25)) - 1.0
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    annual_vol = float(np.std(daily_returns) * np.sqrt(252))

    # Sharpe (zero risk-free rate)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - peak) / peak
    max_drawdown = float(drawdown.min())

    # Win rate & profit factor
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / n_trades
    profit_factor = (
        abs(wins.sum()) / abs(losses.sum()) if len(losses) > 0 else float("inf")
    )

    # Turnover: trades per year
    turnover = n_trades / max(n_years, 0.1)

    # Monthly returns for further analysis
    eq_series = pd.Series(equity_curve, index=dates)
    monthly = eq_series.resample("ME").last().pct_change().dropna()

    return {
        "n_trades": n_trades,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "turnover": float(turnover),
        "cost_bps_used": 5.0,
        "equity_curve": equity_curve,
        "monthly_returns": monthly.tolist(),
    }

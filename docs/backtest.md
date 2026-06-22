# 回测与交易成本分析

## 回测框架

当前项目是纯模型评估，不产生交易信号。以下描述计划中但尚未实现的回测方法。

### 信号生成

```
signal[t] = 1  当 predict_proba(X[t]) >= threshold
            0  其他
position[t] = signal[t]  （全仓进出，无仓位管理）
```

### 收益计算

```
strategy_return[t] = position[t-1] * (close[t] / close[t-1] - 1)
```

关键假设：

1. 信号在 t-1 收盘后产生，t 开盘执行
2. 使用次日开盘价还是收盘价取决于执行假设
3. 当前使用收盘价（学术研究中常见，实践中不现实）

### 回测指标

| 指标 | 公式 | 含义 |
|------|------|------|
| 累计收益 | `cumprod(1 + r) - 1` | 总收益 |
| 年化收益 | `(1 + total_return)^(252/n_days) - 1` | 年化 |
| 年化波动 | `std(daily_return) * sqrt(252)` | 风险 |
| Sharpe | `(annual_return - rf) / annual_vol` | 风险调整收益 |
| 最大回撤 | `max(peak - trough) / peak` | 尾部风险 |
| 胜率 | `n_win / n_trades` | 方向准确率 |
| 盈亏比 | `avg_win / avg_loss` | 赔率 |
| 换手率 | `n_trades / n_days` | 交易频率 |

## 交易成本分析（TCA）

### 美股个股成本模型

以 AAPL 为例（高流动性大盘股）：

| 成本项 | 估计 | 备注 |
|--------|------|------|
| 佣金 | $0（多数券商零佣金） | 零售账户 |
| 交易所费用 | ~0.003%（SEC 费） | 可以忽略 |
| 买卖价差 | ~0.01%（1 bp） | AAPL 流动性极高 |
| 市场冲击 | ~0.02-0.05%（2-5 bp） | 零售订单量小，冲击可忽略 |

**总往返成本估算**：~2-5 bp（0.02%-0.05%）

对于默认阈值 0.2%（20 bp），交易成本占预期收益的 10%-25%，尚可接受。若阈值降至 0.05%（5 bp），交易成本将严重侵蚀收益。

### 更现实的成本模型

对于小盘股或低流动性标的，成本模型应更保守：

```python
def estimate_cost(price: float, volume: int, is_large_cap: bool = True) -> float:
    spread = 0.0001 if is_large_cap else 0.001  # 1bp vs 10bp
    impact = 0.0005  # 零售订单冲击
    return spread + impact  # 单边成本
```

### 涨跌停和流动性约束

A 股有涨跌停板，美股无。但美股有熔断机制（仅极端情况）。对于流动性约束：

1. 如果信号要求买入但当日涨停（A 股）→ 无法成交
2. 如果成交量不足以容纳仓位 → 部分成交或滑点

## Walk-Forward 回测框架（伪代码）

```python
def walk_forward_backtest(
    df, model_class, features, retrain_freq="monthly"
):
    results = []
    for period in split_by_period(df, retrain_freq):
        train = df[df.index < period.start]
        test = df[(df.index >= period.start) & (df.index < period.end)]
        
        model = model_class()
        model.fit(train[features], train["target"])
        test["prob"] = model.predict_proba(test[features])[:, 1]
        test["signal"] = (test["prob"] >= threshold).astype(int)
        test["return"] = test["signal"].shift(1) * test["close"].pct_change()
        
        results.append(test)
    
    return pd.concat(results)
```

关键：每期用历史数据重新训练，绝不使用未来数据。

## TCA 假设总结

1. 使用收盘价计算收益（学术假设，实践中应改用次日开盘价）
2. 零佣金（美股零售券商标准）
3. 无滑点（假设流动性足够）
4. 无涨跌停约束（美股无此限制）
5. 全仓进出（无仓位管理和风险预算）

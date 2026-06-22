# 策略说明

## 策略类型

基于 XGBoost 的时序二分类策略，预测标的次日涨跌方向。

- 标的：美股个股（默认 AAPL），也支持通过 `--symbols` 传入多标的和通过 `--data-source alpaca` 使用 Alpaca 数据源
- 频率：日频
- 预测目标：次日涨跌幅是否达到阈值（默认 +0.2%）
- 模型：XGBoost 二分类器，也支持 LogisticRegression、RandomForest、LightGBM（`--compare-models`）
- 交易方向：仅做多（预测达到阈值时买入）

## 标签构建

```
future_return[t] = close[t+1] / close[t] - 1
target[t] = 1  当 future_return[t] >= threshold
             0  其他
```

关键设计决策：

1. 最后一行被显式丢弃。`close[t+1]` 对最后一行不存在，旧代码将 NaN 比较结果误标为 0（bug 已于 v0.2.0 修复）。
2. 阈值可配置。默认 0.2% 覆盖了最小价格变动和交易成本后的微利空间。
3. 阈值可通过 `--optimize-threshold` 在验证集上 grid search 自动优化（搜索范围 0.30-0.70，步长 0.02）。

## 特征工程

全部自实现，不依赖外部技术指标库。

| 特征 | 计算方式 | 经济含义 |
|------|----------|----------|
| `SMA5_diff` | 5 日均线的日变化率 | 短期趋势强度 |
| `SMA10_diff` | 10 日均线的日变化率 | 中期趋势强度 |
| `SMA20_diff` | 20 日均线的日变化率 | 长期趋势强度 |
| `RSI_14` | 14 日 Wilder RSI | 超买超卖信号 |
| `MACD_hist` | MACD 线与信号线的差值 | 动量方向与强度 |
| `Volume_SMA5_ratio` | 当日量 / 5 日均量 | 异常放量或缩量 |
| `vol` | 原始成交量 | 绝对流动性和关注度 |
| `ATR_14_pct` | ATR(14) / close | 相对波动率水平 |
| `HistVol_20` | 20 日对数收益率的年化标准差 | 历史波动率 |
| `body_ratio` | 实体 /（最高 - 最低） | 蜡烛实体占比，接近 1 为光头光脚 |
| `upper_shadow_ratio` | 上影线 /（最高 - 最低） | 上影线占比 |
| `lower_shadow_ratio` | 下影线 /（最高 - 最低） | 下影线占比 |

特征选择思路：

- 无原始价格。价格是非平稳的，直接用会导致分布漂移。所有价格特征都做了差分或比率处理。
- 无 SMA5 本身。SMA5 与价格高度相关，仅保留其差分。
- 成交量同时保留比率和绝对值。比率捕获相对变化，绝对值保留量级信息。

## 模型架构

```
XGBClassifier(
    n_estimators=200
    learning_rate=0.01
    max_depth=3
    subsample=0.7
    colsample_bytree=0.7
    reg_alpha=1.0
    reg_lambda=1.0
)
```

正则化思路：低学习率加浅树加行列采样加双重正则化，防止过拟合。

## 交叉验证

使用 `PurgedTimeSeriesSplit`（`crossval.py`），在标准时序交叉验证基础上增加了两层保护：

1. Purge（清洗）：每折训练前丢弃末尾 `purge_days` 天样本。因为滚动特征（最长窗口 20 天）使训练集末尾样本的标签可能与测试集共享同一笔未来数据。默认 purge=20。
2. Embargo（禁运期）：训练集和测试集之间预留 `embargo_days` 天间隔，防止序列自相关导致的泄漏。默认 embargo=1。

```
Fold 1: [train ── purge] || embargo || [test]
Fold 2: [train ────── purge] || embargo || [test]
        时间 →
```

## 回测

Walk-forward 回测框架已实现在 `backtest.py`。每月（可配置）用历史数据重新训练模型，在接下来的周期内产生信号并模拟交易。支持：

- 按月度或周度重训频率
- TCA 交易成本建模（往返 bps 可配置，默认 5bp）
- 回测指标：总收益、年化收益、年化波动、Sharpe、最大回撤、胜率、盈亏比、换手率

通过 `--backtest` 命令行选项启用，`--backtest-cost-bps` 调整成本假设。

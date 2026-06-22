# 运行手册

## 环境准备

```bash
cd ~/code/xgboost-aapl
uv sync --dev
source .venv/bin/activate
```

## 基本运行

```bash
# 默认参数（AAPL，过去 5 年数据，TuShare 数据源）
python -m xgboost_aapl.cli

# 自定义标的和参数
python -m xgboost_aapl.cli \
    --symbol MSFT \
    --threshold 0.005 \
    --lookback-days 1095 \
    --test-size 0.15

# 多标的对比（Alpaca 数据源）
python -m xgboost_aapl.cli \
    --data-source alpaca \
    --symbols AAPL,MSFT,GOOGL \
    --compare-models

# 完整流程：优化阈值 + 多模型对比 + walk-forward 回测
python -m xgboost_aapl.cli \
    --symbol AAPL \
    --optimize-threshold \
    --compare-models \
    --backtest \
    --backtest-cost-bps 3.0
```

## 参数说明

### 数据和标的

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | AAPL | 美股代码（单标的模式） |
| `--symbols` | 空 | 逗号分隔的多标的列表，如 `AAPL,MSFT,GOOGL` |
| `--data-source` | tushare | 数据源：`tushare` 或 `alpaca` |
| `--timeframe` | 1Day | Alpaca K 线粒度：1Min、5Min、15Min、30Min、1Hour、1Day |
| `--start-date` | 自动计算 | 起始日期 YYYYMMDD |
| `--end-date` | 今天 | 截止日期 YYYYMMDD |
| `--lookback-days` | 1825（5年） | 回看天数 |

### 模型和训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threshold` | 0.002 | 上涨阈值 |
| `--test-size` | 0.2 | 测试集比例 |
| `--purge-days` | 20 | CV 每折训练前丢弃的天数 |
| `--embargo-days` | 1 | CV 训练和测试之间的间隔天数 |

### 高级功能

| 参数 | 说明 |
|------|------|
| `--optimize-threshold` | 在验证集上 grid search 最优分类阈值 |
| `--compare-models` | 对比 XGBoost、LogisticRegression、RandomForest、LightGBM |
| `--backtest` | 运行 walk-forward 回测 |
| `--backtest-cost-bps` | 回测往返交易成本（bp），默认 5 |
| `--config` | YAML 配置文件路径 |

## 输出解读

运行后会依次输出：

```
Experiment: AAPL
  Data source: tushare
  Date range : 20210101 - 20260622
  Threshold  : 0.002
  Test size  : 20%
  CV purge   : 20 days
  CV embargo : 1 day(s)
[cache] 加载缓存数据: .cache/AAPL_20210101_20260622.parquet
  Raw rows   : 1250
[features] Features built.
[split] Train: 1000 rows, Test: 250 rows
[train] Training XGBoost (purged TS-CV, purge=20d, embargo=1d) ...
  CV folds     : 5
  CV Accuracy  : 0.523 +/- 0.062
[eval] Evaluating ...

============================================================
EVALUATION SUMMARY
============================================================
  Train Accuracy:        0.541
  Test Accuracy:         0.516
  ROC AUC:               0.512
  Overfitting Gap:       0.025
  Precision:             0.483
  Recall:                0.512
  F1 Score:              0.497
  Rank IC:               0.018
  ICIR (rolling):        0.312
  IC p-value:            0.4512

  [OK] Low overfitting -- good generalisation.
  [WARN] Rank IC > 0 but weak -- marginal signal.
  [WARN] ICIR > 0.3 -- modest stability.

-- Baselines --
  Majority                                : 0.544  (model -0.028)
  Persistence (yesterday's direction)     : 0.488  (model +0.028)

-- Test Class Distribution --
  Down/Flat: 136 samples (54.4%)
  Up >=0.2%:  114 samples (45.6%)

-- Classification Report (Test) --
              precision    recall  f1-score   support
   Down/Flat      0.521     0.567     0.543       136
  Up >=0.2%       0.483     0.437     0.459       114

-- Confusion Matrix --
           Pred Down   Pred Up
  True Down       77        59
  True Up         64        50

-- Feature Importance --
  RSI_14                  : 0.183
  MACD_hist               : 0.162
  ...

-- Factor IC Analysis --
  Feature                   Rank IC    Abs IC
  ...
```

### 关键指标

- Test Accuracy > Majority baseline：模型比「永远猜多数类」强，才有基本价值
- Rank IC > 0：预测有正向排序能力
- ICIR > 0.3：IC 有一定稳定性
- Overfitting Gap < 0.05：模型未过拟合

## 测试

```bash
pytest                          # 全部测试（14 个）
pytest tests/test_labels.py -v  # 标签正确性
pytest tests/test_features.py -v  # 特征工程
pytest tests/test_split.py -v   # 时序切分
pytest tests/test_cache.py -v   # 缓存参数化
```

### 代码质量

```bash
ruff check .                    # Lint
ruff format --check .           # 格式检查
pyright                         # 类型检查
```

## 缓存管理

缓存文件存储在 `.cache/` 目录：

| 数据源 | 路径格式 |
|--------|----------|
| TuShare | `.cache/{symbol}_{start}_{end}.parquet` |
| Alpaca | `.cache/alpaca/{symbol}_{tf}_{start}_{end}.parquet` |

如需强制刷新：

```bash
rm -rf .cache/
python -m xgboost_aapl.cli  # 重新下载
```

## 环境变量

| 变量 | 数据源 | 必需 | 说明 |
|------|--------|------|------|
| `TUSHARE_API_KEY` | tushare | 是 | TuShare API 密钥 |
| `ALPACA_API_KEY` | alpaca | 是 | Alpaca API Key |
| `ALPACA_SECRET_KEY` | alpaca | 是 | Alpaca Secret Key |
| `ALPACA_BASE_URL` | alpaca | 否 | Alpaca 交易 API 地址，默认 paper-api |
| `ALPACA_DATA_URL` | alpaca | 否 | Alpaca 数据 API 地址，默认 data.alpaca.markets |

## YAML 配置文件

可以用 YAML 文件替代命令行参数，方便保存和复用实验配置：

```yaml
# example_config.yml
symbol: AAPL
lookback_days: 1095
up_threshold: 0.005
test_size: 0.15
cv_splits: 5
purge_days: 20
embargo_days: 1
xgb_params:
  n_estimators: 200
  learning_rate: 0.01
  max_depth: 3
```

使用：

```bash
python -m xgboost_aapl.cli --config example_config.yml
```

命令行参数优先级高于 YAML 配置。

## 常见问题

### Q: 提示 `TUSHARE_API_KEY environment variable is not set`

```bash
export TUSHARE_API_KEY="your_token_here"
```

### Q: 提示 `No data returned - check symbol and date range`

- 确认 TuShare 积分充足（美股数据需要一定积分）
- 确认 symbol 格式正确（美股代码如 AAPL、MSFT）

### Q: 想用 Alpaca 数据源

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
python -m xgboost_aapl.cli --data-source alpaca --symbols AAPL,MSFT
```

### Q: 测试集 accuracy 低于 0.5

- 正常。股票预测的信噪比极低
- 关注 IC 和 ICIR 而非 accuracy
- 检查是否跑赢 baseline

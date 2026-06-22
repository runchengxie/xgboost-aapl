# 运行手册

## 环境准备

```bash
cd ~/code/xgboost-aapl
source .venv/bin/activate

# 首次设置
uv venv
uv pip install -e ".[dev]"
```

## 基本运行

```bash
# 默认参数（AAPL，过去 5 年数据）
python -m xgboost_aapl.cli

# 自定义标的、阈值、回看期
python -m xgboost_aapl.cli \
    --symbol MSFT \
    --threshold 0.005 \
    --lookback-days 1095 \
    --test-size 0.15
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | AAPL | 美股代码 |
| `--start-date` | 自动计算 | 起始日期 YYYYMMDD |
| `--end-date` | 今天 | 截止日期 |
| `--lookback-days` | 1825 (5年) | 回看天数 |
| `--threshold` | 0.002 | 上涨阈值 |
| `--test-size` | 0.2 | 测试集比例 |

## 输出解读

运行后会依次输出：

```
📊  Experiment: AAPL        ← 实验参数摘要
    Date range: 20210101 - 20260622
📥  从 TuShare 获取数据 …
🛠️   特征工程完成。
✂️   训练集: 1000 行, 测试集: 250 行
🚂  训练 XGBoost（时间序列交叉验证）…
    CV Accuracy: 0.523 ± 0.031
🔍  评估模型…
==============================
📊  EVALUATION SUMMARY
==============================
  Train Accuracy:        0.541
  Test Accuracy:         0.516
  ROC AUC:               0.512
  Overfitting Gap:       0.025
  Precision:             0.483
  Recall:                0.512
  F1 Score:              0.497
  Rank IC:               0.018
  ICIR:                  0.312
  ✅  Low overfitting — good generalisation.

── Baselines ──
  Majority                                : 0.544  (model -0.028)
  Persistence (yesterday's direction)     : 0.488  (model +0.028)
```

### 关键指标

- **Test Accuracy > Majority baseline**：模型比「永远猜多数类」强，才有基本价值
- **Rank IC > 0**：预测有正向排序能力
- **ICIR > 0.3**：IC 有一定稳定性
- **Overfitting Gap < 0.05**：模型未过拟合

## 测试

```bash
# 全部测试
pytest

# 单项测试
pytest tests/test_labels.py -v
pytest tests/test_features.py -v

# 代码质量
ruff check .
ruff format --check .
pyright
```

## 缓存管理

缓存文件存储在 `.cache/` 目录，文件名格式：

```
.cache/{symbol}_{start_date}_{end_date}.parquet
```

如需强制刷新：

```bash
rm -rf .cache/
python -m xgboost_aapl.cli  # 重新下载
```

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `TUSHARE_API_KEY` | 是 | TuShare API 密钥 |

## 常见问题

### Q: 提示 `TUSHARE_API_KEY environment variable is not set`

```bash
export TUSHARE_API_KEY="your_token_here"
```

注意：变量名是 `TUSHARE_API_KEY`，不是 `TUSHARE_TOKEN`（v0.1.0 文档曾有不一致）。

### Q: 提示 `No data returned - check symbol and date range`

- 确认 TuShare 积分充足（美股数据需要一定积分）
- 确认 symbol 格式正确（美股代码如 AAPL、MSFT）

### Q: 测试集 accuracy 低于 0.5

- 正常。股票预测的信噪比极低
- 关注 IC 和 ICIR 而非 accuracy
- 检查是否跑赢 baseline

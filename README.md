# XGBoost AAPL Stock Prediction

使用 XGBoost 预测股票次日涨跌的时序二分类策略。从原始单文件脚本重构为包结构，支持可测试、可复现、可扩展的实验流程。

## 项目结构

```text
xgboost-aapl/
├── src/xgboost_aapl/
│   ├── __init__.py        # 包信息，版本号
│   ├── config.py          # Settings dataclass（不可变配置）
│   ├── config_yaml.py     # YAML 配置文件加载
│   ├── data.py            # TuShare 数据加载 + 参数化缓存
│   ├── data_alpaca.py     # Alpaca 数据源（多标的、多时间框架）
│   ├── features.py        # 自实现技术指标（12 个特征）
│   ├── labels.py          # 标签构建（已修复最后一行 target bug）
│   ├── crossval.py        # PurgedTimeSeriesSplit（purge + embargo）
│   ├── model.py           # XGBoost 训练 + 多模型对比（LR/RF/LightGBM）
│   ├── backtest.py        # Walk-forward 回测 + TCA
│   ├── metrics.py         # 评估：baseline、ROC AUC、混淆矩阵、IC/ICIR、因子 IC、IC 衰减、因子相关性
│   └── cli.py             # 命令行入口（14 个参数，支持 YAML 配置）
├── tests/
│   ├── test_labels.py     # 标签正确性（4 个测试）
│   ├── test_features.py   # 特征工程（5 个测试）
│   ├── test_split.py      # 时序切分（2 个测试）
│   └── test_cache.py      # 缓存参数化（3 个测试）
├── docs/
│   ├── strategy.md        # 策略算法、特征、标签
│   ├── validation.md      # 防过拟合、purge/embargo、IC/ICIR、样本外检验
│   ├── backtest.md        # 回测方法、TCA、walk-forward
│   ├── runbook.md         # 运行手册、参数说明、常见问题
│   └── roadmap.md         # 未来优化方向
├── pyproject.toml         # 项目元数据、依赖、Ruff、Pyright、Pytest 配置
├── uv.lock                # 锁定的依赖版本
├── .gitignore
├── LICENSE                # MIT
└── README.md
```

## 功能速览

| 功能 | 说明 | CLI 选项 |
|------|------|----------|
| 单标的预测 | 默认 AAPL，任意美股代码 | `--symbol` |
| 多标的对比 | 同时加载多只股票，横向比较 | `--symbols AAPL,MSFT,GOOGL` |
| 双数据源 | TuShare（默认）或 Alpaca Markets | `--data-source tushare\|alpaca` |
| 多时间框架 | Alpaca 支持分钟级 K 线 | `--timeframe 1Min\|5Min\|1Hour\|1Day` |
| Purge + Embargo CV | 时序交叉验证中清洗泄漏和设置间隔 | `--purge-days` `--embargo-days` |
| 阈值优化 | 验证集上 grid search 最优分类阈值 | `--optimize-threshold` |
| 多模型对比 | XGBoost / LogisticRegression / RandomForest / LightGBM | `--compare-models` |
| Walk-forward 回测 | 逐月重训 + 模拟交易 + TCA | `--backtest` |
| YAML 配置 | 通过配置文件管理实验参数 | `--config` |
| 因子 IC 分析 | 每个特征的 Rank IC、IC 衰减、相关性矩阵 | 默认输出 |

## 快速开始

### 安装

```bash
cd ~/code/xgboost-aapl
uv sync --dev
source .venv/bin/activate
```

### 设置 API 密钥

TuShare（默认数据源）：

```bash
export TUSHARE_API_KEY="your...n```

Alpaca（可选数据源）：

```bash
export ALPACA_API_KEY="your...port ALPACA_SECRET_KEY="your...n```

### 运行

```bash
# 默认参数（AAPL，5 年数据，TuShare）
python -m xgboost_aapl.cli

# 或通过入口脚本
xgboost-aapl

# 多模型对比 + 阈值优化 + 回测
python -m xgboost_aapl.cli --optimize-threshold --compare-models --backtest

# 多标的（Alpaca 数据源）
python -m xgboost_aapl.cli --data-source alpaca --symbols AAPL,MSFT,GOOGL

# 通过 YAML 配置文件
python -m xgboost_aapl.cli --config my_experiment.yml
```

### 输出示例

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
```

## 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 方向预测准确率 |
| ROC AUC | 排序能力 |
| Precision / Recall / F1 | 类别平衡评估 |
| Confusion Matrix | 预测分布 |
| Rank IC | Spearman 相关系数（预测概率 vs 实际收益） |
| ICIR | IC / IC 标准差（滚动窗口），衡量信号稳定性 |
| Factor IC | 每个特征对未来收益的 Rank IC |
| IC Decay | IC 随前向滞后期的衰减曲线（lag 1-20） |
| Factor Correlation | 特征间相关性矩阵，自动标记 |r| > 0.8 的高相关对 |
| Majority baseline | 永远猜多数类 |
| Persistence baseline | 猜昨日方向延续 |

## 测试

```bash
pytest                          # 全部测试（14 个）
pytest tests/test_labels.py     # 只测标签
```

## 代码质量

```bash
ruff check .                    # Lint（0 错误）
ruff format --check .           # 格式检查
pyright                         # 类型检查（0 错误）
```

## 文档

详细文档见 `docs/` 目录：

- `docs/strategy.md` — 策略算法、12 个特征的设计思路、purge/embargo 交叉验证
- `docs/validation.md` — 防过拟合措施、IC/ICIR 解读、样本外检验清单
- `docs/backtest.md` — 回测框架、TCA 成本模型、walk-forward 流程
- `docs/runbook.md` — 安装、全部参数说明、YAML 配置、常见问题
- `docs/roadmap.md` — 未来优化方向和已知局限

## 版本记录

### v0.2.0

- 从单文件脚本重构为包结构，11 个模块
- 修复最后一行 target 被错误标记为 0 的 bug
- 删除 pandas_ta 依赖，所有特征自实现
- 参数化缓存（symbol + 日期范围）
- 添加 PurgedTimeSeriesSplit（purge + embargo）
- 添加多模型对比（XGBoost / LR / RF / LightGBM）
- 添加阈值优化（grid search）
- 添加 walk-forward 回测 + TCA
- 添加波动率特征（ATR_14_pct、HistVol_20）和蜡烛形态特征
- 添加因子 IC、IC 衰减、因子相关性分析
- 添加 YAML 配置文件支持
- 添加 Alpaca 数据源（多标的、多时间框架）
- 添加混淆矩阵、ROC AUC、Majority/Persistence baseline
- 添加 Rank IC 和 ICIR 指标
- 添加 Ruff + Pyright + Pytest 配置（全部通过）
- 新增 `docs/` 目录完整文档

## 许可证

MIT — 详见 LICENSE 文件。

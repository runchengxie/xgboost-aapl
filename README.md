# XGBoost AAPL Stock Prediction

使用 XGBoost 预测股票次日涨跌的时序二分类策略。从原始单文件脚本重构为包结构，支持可测试、可复现、可扩展的实验流程。

## 项目结构

```text
xgboost-aapl/
├── src/xgboost_aapl/
│   ├── __init__.py       # 包信息，版本号
│   ├── config.py         # Settings dataclass（不可变配置）
│   ├── data.py           # 数据加载 + 参数化缓存
│   ├── features.py       # 自实现技术指标（无 pandas_ta 依赖）
│   ├── labels.py         # 标签构建（已修复最后一行 target bug）
│   ├── model.py          # XGBoost 训练 + 时间序列交叉验证
│   ├── metrics.py        # 评估：baseline、ROC AUC、混淆矩阵、IC/ICIR
│   └── cli.py            # 命令行入口（argparse）
├── tests/
│   ├── test_labels.py    # 标签正确性测试
│   ├── test_features.py  # 特征工程测试
│   ├── test_split.py     # 时序切分测试
│   └── test_cache.py     # 缓存参数化测试
├── docs/
│   ├── strategy.md       # 策略算法、特征、标签
│   ├── validation.md     # 防过拟合、purge/embargo、IC/ICIR、样本外检验
│   ├── backtest.md       # 回测方法、TCA、walk-forward
│   ├── runbook.md        # 运行手册、配置、调试
│   └── roadmap.md        # 未来优化方向
├── pyproject.toml        # 项目元数据、依赖、Ruff、Pyright、Pytest 配置
├── .pre-commit-config.yaml
├── .gitignore
├── LICENSE               # MIT
├── README.md
├── environment.yml       # Conda 运行环境（精简）
└── environment-dev.yml   # Conda 开发环境（含 Jupyter、测试、lint）
```

## 快速开始

### 安装

```bash
cd ~/code/xgboost-aapl

# 使用 uv（推荐）
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate

# 或使用 Conda
conda env create -f environment-dev.yml
conda activate stock_predict
```

### 设置 API 密钥

```bash
export TUSHARE_API_KEY="your...n### 运行

```bash
# 默认参数（AAPL，5 年数据）
python -m xgboost_aapl.cli

# 或通过入口脚本
xgboost-aapl

# 自定义参数
python -m xgboost_aapl.cli --symbol MSFT --threshold 0.005 --lookback-days 1095
```

### 输出示例

```
📊  Experiment: AAPL
    Date range: 20210101 - 20260622
    Threshold : 0.002
    Test size : 20%
📥  从 TuShare 获取 AAPL 日线数据 …
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
  Rank IC:               0.018
  ICIR (rolling):        0.312
  ✅  Low overfitting — good generalisation.
  ⚠️   Rank IC > 0 but weak — marginal signal.
  ⚠️   ICIR > 0.3 — modest stability.
── Baselines ──
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
| **Rank IC** | Spearman 相关系数（预测概率 vs 实际收益） |
| **ICIR** | IC / IC 标准差（滚动窗口），衡量信号稳定性 |
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

- `docs/strategy.md` — 策略算法、特征设计、模型架构
- `docs/validation.md` — 防过拟合、purge/embargo、IC/ICIR、样本外检验清单
- `docs/backtest.md` — 回测框架、TCA 假设、walk-forward
- `docs/runbook.md` — 运行手册、参数说明、常见问题
- `docs/roadmap.md` — 未来优化方向和已知局限

## 修复记录（v0.2.0）

- 修复最后一行 target 被错误标记为 0 的 bug
- 删除 `warnings.filterwarnings("ignore")` 和 `pandas_ta` 依赖
- 参数化缓存路径（symbol + 日期范围）
- 添加混淆矩阵、ROC AUC、Majority/Persistence baseline
- 添加 Rank IC 和 ICIR 指标
- 添加 Ruff + Pyright + Pytest 配置（全部通过）
- 从单文件脚本重构为包结构
- 新增 `docs/` 目录完整文档

## 许可证

MIT — 详见 LICENSE 文件。

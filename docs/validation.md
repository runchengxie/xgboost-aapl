# 验证与防过拟合

## 核心原则

金融时序预测的最大风险是「看起来准但其实在作弊」。常见作弊方式：

1. 未来信息泄露（lookahead bias）：训练时用了测试时才有的数据
2. 过拟合噪音：模型记住了训练集噪音而非信号
3. 幸存者偏差：用全样本统计量做特征标准化
4. 阈值偷看：在测试集上调分类阈值，然后报告测试集结果

本项目采取的防护措施：

## 时序交叉验证（PurgedTimeSeriesSplit）

使用 `crossval.py` 中自定义的 `PurgedTimeSeriesSplit`，在标准时序交叉验证基础上增加了 purge 和 embargo 两层保护。

```
Fold 1: [train ── purge] || embargo || [test]
Fold 2: [train ────── purge] || embargo || [test]
        时间 →
```

- 永远用过去预测未来
- 测试集从不参与特征计算
- 每折模型独立训练

## Purge 和 Embargo

### Purge（清洗）

问题：训练集末尾的样本和测试集开头的样本，其标签可能共享同一笔未来数据。

本项目标签 `target[t]` 使用 `close[t+1]`。特征工程中使用了最长 20 天的滚动窗口（SMA20）。如果训练集末尾和测试集开头之间没有间隔，训练集最后 20 天的样本，其特征计算窗口会延伸到测试集时间范围内，导致信息泄漏。

实现：`PurgedTimeSeriesSplit` 默认 `purge_days=20`，在每折训练前丢弃训练集末尾 20 天数据。可通过 `--purge-days` 调整。

### Embargo（禁运期）

问题：训练集和测试集之间的间隔不够，测试集开头的样本可能通过序列自相关受到训练集末尾样本的影响。

对于日频数据，1 天 embargo 通常足够。本项目标签使用 `shift(-1)`，天然有 1 天间隔；此外 `PurgedTimeSeriesSplit` 在训练和测试边界额外设置 1 天间隔。默认 `embargo_days=1`，可通过 `--embargo-days` 调整。

## 信息系数（IC）

IC 衡量预测值（概率）与实际收益之间的排序关系，是量化策略中比 accuracy 更重要的指标。

### Rank IC（Spearman）

```python
from scipy.stats import spearmanr
ic, p_value = spearmanr(predictions, actual_returns)
```

- 大于 0.05：有一定预测能力
- 大于 0.10：较好的预测能力
- 负值：预测方向错误

### ICIR（Information Coefficient IR）

```
ICIR = mean(IC) / std(IC)
```

衡量 IC 的稳定性：

- 大于 0.5：可接受
- 大于 1.0：较好
- 大于 2.0：优秀

ICIR 比 IC 本身更重要：一个 IC=0.05 但 ICIR=2.0 的策略，比 IC=0.10 但 ICIR=0.3 的策略更可靠。

### 实现

代码中已集成 IC/ICIR 计算（`metrics.py` 的 `compute_ic_icir`），运行 CLI 时自动输出。此外还提供：

- `compute_factor_ic()`：每个特征对 future_return 的 Rank IC
- `compute_ic_decay()`：IC 衰减曲线（lag 1-20）
- `compute_factor_correlation()`：特征间相关性矩阵 + 自动标记 |r| 大于 0.8 的高相关对

## 样本外检验清单

运行 CI 或手动检查时，应确认以下全部通过：

| 检查项 | 方法 | 通过标准 |
|--------|------|----------|
| 时序切分 | 确认 train 日期 < test 日期 | 严格不等 |
| 无 NaN 特征 | `df.isna().sum() == 0` | 全部为 0 |
| 无 inf 特征 | `np.isinf(df).sum() == 0` | 全部为 0 |
| 标签无 NaN | y.isna().sum() == 0 | 全部为 0 |
| 最后一行已丢弃 | 标签行数 = 原始行数 - 1 | 严格相等 |
| Majority baseline | test_acc > majority_acc | 测试 > baseline |
| Persistence baseline | test_acc > persistence_acc | 测试 > baseline |
| 过拟合 gap | train_acc - test_acc | < 0.10 |
| IC 为正 | Rank IC > 0 | > 0 |
| ICIR | mean_IC / std_IC | > 0.3 |
| CV 标准差 | cv_scores.std() | < 0.05 |

## 过拟合预警信号

以下信号出现时应警惕：

1. train_acc - test_acc > 0.10
2. CV 折间标准差 > 0.05
3. 特征重要性极度集中在 1-2 个特征（> 80% 总重要性）
4. IC 在近期明显衰减
5. 预测概率分布极端（大量 0 或 1，缺乏中间值）

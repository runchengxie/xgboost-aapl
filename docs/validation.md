# 验证与防过拟合

## 核心原则

金融时序预测的最大风险不是「模型不准」，而是「看起来准但其实在作弊」。常见作弊方式：

1. **未来信息泄露**（lookahead bias）：训练时用了测试时才有的数据
2. **过拟合噪音**：模型记住了训练集噪音而非信号
3. **幸存者偏差**：用全样本统计量做特征标准化
4. **阈值偷看**：在测试集上调分类阈值，然后报告测试集结果

本项目采取的防护措施：

## 时序交叉验证（TimeSeriesSplit）

```
Fold 1: |████████████|░░░░|              train
Fold 2: |███████████████████|░░░░|        train
Fold 3: |██████████████████████████|░░░░|  train
        时间 →
```

- 永远用过去预测未来
- 测试集从不参与特征计算
- 每折模型独立训练

## Purge & Embargo 分析

### Purge（清洗）

问题：训练集末尾的样本和测试集开头的样本，其标签可能共享同一笔未来数据。

本项目中，标签 `target[t]` 使用 `close[t+1]`。在标准 `TimeSeriesSplit` 中，fold k 的测试集从 `t_k` 开始，fold k+1 的训练集包含到 `t_{k+1}-1`。

如果 `t_k` 和 `t_{k+1}` 之间没有间隙，fold k 的测试样本 `t_k` 的标签用到了 `close[t_k+1]`，而这个 `close[t_k+1]` 可能也是 fold k+1 训练集中某样本的特征计算的一部分（如 SMA20 的滚动窗口）。

**当前状态**：未实现显式 purge。由于特征使用 20 日滚动窗口，理论上需要 purge 最近 20 个训练样本。

**建议**：在训练每折前，丢弃训练集末尾最近 `max_window` 天的样本：

```python
purge_days = 20  # max rolling window
X_train_fold = X_train_fold.iloc[:-purge_days]
y_train_fold = y_train_fold.iloc[:-purge_days]
```

### Embargo（禁运期）

问题：训练集和测试集之间的间隔不够，导致测试集开头的样本受训练集末尾样本的影响（通过序列自相关）。

对于日频数据，1-2 天的 embargo 通常足够。本项目标签使用 `shift(-1)`，天然有 1 天间隔，但严格来说应在训练/测试边界加至少 1 天 embargo。

**当前状态**：未实现。`TimeSeriesSplit` 的 gap 参数可配置但本项目未使用。

## 信息系数（IC）

IC 衡量预测值（概率）与实际收益之间的排序关系，是量化策略中比 accuracy 更重要的指标。

### Rank IC（Spearman）

```python
from scipy.stats import spearmanr
ic, p_value = spearmanr(predictions, actual_returns)
```

- > 0.05：有一定预测能力
- > 0.10：较好的预测能力
- 负值：预测方向错误

### ICIR（Information Coefficient IR）

```
ICIR = mean(IC) / std(IC)
```

衡量 IC 的稳定性：

- > 0.5：可接受
- > 1.0：较好
- > 2.0：优秀

ICIR 比 IC 本身更重要：一个 IC=0.05 但 ICIR=2.0 的策略比 IC=0.10 但 ICIR=0.3 的策略更可靠。

### 实现

代码中已集成 IC/ICIR 计算，运行 `python -m xgboost_aapl.cli` 时自动输出。

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

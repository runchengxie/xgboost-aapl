# 未来优化方向

按优先级从高到低排列。已完成的标记 [x]。

## P0：验证加强

- [x] Purge 实现：`crossval.py` — `PurgedTimeSeriesSplit`，每折训练前丢弃 `purge_days` 天
- [x] Embargo 实现：`crossval.py` — `embargo_days` 参数，训练/测试边界预留间隔
- [x] Walk-forward 回测：`backtest.py` — `walk_forward()` 逐月重训，含 TCA
- [x] TCA 实际成本：`backtest.py` — 往返成本 `cost_bps` 参数（默认 5bp）

## P1：模型改进

- [x] 阈值优化：`cli.py --optimize-threshold` — 验证集上 grid search 最优阈值
- [x] 特征扩展：
  - 波动率特征（ATR_14、ATR_14_pct、HistVol_20）
  - 蜡烛形态（body_ratio、upper_shadow_ratio、lower_shadow_ratio）
- [x] 多模型对比：`cli.py --compare-models` — XGBoost / LR / RF / LightGBM
- [ ] 多标的：从单标的扩展到多标的横向比较（Alpaca 路径已支持，TuShare 路径待扩展）
- [ ] 概率校准：`CalibratedClassifierCV` 使概率更准确

## P2：执行与风控

- [ ] 仓位管理：Kelly 公式、风险平价
- [ ] 止损/止盈：基于波动率的动态止损
- [ ] 信号过滤：低置信度信号不交易
- [ ] 多时间框架：日线信号 + 小时线确认

## P3：工程化

- [x] 配置文件：`config_yaml.py` — YAML 配置文件支持，`cli.py --config`
- [ ] 实验追踪：MLflow 或 W&B 记录每次实验的参数和指标
- [ ] 数据版本控制：DVC 管理数据和模型版本
- [ ] CI/CD：GitHub Actions 自动化测试 + 模型验证
- [ ] API 服务：FastAPI 提供实时预测接口

## P4：量化研究

- [x] 因子 IC 分析：`metrics.py — compute_factor_ic()` 每个特征的 Rank IC
- [x] 因子相关性矩阵：`metrics.py — compute_factor_correlation()`，自动标记 |r| > 0.8
- [x] IC 衰减曲线：`metrics.py — compute_ic_decay()` lag 1-20
- [ ] 行业中性：如果用行业数据，做行业中性化
- [ ] 市场状态分类：牛/熊/震荡市分别建模
- [ ] 另类数据：新闻情绪、社交媒体、期权数据

## 测试覆盖缺口

当前测试仅覆盖 labels、features、split、cache 四个模块。以下模块无测试：

- `model.py`（train_model、compare_models）
- `metrics.py`（evaluate、compute_ic_icir、compute_factor_ic、compute_factor_correlation、compute_ic_decay）
- `crossval.py`（PurgedTimeSeriesSplit、purged_cross_val_score）
- `backtest.py`（walk_forward）

## 已知局限

1. 单标的：AAPL 的结果不代表其他股票
2. 日频：无法捕捉日内模式
3. 无宏观变量：利率、VIX 等未纳入
4. 样本量小：5 年日线约 1250 个样本，对 ML 来说偏少
5. 幸存者偏差：只用了存续至今的 AAPL
6. 收盘价执行：实际不可能以收盘价成交

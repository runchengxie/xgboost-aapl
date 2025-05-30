# XGBoost AAPL Stock Prediction

一个使用XGBoost机器学习算法预测苹果公司(AAPL)股票次日涨跌的项目。该模型专注于预测股价是否会在次日上涨≥0.2%。

## 📊 项目概述

本项目使用技术指标和历史价格数据训练XGBoost分类器，预测AAPL股票的短期价格走势。模型采用时间序列交叉验证来确保预测的可靠性。

### 主要特性

- 🎯 **二元分类**: 预测次日股价是否上涨≥0.2%
- 📈 **技术指标**: 使用SMA、RSI、MACD等多种技术分析指标
- ⏰ **时间序列验证**: 采用TimeSeriesSplit进行交叉验证
- 💾 **数据缓存**: 自动缓存历史数据，避免重复下载
- 🔧 **防过拟合**: 使用正则化和保守的超参数设置

## 🛠️ 环境设置

### 前置要求

- Python 3.11+
- Conda 或 Miniconda
- Tushare API密钥

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd xgboost-aapl
   ```

2. **创建Conda环境**
   ```bash
   conda env create -f environment.yml
   conda activate stock_predict
   ```

3. **设置API密钥**
   
   获取Tushare API密钥：
   - 访问 [Tushare官网](https://tushare.pro/)
   - 注册账户并获取API密钥
   
   设置环境变量：
   ```bash
   # Windows (PowerShell)
   $env:TUSHARE_API_KEY="your_api_key_here"
   
   # Windows (CMD)
   set TUSHARE_API_KEY=your_api_key_here
   
   # Linux/Mac
   export TUSHARE_API_KEY="your_api_key_here"
   ```

## 🚀 使用方法

### 基本运行

```bash
python main.py
```

### 运行流程

1. **数据获取**: 自动下载AAPL过去5年的日线数据
2. **特征工程**: 计算技术指标和衍生特征
3. **模型训练**: 使用时间序列交叉验证训练XGBoost模型
4. **模型评估**: 在测试集上评估模型性能
5. **结果分析**: 输出详细的性能指标和特征重要性

## 📈 技术指标

模型使用以下技术指标作为特征：

- **简单移动平均线 (SMA)**: 5日、10日、20日
- **SMA变化率**: 各周期SMA的日变化率
- **相对强弱指数 (RSI)**: 14日RSI
- **MACD柱状图**: MACD直方图值
- **成交量比率**: 当日成交量与5日平均成交量的比值
- **成交量**: 原始成交量数据

## 🎯 模型配置

### XGBoost超参数

```python
XGB_PARAMS = {
    'n_estimators': 200,        # 树的数量
    'learning_rate': 0.01,      # 学习率
    'max_depth': 3,             # 最大树深度
    'subsample': 0.7,           # 样本采样率
    'colsample_bytree': 0.7,    # 特征采样率
    'reg_alpha': 1.0,           # L1正则化
    'reg_lambda': 1.0,          # L2正则化
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42
}
```

### 关键参数

- **预测阈值**: 0.2% (UP_THRESHOLD = 0.002)
- **测试集比例**: 20% (TEST_SIZE = 0.2)
- **数据范围**: 过去5年的历史数据
- **交叉验证**: 5折时间序列交叉验证

## 📊 输出结果

运行后将显示：

1. **交叉验证结果**: CV准确率和标准差
2. **测试集性能**: 分类报告和准确率
3. **训练集性能**: 用于检测过拟合
4. **特征重要性**: 各特征对预测的贡献度
5. **模型诊断**: 过拟合检测和性能总结

## 📁 项目结构

```
xgboost-aapl/
├── main.py              # 主程序文件
├── environment.yml      # Conda环境配置
├── data_cache.parquet   # 缓存的历史数据
├── README.md           # 项目说明文档
└── .gitattributes      # Git配置文件
```

## ⚠️ 注意事项

1. **仅供学习**: 本项目仅用于教育和研究目的，不构成投资建议
2. **数据延迟**: Tushare数据可能存在延迟，实际交易请使用实时数据
3. **市场风险**: 股票投资存在风险，过往表现不代表未来收益
4. **模型局限**: 机器学习模型无法预测所有市场情况，请谨慎使用

## 🔧 故障排除

### 常见问题

1. **API密钥错误**
   ```
   ❌ Please set the TUSHARE_API_KEY environment variable first!
   ```
   解决方案：检查环境变量设置是否正确

2. **数据下载失败**
   - 检查网络连接
   - 验证API密钥有效性
   - 确认Tushare账户积分充足

3. **依赖包问题**
   ```bash
   conda env update -f environment.yml
   ```

## 📝 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

---

**免责声明**: 本项目仅用于教育目的，不构成任何投资建议。投资有风险，入市需谨慎。
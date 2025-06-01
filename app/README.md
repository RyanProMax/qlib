# 🚀 Qlib多模型智能量化分析系统

>
> 以下内容均由AI自动生成，可能有描述不准确的地方，以实际使用情况为准
>

基于Docker的专业股票量化分析平台，集成qlib框架和多种机器学习模型，提供完整的技术分析和智能预测功能。

## 📊 系统特性

- **🎯 多模型融合**: 集成15+机器学习模型 + 5种技术分析模型
- **🔧 动态权重分配**: 基于模型置信度智能调整权重
- **📈 实时股票分析**: 支持任意美股股票的深度分析
- **🐳 Docker容器化**: 一键部署，环境隔离，跨平台兼容
- **📊 可视化报告**: 自动生成图表和Markdown分析报告
- **⚡ 高性能计算**: 优化的数据处理和模型训练流程

## 🏗️ 系统架构

```
📦 qlib-analysis-system/
├── 🐳 Docker环境
│   ├── qlib核心框架
│   ├── PyTorch + XGBoost + CatBoost
│   └── 科学计算栈 (pandas, numpy, sklearn)
├── 🤖 机器学习模型
│   ├── qlib原生模型 (LGB, XGB, GRU, LSTM, Linear)
│   └── 第三方模型 (MLP, Ridge, SVM, RandomForest等)
├── 📈 技术分析引擎
│   ├── 趋势跟踪 (MA, MACD)
│   ├── 动量振荡 (RSI, ROC)
│   └── 量价分析 (Volume, Bollinger)
└── 📊 智能决策系统
    ├── 动态权重分配
    ├── 风险评估
    └── 投资建议生成
```

## 🔧 系统要求

### 主机环境

- **操作系统**: Windows 10/11, macOS, Linux
- **Docker**: Docker Desktop 4.0+
- **Python**: Python 3.8+ (用于运行控制脚本)
- **内存**: 建议8GB+
- **存储**: 至少5GB可用空间

### 网络要求

- 能够访问Docker Hub (拉取镜像)
- 能够访问Yahoo Finance API (获取股票数据)
- 端口8888可用 (Jupyter Notebook, 可选)

## 🚀 快速开始

### 第1步: 环境准备

1. **安装Docker Desktop**

   ```bash
   # Windows/macOS: 下载Docker Desktop官方安装包
   # https://www.docker.com/products/docker-desktop
   
   # Linux (Ubuntu/Debian):
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   ```

2. **验证Docker安装**

   ```bash
   docker --version
   docker-compose --version
   ```

3. **克隆项目**

   ```bash
   git clone <项目地址>
   cd qlib-analysis
   ```

### 第2步: Docker环境部署

1. **启动Docker容器**

   ```bash
   # 使用docker-compose一键启动
   docker-compose up -d
   
   # 或者手动启动（如果没有docker-compose.yml）
   docker run -d --name qlib_jupyter \
     -v $(pwd):/workspace \
     -p 8888:8888 \
     microsoft/qlib:latest
   ```

2. **验证容器运行状态**

   ```bash
   docker ps
   # 应该看到 qlib_jupyter 容器在运行
   ```

3. **检查qlib安装**

   ```bash
   python run_in_docker.py --command "python -c 'import qlib; print(f\"✅ Qlib版本: {qlib.__version__}\")'"
   ```

### 第3步: 依赖安装

系统会自动检测并安装缺失的依赖包：

**核心依赖** (容器中预装):

- `qlib` - 量化分析框架
- `torch` - PyTorch深度学习
- `xgboost` - 梯度提升模型
- `catboost` - CatBoost模型
- `pytorch-tabnet` - TabNet模型

**数据科学栈**:

- `pandas, numpy` - 数据处理
- `scikit-learn` - 机器学习
- `matplotlib, seaborn` - 可视化
- `yfinance` - 股票数据获取

**手动安装额外依赖** (如需要):

```bash
python run_in_docker.py --command "pip install lightgbm optuna ta-lib"
```

## 📈 使用方法

### 核心脚本说明

- **`run_in_docker.py`** - Docker环境控制脚本
- **`tsla_analysis_2025.py`** - 主要分析脚本 (可修改股票代码)

### 基本使用方式

#### 1. 快速股票分析

```bash
# 分析TSLA股票 (默认)
python run_in_docker.py --command "python tsla_analysis_2025.py"

# 分析其他股票 (修改脚本中的股票代码)
# 将脚本中的 "TSLA" 改为 "AAPL", "MSFT", "GOOGL" 等
```

#### 2. 交互式分析

```bash
# 进入Docker容器进行交互式分析
python run_in_docker.py

# 在容器中运行
python tsla_analysis_2025.py
```

#### 3. 自定义命令执行

```bash
# 检查模型状态
python run_in_docker.py --command "python -c 'import qlib.contrib.model.gbdt as gbdt; print(\"✅ GBDT模型可用\")'"

# 获取股票数据
python run_in_docker.py --command "python -c 'import yfinance as yf; print(yf.download(\"AAPL\", period=\"1mo\").tail())'"

# 查看系统状态
python run_in_docker.py --command "python -c 'import sys; print(f\"Python: {sys.version}\"); import torch; print(f\"PyTorch: {torch.__version__}\")'"
```

### 高级功能

#### 1. Jupyter Notebook分析

```bash
# 启动Jupyter Notebook服务
python run_in_docker.py --jupyter

# 访问 http://localhost:8888 进行交互式分析
```

#### 2. 批量股票分析

创建批量分析脚本：

```python
# batch_analysis.py
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
for stock in stocks:
    # 修改分析脚本中的股票代码并运行
    print(f"正在分析 {stock}...")
```

```bash
python run_in_docker.py --script batch_analysis.py
```

#### 3. 定制化分析

修改 `tsla_analysis_2025.py` 中的参数：

- **股票代码**: 第185行 `tsla_data = yf.download("TSLA", ...)`
- **分析周期**: 第182行 `start_date = datetime(2023, 1, 1)`
- **预测天数**: prepare_ml_data函数中的 `prediction_days=5`

## 📊 输出文件说明

分析完成后，系统会生成以下文件：

```
📁 /workspace/output/
├── 📊 plots/
│   └── tsla_comprehensive_analysis.png  # 综合分析图表
└── 📄 tsla_analysis_report.md          # 详细分析报告
```

### 报告内容包括

- **💡 投资建议总结** - 核心建议和关键指标
- **📈 技术分析详情** - 5种技术指标的详细分析
- **🤖 机器学习预测** - 15个模型的预测结果
- **⚖️ 动态权重分配** - 模型权重分配详情
- **⚠️ 风险提示** - 投资风险评估

## 🔍 故障排除

### 常见问题

1. **Docker容器无法启动**

   ```bash
   # 检查Docker状态
   docker --version
   docker ps -a
   
   # 重新启动Docker Desktop
   # 清理旧容器
   docker container prune
   ```

2. **模型导入失败**

   ```bash
   # 检查模型导入状态
   python run_in_docker.py --command "python -c 'import qlib.contrib.model.gbdt; print(\"✅ qlib模型正常\")'"
   
   # 重新安装依赖
   python run_in_docker.py --command "pip install --upgrade qlib torch xgboost catboost"
   ```

3. **股票数据获取失败**

   ```bash
   # 检查网络连接
   python run_in_docker.py --command "python -c 'import yfinance as yf; print(yf.download(\"AAPL\", period=\"1d\"))'"
   
   # 更换数据源或检查代理设置
   ```

4. **内存不足**

   ```bash
   # 增加Docker内存限制 (Docker Desktop设置)
   # 或减少模型数量，修改脚本中的model_imports列表
   ```

5. **端口冲突**

   ```bash
   # 修改端口映射
   docker run -p 8889:8888 ...  # 使用8889端口
   ```

### 调试命令

```bash
# 查看容器日志
docker logs qlib_jupyter

# 进入容器调试
docker exec -it qlib_jupyter bash

# 检查系统资源
python run_in_docker.py --command "python -c 'import psutil; print(f\"内存使用: {psutil.virtual_memory().percent}%\")'"

# 测试模型导入
python run_in_docker.py --command "python -c 'from sklearn.ensemble import RandomForestRegressor; print(\"✅ sklearn正常\")'"
```

## 📚 高级配置

### 自定义模型配置

在 `tsla_analysis_2025.py` 中修改模型参数：

```python
# 添加新的第三方模型
('CustomModel', 'sklearn.ensemble', 'ExtraTreesRegressor', 'sklearn_direct'),

# 修改模型参数
elif model_name == 'RandomForestModel':
    model = model_class(n_estimators=200, max_depth=15, random_state=42)
```

### 技术指标自定义

```python
# 添加新的技术指标
df['STOCH_K'] = stoch_k_calculation(df)
df['ATR'] = atr_calculation(df)

# 修改信号逻辑
if latest['RSI'] < 25:  # 更严格的超卖条件
    score += 3
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Microsoft Qlib](https://github.com/microsoft/qlib) - 核心量化框架
- [PyTorch](https://pytorch.org/) - 深度学习支持
- [scikit-learn](https://scikit-learn.org/) - 机器学习算法
- [yfinance](https://github.com/ranaroussi/yfinance) - 股票数据源

## 📞 支持

如果您遇到问题或有功能建议，请：

1. 查看 [故障排除](#故障排除) 部分
2. 在 GitHub 上提出 Issue
3. 查看项目 Wiki 获取更多文档

---

**⚡ 快速开始命令总结**:

```bash
# 1. 启动Docker环境
docker-compose up -d

# 2. 运行TSLA分析
python run_in_docker.py --command "python tsla_analysis_2025.py"

# 3. 查看结果
python run_in_docker.py --command "ls -la /workspace/output/"
```

🎯 **现在您拥有了一个专业级的量化分析系统！** 🚀

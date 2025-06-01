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

## 🚀 快速开始

### 第1步: 环境准备

1. **安装Docker Desktop**
   ```bash
   # Windows/macOS: 下载Docker Desktop官方安装包
   # https://www.docker.com/products/docker-desktop
   ```

2. **验证Docker安装**
   ```bash
   docker --version
   docker-compose --version
   ```

### 第2步: 启动Docker环境

```bash
# 在项目根目录启动Docker容器
docker-compose up -d

# 验证容器运行状态
docker ps
```

### 第3步: 运行股票分析

```bash
# 进入app目录
cd ./app

# 运行TSLA股票分析
python run_in_docker.py --script tsla_analysis_2025.py
```

## 📈 使用方法

### 核心命令

```bash
# 1. 进入app目录
cd ./app

# 2. 运行股票分析
python run_in_docker.py --script tsla_analysis_2025.py
```

### 其他常用命令

```bash
# 检查系统状态
python run_in_docker.py --status

# 进入交互模式
python run_in_docker.py

# 执行自定义命令
python run_in_docker.py --command "python -c 'import qlib; print(qlib.__version__)'"
```

### 修改股票代码

要分析其他股票，请修改 `tsla_analysis_2025.py` 文件中的股票代码：

- 找到第185行左右: `tsla_data = yf.download("TSLA", ...)`
- 将 `"TSLA"` 改为其他股票代码，如 `"AAPL"`, `"MSFT"`, `"GOOGL"` 等

## 📊 输出文件

分析完成后，结果保存在：

```
📁 output/
├── 📊 plots/
│   └── tsla_comprehensive_analysis.png  # 综合分析图表
└── 📄 tsla_analysis_report.md          # 详细分析报告
```

## 🔍 故障排除

### 常见问题

1. **Docker容器无法启动**
   ```bash
   docker ps -a
   docker container prune
   ```

2. **模型导入失败**
   ```bash
   python run_in_docker.py --command "pip install --upgrade qlib torch xgboost catboost"
   ```

3. **股票数据获取失败**
   ```bash
   python run_in_docker.py --command "python -c 'import yfinance as yf; print(yf.download(\"AAPL\", period=\"1d\"))'"
   ```

## 📚 脚本说明

- **`run_in_docker.py`** - Docker环境控制脚本
- **`tsla_analysis_2025.py`** - 主要分析脚本，可修改股票代码

## ⚡ 完整流程

```bash
# 1. 启动Docker环境（项目根目录）
docker-compose up -d

# 2. 进入app目录并运行分析
cd ./app
python run_in_docker.py --script tsla_analysis_2025.py

# 3. 查看结果
python run_in_docker.py --command "ls -la /workspace/output/"
```

🎯 **简单高效的量化分析系统！** 🚀

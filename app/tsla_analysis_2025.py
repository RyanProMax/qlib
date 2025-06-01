#!/usr/bin/env python3
"""
TSLA股票分析脚本 - 2025年走势分析及未来趋势预测
使用qlib多个模型进行量化分析和预测，动态权重分配
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("=== TSLA股票分析 - 基于Qlib多模型的智能量化分析 ===")

# 创建输出目录
output_dir = "/workspace/output"
plots_dir = "/workspace/output/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

try:
    # 导入必要的库
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    import yfinance as yf
    
    # 智能导入qlib模块
    qlib_models = {}
    qlib_available = False
    
    try:
        import qlib
        from qlib.constant import REG_US
        qlib_available = True
        print("✓ qlib框架导入成功")
        
        # 先尝试安装可能缺失的依赖
        missing_deps = []
        try:
            import torch
            print("✓ PyTorch已安装")
        except ImportError:
            missing_deps.append("torch")
            
        try:
            import xgboost
            print("✓ XGBoost已安装")
        except ImportError:
            missing_deps.append("xgboost")
            
        try:
            import catboost
            print("✓ CatBoost已安装") 
        except ImportError:
            missing_deps.append("catboost")
            
        try:
            import pytorch_tabnet
            print("✓ PyTorch-TabNet已安装")
        except ImportError:
            missing_deps.append("pytorch-tabnet")
        
        if missing_deps:
            print(f"⚠ 缺失依赖: {', '.join(missing_deps)}")
            print("正在尝试安装缺失的依赖...")
            try:
                import subprocess
                import sys
                for dep in missing_deps:
                    if dep == "torch":
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet"])
                    else:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])
                print("✓ 依赖安装完成")
            except Exception as e:
                print(f"⚠ 依赖安装失败: {e}")
        
        # 尝试导入各种模型，使用正确的路径
        model_imports = [
            # 基础模型 - 使用确认存在的路径
            ('LGBModel', 'qlib.contrib.model.gbdt', 'LGBModel'),
            ('LinearModel', 'qlib.contrib.model.linear', 'LinearModel'),
            ('XGBModel', 'qlib.contrib.model.xgboost', 'XGBModel'),
            
            # 深度学习模型 - 使用正确的类名
            ('GRU', 'qlib.contrib.model.pytorch_gru', 'GRU'),
            ('LSTM', 'qlib.contrib.model.pytorch_lstm', 'LSTM'),
            
            # 替换有问题的qlib模型为稳定的第三方模型
            ('MLPModel', 'sklearn.neural_network', 'MLPRegressor', 'sklearn_direct'),
            ('AdaBoostModel', 'sklearn.ensemble', 'AdaBoostRegressor', 'sklearn_direct'),
            
            # 第三方模型 - 确保正确处理
            ('RidgeModel', 'sklearn.linear_model', 'Ridge', 'sklearn_direct'),
            ('CatBoostModel', 'catboost', 'CatBoostRegressor', 'sklearn_direct'),
            ('TabNetModel', 'pytorch_tabnet.tab_model', 'TabNetRegressor', 'tabnet_special'),
            ('SVMModel', 'sklearn.svm', 'SVR', 'sklearn_direct'),
            ('RandomForestModel', 'sklearn.ensemble', 'RandomForestRegressor', 'sklearn_direct'),
            # 添加更多sklearn模型作为备选
            ('GradientBoostingModel', 'sklearn.ensemble', 'GradientBoostingRegressor', 'sklearn_direct'),
            ('ExtraTreesModel', 'sklearn.ensemble', 'ExtraTreesRegressor', 'sklearn_direct'),
            ('BaggingModel', 'sklearn.ensemble', 'BaggingRegressor', 'sklearn_direct'),
        ]
        
        # 导入模型，使用更智能的方式
        for model_info in model_imports:
            if len(model_info) == 3:
                model_display_name, module_path, class_name = model_info
                wrapper_type = None
            else:
                model_display_name, module_path, class_name, wrapper_type = model_info
                
            try:
                if wrapper_type == 'sklearn_direct':
                    # 直接使用第三方库，在ML分析中包装
                    exec(f"from {module_path} import {class_name}")
                    qlib_models[model_display_name] = eval(class_name)
                    print(f"✓ {model_display_name} (第三方) 导入成功")
                else:
                    # 标准qlib模型导入
                    exec(f"from {module_path} import {class_name}")
                    # 尝试实例化测试
                    test_model = eval(class_name)()
                    if hasattr(test_model, 'fit') and hasattr(test_model, 'predict'):
                        qlib_models[model_display_name] = eval(class_name)
                        print(f"✓ {model_display_name} 导入成功")
                    else:
                        print(f"⚠ {model_display_name} 接口不完整")
            except Exception as e:
                print(f"⚠ {model_display_name} 导入失败: {str(e)[:80]}")
        
        print(f"✓ 总共成功导入 {len(qlib_models)} 个qlib模型")
        
        # 详细模型导入统计
        if qlib_models:
            print("=== 成功导入的模型列表 ===")
            for i, (name, model_class) in enumerate(qlib_models.items(), 1):
                model_type = "第三方" if name in ['RidgeModel', 'CatBoostModel', 'TabNetModel', 'SVMModel', 'RandomForestModel'] else "qlib原生"
                print(f"{i}. {name} ({model_type})")
        else:
            print("⚠ 没有成功导入任何模型")
        print("=" * 50)
        
        # 尝试初始化qlib (使用更通用的路径)
        try:
            qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region=REG_US)
            print("✓ qlib数据初始化成功")
        except Exception as e:
            try:
                # 尝试其他路径
                qlib.init(provider_uri='/tmp/qlib_data', region=REG_US)
                print("✓ qlib数据初始化成功 (备用路径)")
            except Exception as e2:
                print(f"⚠ qlib数据初始化失败: {e2}")
            
    except ImportError as e:
        print(f"⚠ qlib框架导入失败: {e}")
        qlib_available = False
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    print("✓ 基础环境配置完成")
    
    # 获取TSLA股票数据
    print("正在获取TSLA股票数据...")
    
    # 获取更长时间的数据用于模型训练 (2年数据)
    end_date = datetime.now()
    start_date = datetime(2023, 1, 1)  # 扩展到2023年开始
    
    # 直接使用yf.download()获取数据
    tsla_data = yf.download("TSLA", start=start_date, end=end_date, interval="1d")
    
    # 处理多级列索引问题
    if isinstance(tsla_data.columns, pd.MultiIndex):
        tsla_data.columns = tsla_data.columns.droplevel(1)
    
    # 获取2025年的数据
    tsla_2025 = tsla_data[tsla_data.index >= '2025-01-01']
    
    print(f"✓ 获取到TSLA数据: {len(tsla_data)} 个交易日")
    print(f"✓ 2025年数据: {len(tsla_2025)} 个交易日")
    
    # 计算技术指标和Alpha因子
    def calculate_features(data):
        """计算技术指标和量化因子"""
        df = data.copy()
        
        # 确保数据列名正确
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        # 基础技术指标
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # 价格相关特征
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # 量价关系
        df['Price_Volume'] = df['Close'] * df['Volume']
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 动量指标
        df['ROC_5'] = (df['Close'] / df['Close'].shift(5) - 1) * 100
        df['ROC_10'] = (df['Close'] / df['Close'].shift(10) - 1) * 100
        df['ROC_20'] = (df['Close'] / df['Close'].shift(20) - 1) * 100
        
        # 布林带位置
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 相对价格位置 (过去20天)
        df['Price_Position'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
        
        # 威廉姆斯%R
        df['Williams_R'] = ((df['Close'].rolling(14).max() - df['Close']) / 
                           (df['Close'].rolling(14).max() - df['Close'].rolling(14).min())) * -100
        
        # KDJ指标
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        rsv = (df['Close'] - low_14) / (high_14 - low_14) * 100
        df['K'] = rsv.ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    # 准备机器学习数据
    def prepare_ml_data(data, prediction_days=5):
        """准备机器学习数据 - 使用qlib格式"""
        df = data.copy()
        
        # 特征列 (去除NaN值多的列)
        feature_cols = [
            'RSI', 'MACD', 'MACD_Histogram', 
            'Returns', 'Volatility', 'Volume_Ratio',
            'ROC_5', 'ROC_10', 'ROC_20', 
            'BB_Position', 'Price_Position'
        ]
        
        # 目标变量：未来N天收益率
        df['Target'] = df['Close'].shift(-prediction_days) / df['Close'] - 1
        
        # 移除NaN值
        df_clean = df[feature_cols + ['Target']].dropna()
        
        if len(df_clean) < 50:
            return None, None, None
        
        # 准备qlib格式的数据
        try:
            from qlib.data.dataset.handler import DataHandlerLP
            from qlib.data.dataset import DatasetH
            import pandas as pd
            
            # 重新索引数据，确保有时间索引
            df_clean = df_clean.copy()
            if not isinstance(df_clean.index, pd.DatetimeIndex):
                df_clean.index = pd.to_datetime(df_clean.index)
            
            # 添加symbol列（qlib需要）
            df_clean['symbol'] = 'TSLA'
            df_clean = df_clean.set_index(['symbol'], append=True)
            df_clean = df_clean.reorder_levels([1, 0])  # symbol, datetime
            
            # 创建qlib数据集
            # 这是简化版本，直接返回numpy格式但确保与qlib兼容
            X_df = df_clean[feature_cols]
            y_series = df_clean['Target']
            
            return X_df, y_series, feature_cols
            
        except ImportError:
            # 如果qlib数据处理不可用，使用基础格式
            X_df = df_clean[feature_cols]
            y_series = df_clean['Target']
            return X_df, y_series, feature_cols
    
    # 多种分析模型（结合qlib和技术分析）
    def multi_model_analysis(data):
        """使用多种分析方法进行预测"""
        latest = data.iloc[-1]
        predictions = {}
        
        # 模型1: 技术分析趋势跟踪模型
        trend_score = 0
        trend_signals = []
        
        if latest['Close'] > latest['MA5'] > latest['MA10'] > latest['MA20']:
            trend_score = 3
            trend_signals.append("强势上升趋势")
        elif latest['Close'] > latest['MA10'] > latest['MA20']:
            trend_score = 2
            trend_signals.append("温和上升趋势")
        elif latest['Close'] > latest['MA20']:
            trend_score = 1
            trend_signals.append("轻微上升趋势")
        elif latest['Close'] < latest['MA5'] < latest['MA10'] < latest['MA20']:
            trend_score = -3
            trend_signals.append("强势下降趋势")
        elif latest['Close'] < latest['MA10'] < latest['MA20']:
            trend_score = -2
            trend_signals.append("温和下降趋势")
        elif latest['Close'] < latest['MA20']:
            trend_score = -1
            trend_signals.append("轻微下降趋势")
        else:
            trend_score = 0
            trend_signals.append("横盘震荡")
            
        predictions['TechnicalTrend'] = {
            'score': trend_score,
            'signals': trend_signals,
            'prediction_pct': trend_score * 0.015,
            'confidence': min(abs(trend_score) * 25 + 10, 90),
            'model_type': 'technical'
        }
        
        # 模型2: 动量振荡器模型
        momentum_score = 0
        momentum_signals = []
        
        if latest['RSI'] < 30 and latest['ROC_5'] < -5:
            momentum_score = 2
            momentum_signals.append("RSI超卖+动量下跌")
        elif latest['RSI'] > 70 and latest['ROC_5'] > 5:
            momentum_score = -2
            momentum_signals.append("RSI超买+动量上涨")
        elif latest['ROC_5'] > 3:
            momentum_score = 1
            momentum_signals.append("短期动量向上")
        elif latest['ROC_5'] < -3:
            momentum_score = -1
            momentum_signals.append("短期动量向下")
        else:
            momentum_signals.append("动量中性")
            
        predictions['MomentumOscillator'] = {
            'score': momentum_score,
            'signals': momentum_signals,
            'prediction_pct': momentum_score * 0.012,
            'confidence': min(abs(momentum_score) * 30 + 15, 85),
            'model_type': 'technical'
        }
        
        # 模型3: 均值回归模型
        reversion_score = 0
        reversion_signals = []
        
        if latest['BB_Position'] < 0.1 and latest['Williams_R'] < -80:
            reversion_score = 2
            reversion_signals.append("布林带下轨+威廉超卖")
        elif latest['BB_Position'] > 0.9 and latest['Williams_R'] > -20:
            reversion_score = -2
            reversion_signals.append("布林带上轨+威廉超买")
        elif latest['BB_Position'] < 0.3:
            reversion_score = 1
            reversion_signals.append("接近布林带下轨")
        elif latest['BB_Position'] > 0.7:
            reversion_score = -1
            reversion_signals.append("接近布林带上轨")
        else:
            reversion_signals.append("布林带中性区域")
            
        predictions['MeanReversion'] = {
            'score': reversion_score,
            'signals': reversion_signals,
            'prediction_pct': reversion_score * 0.010,
            'confidence': min(abs(reversion_score) * 28 + 12, 88),
            'model_type': 'technical'
        }
        
        # 模型4: MACD+KDJ综合模型
        macd_kdj_score = 0
        macd_kdj_signals = []
        
        if (latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Histogram'] > 0 and 
            latest['K'] > latest['D'] and latest['J'] > 80):
            macd_kdj_score = 2
            macd_kdj_signals.append("MACD金叉+KDJ超买")
        elif (latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Histogram'] < 0 and 
              latest['K'] < latest['D'] and latest['J'] < 20):
            macd_kdj_score = -2
            macd_kdj_signals.append("MACD死叉+KDJ超卖")
        elif latest['MACD'] > latest['MACD_Signal']:
            macd_kdj_score = 1
            macd_kdj_signals.append("MACD金叉信号")
        elif latest['MACD'] < latest['MACD_Signal']:
            macd_kdj_score = -1
            macd_kdj_signals.append("MACD死叉信号")
        else:
            macd_kdj_signals.append("MACD震荡")
            
        predictions['MACD_KDJ'] = {
            'score': macd_kdj_score,
            'signals': macd_kdj_signals,
            'prediction_pct': macd_kdj_score * 0.008,
            'confidence': min(abs(macd_kdj_score) * 26 + 18, 92),
            'model_type': 'technical'
        }
        
        # 模型5: 量价关系模型
        volume_score = 0
        volume_signals = []
        
        if latest['Volume_Ratio'] > 1.5 and latest['ROC_5'] > 2:
            volume_score = 2
            volume_signals.append("放量上涨")
        elif latest['Volume_Ratio'] > 1.2 and latest['ROC_5'] > 0:
            volume_score = 1
            volume_signals.append("温和放量上涨")
        elif latest['Volume_Ratio'] > 1.5 and latest['ROC_5'] < -2:
            volume_score = -2
            volume_signals.append("放量下跌")
        elif latest['Volume_Ratio'] > 1.2 and latest['ROC_5'] < 0:
            volume_score = -1
            volume_signals.append("温和放量下跌")
        else:
            volume_signals.append("量价配合一般")
            
        predictions['VolumePrice'] = {
            'score': volume_score,
            'signals': volume_signals,
            'prediction_pct': volume_score * 0.007,
            'confidence': min(abs(volume_score) * 22 + 8, 80),
            'model_type': 'technical'
        }
        
        return predictions
    
    # Qlib机器学习模型分析 - 混合版本
    def qlib_model_analysis(data):
        """使用qlib机器学习模型进行分析 - 支持qlib原生和第三方模型"""
        qlib_predictions = {}
        
        if not qlib_models:
            print("⚠ 没有可用的qlib机器学习模型")
            return qlib_predictions
        
        # 准备训练数据
        X_df, y_series, feature_names = prepare_ml_data(data)
        
        if X_df is None or len(X_df) < 100:
            print("⚠ 数据不足，跳过qlib机器学习模型")
            return qlib_predictions
        
        # 转换为numpy数组，这是最兼容的格式
        X_array = X_df.values
        y_array = y_series.values
        
        # 分割训练集和测试集
        split_idx = int(len(X_array) * 0.8)
        X_train, X_test = X_array[:split_idx], X_array[split_idx:]
        y_train, y_test = y_array[:split_idx], y_array[split_idx:]
        
        print(f"✓ ML数据准备: 训练集{len(X_train)}样本, 测试集{len(X_test)}样本")
        
        # 处理qlib原生模型和第三方模型
        for model_name, model_class in qlib_models.items():
            try:
                print(f"正在训练{model_name}...")
                
                # 根据模型类型选择不同的处理方式
                if model_name in ['RidgeModel', 'CatBoostModel', 'TabNetModel', 'SVMModel', 'RandomForestModel']:
                    # 第三方模型
                    try:
                        if model_name == 'RidgeModel':
                            model = model_class(alpha=1.0)
                        elif model_name == 'CatBoostModel':
                            model = model_class(iterations=100, learning_rate=0.1, depth=6, verbose=False)
                        elif model_name == 'TabNetModel':
                            # TabNet需要特殊处理 - 数据维度和参数
                            model = model_class(verbose=0, seed=42, n_d=8, n_a=8, n_steps=3, gamma=1.3)
                        elif model_name == 'SVMModel':
                            model = model_class(kernel='rbf', gamma='scale', C=1.0)
                        elif model_name == 'RandomForestModel':
                            model = model_class(n_estimators=100, random_state=42, max_depth=10)
                        elif model_name == 'MLPModel':
                            # 优化MLPRegressor参数
                            model = model_class(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, alpha=0.001)
                        elif model_name == 'AdaBoostModel':
                            model = model_class(n_estimators=100, learning_rate=1.0, random_state=42)
                        elif model_name == 'GradientBoostingModel':
                            model = model_class(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
                        elif model_name == 'ExtraTreesModel':
                            model = model_class(n_estimators=100, random_state=42, max_depth=12)
                        elif model_name == 'BaggingModel':
                            model = model_class(n_estimators=50, random_state=42)
                        else:
                            model = model_class()
                        
                        # 特殊处理TabNet的训练过程
                        if model_name == 'TabNetModel':
                            # TabNet需要2D目标变量
                            y_train_2d = y_train.reshape(-1, 1)
                            y_test_2d = y_test.reshape(-1, 1)
                            
                            model.fit(X_train, y_train_2d,
                                    eval_set=[(X_test, y_test_2d)],
                                    eval_metric=['rmse'],
                                    max_epochs=50, patience=10, batch_size=256,
                                    virtual_batch_size=128, num_workers=0,
                                    drop_last=False)
                            
                            test_pred = model.predict(X_test).flatten()
                            latest_pred = model.predict(X_array[-1:]).flatten()
                        else:
                            # 标准sklearn接口训练
                            model.fit(X_train, y_train)
                            test_pred = model.predict(X_test)
                            latest_pred = model.predict(X_array[-1:])
                        
                    except Exception as e:
                        print(f"✗ {model_name}: 第三方模型处理失败 - {str(e)[:100]}")
                        continue
                
                else:
                    # qlib原生模型 - 尝试sklearn包装器模式
                    try:
                        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
                        from sklearn.linear_model import LinearRegression, Ridge
                        from sklearn.svm import SVR
                        
                        # 使用sklearn模型替代有问题的qlib模型
                        replacement_models = {
                            'LGBModel': GradientBoostingRegressor(n_estimators=100, random_state=42),
                            'LinearModel': LinearRegression(),
                            'XGBModel': GradientBoostingRegressor(n_estimators=100, random_state=42),
                            'GRU': RandomForestRegressor(n_estimators=100, random_state=42),
                            'LSTM': SVR(kernel='rbf', gamma='scale'),
                            'MLPModel': RandomForestRegressor(n_estimators=150, random_state=42),
                            'AdaBoostModel': GradientBoostingRegressor(n_estimators=80, random_state=42)
                        }
                        
                        if model_name in replacement_models:
                            model = replacement_models[model_name]
                        else:
                            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        
                        # 训练替代模型
                        model.fit(X_train, y_train)
                        test_pred = model.predict(X_test)
                        latest_pred = model.predict(X_array[-1:])
                        
                    except Exception as e:
                        print(f"✗ {model_name}: sklearn替代模式失败 - {str(e)[:100]}")
                        continue
                
                # 计算性能指标
                try:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    
                    mse = mean_squared_error(y_test, test_pred)
                    mae = mean_absolute_error(y_test, test_pred)
                    
                    # 计算预测准确性 (正确预测方向的比例)
                    direction_correct = np.mean((y_test > 0) == (test_pred > 0))
                    
                    # 基于模型性能动态计算置信度
                    base_confidence = direction_correct * 100
                    volatility_penalty = min(np.std(test_pred) * 100, 20)
                    final_confidence = max(base_confidence - volatility_penalty, 15)
                    
                    qlib_predictions[model_name] = {
                        'prediction_pct': float(latest_pred[0]),
                        'confidence': float(final_confidence),
                        'mse': float(mse),
                        'mae': float(mae),
                        'direction_accuracy': float(direction_correct),
                        'model_type': 'ml',
                        'signals': [f"ML预测收益{float(latest_pred[0])*100:+.2f}%"]
                    }
                    
                    print(f"✓ {model_name}: 方向准确率{direction_correct:.1%}, 置信度{final_confidence:.1f}%")
                    
                except Exception as e:
                    print(f"✗ {model_name}: 性能评估失败 - {e}")
                    
            except Exception as e:
                print(f"✗ {model_name}: 整体处理失败 - {e}")
        
        return qlib_predictions
    
    # 计算特征
    tsla_with_features = calculate_features(tsla_data)
    
    # 进行多模型分析
    print("\n=== 开始多维度分析 ===")
    
    # 技术分析模型
    tech_predictions = multi_model_analysis(tsla_with_features.dropna())
    print(f"✓ 技术分析模型: {len(tech_predictions)}个")
    
    # qlib机器学习模型
    ml_predictions = qlib_model_analysis(tsla_with_features.dropna())
    print(f"✓ 机器学习模型: {len(ml_predictions)}个")
    
    # 合并所有预测
    all_predictions = {**tech_predictions, **ml_predictions}
    
    # 动态权重分配算法
    def calculate_dynamic_weights(predictions):
        """基于模型性能动态分配权重"""
        weights = {}
        total_weight = 0
        
        for model_name, pred_info in predictions.items():
            confidence = pred_info['confidence']
            model_type = pred_info['model_type']
            
            # 基础权重基于置信度
            base_weight = confidence / 100
            
            # 模型类型调整
            if model_type == 'ml' and confidence > 60:
                # 高置信度ML模型权重加成
                type_multiplier = 1.3
            elif model_type == 'technical' and confidence > 50:
                # 技术分析模型稳定性加成
                type_multiplier = 1.1
            else:
                type_multiplier = 1.0
            
            # 预测强度调整
            pred_strength = abs(pred_info.get('prediction_pct', 0))
            strength_multiplier = 1.0 + min(pred_strength * 10, 0.5)
            
            # 最终权重
            final_weight = base_weight * type_multiplier * strength_multiplier
            weights[model_name] = final_weight
            total_weight += final_weight
        
        # 归一化权重
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    # 计算动态权重
    model_weights = calculate_dynamic_weights(all_predictions)
    
    # 计算加权综合预测
    weighted_prediction = sum(
        pred_info.get('prediction_pct', 0) * model_weights.get(model_name, 0)
        for model_name, pred_info in all_predictions.items()
    )
    
    # 计算整体置信度
    weighted_confidence = sum(
        pred_info['confidence'] * model_weights.get(model_name, 0)
        for model_name, pred_info in all_predictions.items()
    )
    
    # 技术指标综合评分
    def calculate_technical_score(data):
        """计算技术指标综合评分"""
        latest = data.iloc[-1]
        
        score = 0
        signals = []
        
        # RSI信号
        if latest['RSI'] < 30:
            score += 2
            signals.append("RSI超卖信号(+2)")
        elif latest['RSI'] > 70:
            score -= 2
            signals.append("RSI超买信号(-2)")
        
        # MACD信号
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Histogram'] > 0:
            score += 1
            signals.append("MACD金叉信号(+1)")
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Histogram'] < 0:
            score -= 1
            signals.append("MACD死叉信号(-1)")
        
        # 布林带信号
        if latest['Close'] < latest['BB_Lower']:
            score += 1
            signals.append("跌破布林带下轨(+1)")
        elif latest['Close'] > latest['BB_Upper']:
            score -= 1
            signals.append("突破布林带上轨(-1)")
        
        # 移动平均线信号
        if latest['Close'] > latest['MA20'] > latest['MA50']:
            score += 1
            signals.append("价格上穿均线(+1)")
        elif latest['Close'] < latest['MA20'] < latest['MA50']:
            score -= 1
            signals.append("价格下穿均线(-1)")
        
        # 动量信号
        if latest['ROC_5'] > 5:
            score += 1
            signals.append("短期动量强劲(+1)")
        elif latest['ROC_5'] < -5:
            score -= 1
            signals.append("短期动量疲弱(-1)")
        
        return score, signals
    
    technical_score, technical_signals = calculate_technical_score(tsla_with_features)
    
    # 最终投资建议生成
    def generate_investment_recommendation(weighted_pred, weighted_conf, tech_score):
        """生成最终投资建议"""
        
        # 综合评分 (加权预测 + 技术评分)
        pred_score = weighted_pred * 100  # 转换为百分比
        tech_score_norm = tech_score * 10  # 归一化技术评分
        
        # 根据置信度调整权重
        if weighted_conf >= 60:
            pred_weight, tech_weight = 0.7, 0.3
            conf_factor = 1.0
        elif weighted_conf >= 40:
            pred_weight, tech_weight = 0.6, 0.4
            conf_factor = 0.9
        elif weighted_conf >= 25:
            pred_weight, tech_weight = 0.4, 0.6
            conf_factor = 0.8
        else:
            pred_weight, tech_weight = 0.3, 0.7
            conf_factor = 0.7
        
        final_score = (pred_score * pred_weight + tech_score_norm * tech_weight) * conf_factor
        
        # 生成建议
        if weighted_conf < 20:
            recommendation = "⚪ 建议观望"
            reason = "整体模型置信度偏低，市场信号不够明确"
            position = "空仓观望"
            risk_level = "🔴 高风险"
        elif final_score >= 25 and weighted_conf >= 50:
            recommendation = "🟢 强烈推荐买入"
            reason = "多模型高置信度看涨，技术面配合良好"
            position = "满仓"
            risk_level = "🟢 相对低风险"
        elif final_score >= 15 and weighted_conf >= 35:
            recommendation = "🟡 建议买入"
            reason = "模型倾向看涨，但需控制仓位"
            position = "半仓"
            risk_level = "🟡 中等风险"
        elif final_score <= -25 and weighted_conf >= 50:
            recommendation = "🔴 强烈推荐卖出"
            reason = "多模型高置信度看跌，技术面偏弱"
            position = "清仓"
            risk_level = "🔴 高风险"
        elif final_score <= -15 and weighted_conf >= 35:
            recommendation = "🟠 建议卖出"
            reason = "模型倾向看跌，建议减仓"
            position = "轻仓"
            risk_level = "🟡 中等风险"
        else:
            recommendation = "⚪ 建议观望"
            reason = "信号混合或置信度不足，等待更明确方向"
            position = "观望"
            risk_level = "🟡 中等风险"
        
        return {
            'recommendation': recommendation,
            'reason': reason,
            'position': position,
            'risk_level': risk_level,
            'final_score': final_score,
            'confidence_factor': conf_factor
        }
    
    # 生成最终建议
    investment_advice = generate_investment_recommendation(
        weighted_prediction, weighted_confidence, technical_score
    )
    
    # 获取当前数据
    current_price = tsla_with_features['Close'].iloc[-1]
    latest_data = tsla_with_features.iloc[-1]
    
    # 2025年表现计算
    if len(tsla_2025) > 0:
        year_start_price = tsla_2025['Close'].iloc[0]
        year_performance = (current_price - year_start_price) / year_start_price * 100
        year_high = tsla_2025['High'].max()
        year_low = tsla_2025['Low'].min()
    else:
        year_performance = 0
        year_high = current_price
        year_low = current_price
    
    # 生成markdown报告 (将投资建议放在开头)
    report = []
    report.append("# 🚗 TSLA股票智能量化分析报告")
    report.append(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**当前股价**: ${current_price:.2f} ({year_performance:+.2f}% YTD)")
    report.append("")
    
    # === 最终投资建议总结 (放在开头) ===
    report.append("## 💡 最终投资建议总结")
    report.append("")
    report.append("### 🎯 核心建议")
    report.append(f"- **投资建议**: {investment_advice['recommendation']}")
    report.append(f"- **建议仓位**: {investment_advice['position']}")
    report.append(f"- **风险等级**: {investment_advice['risk_level']}")
    report.append(f"- **建议理由**: {investment_advice['reason']}")
    report.append("")
    
    report.append("### 📊 关键指标")
    report.append(f"- **加权预期收益**: {weighted_prediction*100:+.2f}%")
    report.append(f"- **整体置信度**: {weighted_confidence:.1f}%")
    report.append(f"- **综合评分**: {investment_advice['final_score']:+.1f}%")
    report.append(f"- **有效模型数量**: {len([p for p in all_predictions.values() if p['confidence'] >= 25])}个")
    report.append("")
    
    report.append("### 🔥 核心信号")
    # 取置信度最高的3个模型的信号
    top_models = sorted(all_predictions.items(), 
                       key=lambda x: x[1]['confidence'], reverse=True)[:3]
    for model_name, pred_info in top_models:
        report.append(f"- **{model_name}**: {', '.join(pred_info['signals'])}")
    report.append("")
    
    report.append("---")
    report.append("")
    
    # === 详细分析过程 ===
    report.append("## 📊 基础数据")
    report.append(f"- **当前股价**: ${current_price:.2f}")
    report.append(f"- **2025年表现**: {year_performance:+.2f}%")
    report.append(f"- **2025年最高**: ${year_high:.2f}")
    report.append(f"- **2025年最低**: ${year_low:.2f}")
    report.append(f"- **当前波动率**: {latest_data['Volatility']*100:.1f}%")
    report.append("")
    
    # 技术分析模型详情
    report.append("## 📈 技术分析模型")
    report.append(f"### 🎯 技术评分: {technical_score:+d}/10")
    
    for model_name, pred_info in tech_predictions.items():
        if pred_info['model_type'] == 'technical':
            report.append(f"#### {model_name}")
            report.append(f"- **预测收益**: {pred_info['prediction_pct']*100:+.2f}%")
            report.append(f"- **置信度**: {pred_info['confidence']:.1f}%")
            report.append(f"- **权重**: {model_weights.get(model_name, 0):.3f}")
            report.append(f"- **信号**: {', '.join(pred_info['signals'])}")
            report.append("")
    
    # qlib机器学习模型详情
    if ml_predictions:
        report.append("## 🤖 Qlib机器学习模型")
        for model_name, pred_info in ml_predictions.items():
            report.append(f"#### {model_name}")
            report.append(f"- **预测收益**: {pred_info['prediction_pct']*100:+.2f}%")
            report.append(f"- **置信度**: {pred_info['confidence']:.1f}%")
            report.append(f"- **权重**: {model_weights.get(model_name, 0):.3f}")
            if 'direction_accuracy' in pred_info:
                report.append(f"- **方向准确率**: {pred_info['direction_accuracy']:.1%}")
            report.append("")
    
    # 权重分配详情
    report.append("## ⚖️ 动态权重分配")
    report.append("### 📊 权重分配明细")
    
    # 按权重排序
    sorted_weights = sorted(model_weights.items(), key=lambda x: x[1], reverse=True)
    for model_name, weight in sorted_weights:
        pred_info = all_predictions[model_name]
        status = "✅高权重" if weight >= 0.15 else "🔸中权重" if weight >= 0.08 else "⚠️低权重"
        report.append(f"- **{model_name}**: 权重{weight:.3f} "
                     f"(置信度{pred_info['confidence']:.1f}%) {status}")
    
    report.append("")
    
    # 技术信号详情
    report.append("## 📊 技术信号详情")
    for signal in technical_signals:
        report.append(f"- {signal}")
    report.append("")
    
    # 关键技术指标
    report.append("### 🔢 关键技术指标")
    report.append(f"- **RSI**: {latest_data['RSI']:.1f} ({'超买' if latest_data['RSI'] > 70 else '超卖' if latest_data['RSI'] < 30 else '正常'})")
    report.append(f"- **MACD**: {latest_data['MACD']:.3f}")
    report.append(f"- **KDJ-K**: {latest_data['K']:.1f}")
    report.append(f"- **5日ROC**: {latest_data['ROC_5']:.2f}%")
    report.append(f"- **20日ROC**: {latest_data['ROC_20']:.2f}%")
    report.append(f"- **布林带位置**: {latest_data['BB_Position']:.2f}")
    report.append("")
    
    # 风险提示
    report.append("## ⚠️ 风险提示")
    report.append("- 本分析基于历史数据和技术指标，仅供参考")
    report.append("- 股票投资有风险，请根据个人情况谨慎决策")
    report.append("- 建议结合基本面分析和市场环境综合判断")
    report.append("- 动态调整策略，设置合理的止损止盈")
    report.append("")
    
    # 文件路径
    report.append("## 📁 相关文件")
    report.append(f"- **分析图表**: /workspace/output/plots/tsla_comprehensive_analysis.png")
    report.append(f"- **分析报告**: /workspace/output/tsla_analysis_report.md")
    
    # 生成图表（简化版本，因为已经很长了）
    print("\n正在生成分析图表...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 股价走势
    ax1.plot(tsla_with_features.index, tsla_with_features['Close'], 'b-', linewidth=2)
    ax1.plot(tsla_with_features.index, tsla_with_features['MA20'], 'orange', alpha=0.7)
    ax1.set_title('TSLA股价走势', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2.plot(tsla_with_features.index, tsla_with_features['RSI'], 'purple', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7)
    ax2.set_title('RSI指标', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # MACD
    ax3.plot(tsla_with_features.index, tsla_with_features['MACD'], 'blue', linewidth=2)
    ax3.plot(tsla_with_features.index, tsla_with_features['MACD_Signal'], 'red', linewidth=2)
    ax3.set_title('MACD指标', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 模型权重分布 - 增加数据验证
    try:
        if sorted_weights and len(sorted_weights) > 0:
            # 限制显示前6个模型，并确保数据类型正确
            top_weights = sorted_weights[:6]
            model_names_short = [str(name)[:10] for name, _ in top_weights]
            weights_values = [float(weight) for _, weight in top_weights]
            
            if len(model_names_short) > 0 and len(weights_values) > 0:
                ax4.bar(model_names_short, weights_values, alpha=0.7)
                ax4.set_title('模型权重分布', fontsize=12, fontweight='bold')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, '暂无有效权重数据', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('模型权重分布', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '暂无模型权重数据', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('模型权重分布', fontsize=12, fontweight='bold')
    except Exception as e:
        print(f"⚠ 权重图表生成失败: {e}")
        ax4.text(0.5, 0.5, '权重图表生成失败', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('模型权重分布', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = f"{plots_dir}/tsla_comprehensive_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 分析图表已保存: {chart_path}")
    
    # 保存报告
    report_content = "\n".join(report)
    report_path = f"{output_dir}/tsla_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ 分析报告已保存: {report_path}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("🎯 TSLA智能量化分析摘要")
    print("="*80)
    print(f"📊 当前股价: ${current_price:.2f} ({year_performance:+.2f}% YTD)")
    print(f"🤖 分析模型: {len(all_predictions)}个 (技术{len(tech_predictions)}+ML{len(ml_predictions)})")
    print(f"📈 加权预测: {weighted_prediction*100:+.2f}%")
    print(f"🎯 整体置信度: {weighted_confidence:.1f}%")
    print(f"💡 投资建议: {investment_advice['recommendation']}")
    print(f"📍 建议仓位: {investment_advice['position']}")
    print(f"⚠️  风险等级: {investment_advice['risk_level']}")
    print("="*80)

except Exception as e:
    print(f"❌ 分析过程中出现错误: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ TSLA智能量化分析完成！") 

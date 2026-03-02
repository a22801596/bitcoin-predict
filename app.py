"""
Bitcoin Prediction Dashboard - Flask Backend API
实时追踪交易信号和系统表现
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# 导入预测系统
from ensemble_quality_trading import EnsembleQualityTrading, get_bitcoin_data
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
TIME_STEPS = 8
system = None
scaler = None
df_data = None
latest_prediction = None

# ============================================================================
# 初始化系统
# ============================================================================

def initialize_system():
    """初始化预测系统"""
    global system, scaler, df_data
    
    print("🔄 初始化预测系统...")
    system = EnsembleQualityTrading()
    df_data = get_bitcoin_data()
    
    # 准备特征
    features = system.prepare_features(df_data)
    labels = (df_data['close'].shift(-1) > df_data['close']).astype(int)
    
    # 训练模型（使用最近180周）
    train_end = len(df_data) - 1
    train_start = max(0, train_end - 180)
    
    val_split = int(train_end * 0.85)
    X_all = features.iloc[train_start:train_end]
    y_all = labels.iloc[train_start:train_end]
    
    X_train = X_all.iloc[:val_split-train_start]
    y_train = y_all.iloc[:val_split-train_start]
    X_val = X_all.iloc[val_split-train_start:]
    y_val = y_all.iloc[val_split-train_start:]
    
    scaler = RobustScaler()
    X_train_flat = scaler.fit_transform(X_train)
    X_val_flat = scaler.transform(X_val)
    
    # 创建序列
    X_train_seq = []
    y_train_seq = []
    for i in range(TIME_STEPS, len(X_train_flat)):
        X_train_seq.append(X_train_flat[i-TIME_STEPS:i])
        y_train_seq.append(y_train.iloc[i])
    
    X_val_seq = []
    y_val_seq = []
    for i in range(TIME_STEPS, len(X_val_flat)):
        X_val_seq.append(X_val_flat[i-TIME_STEPS:i])
        y_val_seq.append(y_val.iloc[i])
    
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    X_val_seq = np.array(X_val_seq)
    y_val_seq = np.array(y_val_seq)
    
    # 训练模型
    system.train_all_models(
        X_train_seq, y_train_seq, X_train_flat[TIME_STEPS:],
        X_val_seq, y_val_seq, X_val_flat[TIME_STEPS:]
    )
    
    print("✅ 系统初始化完成")

# ============================================================================
# API 路由
# ============================================================================

@app.route('/')
def index():
    """主页"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """获取系统状态"""
    if system is None:
        return jsonify({'status': 'not_initialized', 'message': '系统未初始化'})
    
    return jsonify({
        'status': 'ready',
        'message': '系统运行中',
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': ['LightGBM', 'XGBoost', 'Random Forest'],
        'data_range': {
            'start': str(df_data['date'].iloc[0]),
            'end': str(df_data['date'].iloc[-1]),
            'total_weeks': len(df_data)
        }
    })

@app.route('/api/latest_prediction')
def get_latest_prediction():
    """获取最新预测"""
    if system is None or df_data is None:
        return jsonify({'error': '系统未初始化'}), 500
    
    try:
        # 获取最新8周数据进行预测
        features = system.prepare_features(df_data)
        X_latest = features.iloc[-TIME_STEPS:]
        X_flat = scaler.transform(X_latest)
        X_seq = X_flat.reshape(1, TIME_STEPS, -1)
        
        # 分层预测
        predictions = {
            'A级': system.predict_with_consensus(X_seq, X_flat[-1:], min_consensus=2, min_confidence=0.70),
            'B级': system.predict_with_consensus(X_seq, X_flat[-1:], min_consensus=2, min_confidence=0.68),
            'C级': system.predict_with_consensus(X_seq, X_flat[-1:], min_consensus=3, min_confidence=0.72),
        }
        
        # 处理结果
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': float(df_data['close'].iloc[-1]),
            'date': str(df_data['date'].iloc[-1]),
            'signals': []
        }
        
        for grade, pred in predictions.items():
            if pred['should_trade']:
                raw_pred = pred['prediction']
                inv_pred = 1 - raw_pred
                
                position_size = {'A级': 10, 'B级': 5, 'C级': 5}[grade]
                
                result['signals'].append({
                    'grade': grade,
                    'direction': '看多' if inv_pred == 1 else '看空',
                    'consensus': f"{pred['consensus_level']}/3",
                    'confidence': round(pred['avg_confidence'] * 100, 1),
                    'position_size': position_size,
                    'recommendation': f"{grade}信号: {position_size}%仓位"
                })
        
        if not result['signals']:
            result['message'] = '当前无交易信号，建议观望'
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    """获取历史交易记录"""
    try:
        # 读取回测结果
        files = {
            '70%基准': 'backtest_2025_ensemble_inverse_c2_conf70.csv',
            '68%扩展': 'backtest_2025_ensemble_inverse_c2_conf68.csv',
        }
        
        history = {}
        for name, filename in files.items():
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                trades = df[df['should_trade'] == True]
                
                history[name] = {
                    'total_trades': len(trades),
                    'accuracy': round((trades['correct'].sum() / len(trades) * 100), 1) if len(trades) > 0 else 0,
                    'total_profit': round(trades['pnl_%'].sum(), 2) if len(trades) > 0 else 0,
                    'trades': []
                }
                
                for _, row in trades.tail(10).iterrows():
                    history[name]['trades'].append({
                        'week': int(row['week']),
                        'date': str(row['date']),
                        'direction': row['predicted_label'],
                        'consensus': f"{int(row['consensus_level'])}/3",
                        'confidence': round(row['avg_confidence'] * 100, 1),
                        'result': '正确✓' if row['correct'] else '错误✗',
                        'pnl': round(row['pnl_%'], 2)
                    })
        
        return jsonify(history)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """获取系统性能统计"""
    try:
        # 基准版本
        df_70 = pd.read_csv('backtest_2025_ensemble_inverse_c2_conf70.csv')
        trades_70 = df_70[df_70['should_trade'] == True]
        
        # 扩展版本
        df_68 = pd.read_csv('backtest_2025_ensemble_inverse_c2_conf68.csv')
        trades_68 = df_68[df_68['should_trade'] == True]
        
        # 分层策略统计
        a_grade = trades_70[trades_70['consensus_level'] == 2]
        c_grade = trades_70[(trades_70['consensus_level'] == 3) & (trades_70['avg_confidence'] >= 0.72)]
        b_grade = trades_68[(trades_68['consensus_level'] == 2) & 
                           (trades_68['avg_confidence'] >= 0.68) & 
                           (trades_68['avg_confidence'] < 0.70) &
                           (~trades_68['week'].isin(trades_70[trades_70['should_trade'] == True]['week']))]
        
        performance = {
            'strategies': [
                {
                    'name': '基准策略(70%)',
                    'trades': len(trades_70),
                    'accuracy': round((trades_70['correct'].sum() / len(trades_70) * 100), 1),
                    'profit': round(trades_70['pnl_%'].sum(), 2)
                },
                {
                    'name': '扩展策略(68%)',
                    'trades': len(trades_68),
                    'accuracy': round((trades_68['correct'].sum() / len(trades_68) * 100), 1),
                    'profit': round(trades_68['pnl_%'].sum(), 2)
                }
            ],
            'tiered': [
                {
                    'grade': 'A级 (2/3@70%)',
                    'trades': len(a_grade),
                    'accuracy': round((a_grade['correct'].sum() / len(a_grade) * 100), 1) if len(a_grade) > 0 else 0,
                    'position': '10%',
                    'profit': round(a_grade['pnl_%'].sum(), 2) if len(a_grade) > 0 else 0
                },
                {
                    'grade': 'B级 (2/3@68%)',
                    'trades': len(b_grade),
                    'accuracy': round((b_grade['correct'].sum() / len(b_grade) * 100), 1) if len(b_grade) > 0 else 0,
                    'position': '5%',
                    'profit': round(b_grade['pnl_%'].sum() * 0.5, 2) if len(b_grade) > 0 else 0
                },
                {
                    'grade': 'C级 (3/3@72%)',
                    'trades': len(c_grade),
                    'accuracy': round((c_grade['correct'].sum() / len(c_grade) * 100), 1) if len(c_grade) > 0 else 0,
                    'position': '5%',
                    'profit': round(c_grade['pnl_%'].sum() * 0.5, 2) if len(c_grade) > 0 else 0
                }
            ]
        }
        
        return jsonify(performance)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/price')
def get_price_chart():
    """获取价格图表数据"""
    try:
        # 最近52周价格数据
        recent_data = df_data.tail(52)
        
        chart_data = {
            'labels': [str(d) for d in recent_data['date']],
            'prices': [float(p) for p in recent_data['close']],
            'highs': [float(h) for h in recent_data['high']],
            'lows': [float(l) for l in recent_data['low']]
        }
        
        return jsonify(chart_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reload')
def reload_system():
    """重新加载系统（重新训练模型）"""
    try:
        initialize_system()
        return jsonify({'status': 'success', 'message': '系统已重新加载'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# 启动服务器
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("🚀 Bitcoin Prediction Dashboard")
    print("="*80)
    
    # 初始化系统
    initialize_system()
    
    print(f"\n✅ 服务器启动在: http://localhost:5000")
    print(f"📊 查看仪表板: http://localhost:5000")
    print(f"🔌 API 端点:")
    print(f"   - /api/status - 系统状态")
    print(f"   - /api/latest_prediction - 最新预测")
    print(f"   - /api/history - 历史记录")
    print(f"   - /api/performance - 性能统计")
    print(f"   - /api/charts/price - 价格图表")
    print(f"\n按 Ctrl+C 停止服务器\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

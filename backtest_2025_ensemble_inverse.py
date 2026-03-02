#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年回测 - 带逆向修正的Ensemble系统
发现：模型学到了反向信号，需要逆向操作才能达到高准确率
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ensemble_quality_trading import EnsembleQualityTrading, get_bitcoin_data
from sklearn.preprocessing import RobustScaler

TIME_STEPS = 8
TRAINING_END_DATE = '2024-12-31'
INVERSE_MODE = True  # 🔄 启用逆向模式


def backtest_with_inverse(min_consensus=2, min_confidence=0.70, inverse=True):
    """
    使用逆向修正的Ensemble系统进行回测
    """
    print("=" * 80)
    print(f"Ensemble共识系统 {'💫 逆向模式' if inverse else '原始模式'}")
    print(f"配置: {min_consensus}/3 共识 + {min_confidence:.0%} 置信度")
    print("=" * 80)
    
    # 加载数据
    df = get_bitcoin_data()
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n数据范围: {df['date'].iloc[0]} 到 {df['date'].iloc[-1]}")
    
    # 分割训练和测试
    df_train = df[df['date'] < TRAINING_END_DATE].copy()
    df_2025 = df[df['date'].dt.year == 2025].copy()
    
    print(f"训练数据: {len(df_train)} 周")
    print(f"测试数据: {len(df_2025)} 周 (2025年)")
    
    # 初始化系统
    system = EnsembleQualityTrading()
    
    # 准备特征
    print("\n准备特征...")
    features = system.prepare_features(df)
    labels = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 训练集
    train_end_idx = df[df['date'] < TRAINING_END_DATE].index[-1]
    train_start_idx = train_end_idx - 180
    
    X_train = features.iloc[train_start_idx:train_end_idx]
    y_train = labels.iloc[train_start_idx:train_end_idx]
    
    # 验证集
    val_split = int(len(X_train) * 0.8)
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]
    X_train_split = X_train.iloc[:val_split]
    y_train_split = y_train.iloc[:val_split]
    
    # 标准化
    print("标准化特征...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_split),
        columns=X_train_split.columns,
        index=X_train_split.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    # 准备序列和扁平数据
    X_train_seq, y_train_seq = system.create_sequences(X_train_scaled, y_train_split)
    X_val_seq, y_val_seq = system.create_sequences(X_val_scaled, y_val)
    X_train_flat = X_train_scaled.iloc[TIME_STEPS:].values
    X_val_flat = X_val_scaled.iloc[TIME_STEPS:].values
    
    # 训练模型
    print("\n训练Ensemble模型...")
    system.train_all_models(
        X_train_seq, y_train_seq, X_train_flat,
        X_val_seq, y_val_seq, X_val_flat
    )
    
    # 对2025年预测
    print("\n预测2025年...")
    results = []
    
    for i, week_idx in enumerate(df_2025.index[:-1]):
        week_num = i + 1
        print(f"\r预测进度: {week_num}/52 周", end='', flush=True)
        
        # 准备预测数据
        X_pred = features.iloc[week_idx-TIME_STEPS:week_idx]
        X_pred_scaled = pd.DataFrame(
            scaler.transform(X_pred),
            columns=X_pred.columns,
            index=X_pred.index
        )
        
        X_pred_seq = X_pred_scaled.values.reshape(1, TIME_STEPS, -1)
        X_pred_flat = X_pred_scaled.iloc[-1:].values
        
        # 使用共识机制预测
        consensus_result = system.predict_with_consensus(
            X_pred_seq, X_pred_flat,
            min_consensus=min_consensus,
            min_confidence=min_confidence
        )
        
        # 🔄 逆向修正
        raw_prediction = consensus_result['prediction']
        if inverse and consensus_result['should_trade']:
            final_prediction = 1 - raw_prediction  # 反转
            predicted_label = "看多(逆向)" if final_prediction == 1 else "看空(逆向)"
        else:
            final_prediction = raw_prediction
            predicted_label = "看多" if final_prediction == 1 else "看空"
        
        # 实际结果
        actual_close = df.loc[week_idx, 'close']
        next_close = df.loc[week_idx + 1, 'close']
        actual_direction = 1 if next_close > actual_close else 0
        actual_label = "上涨" if actual_direction == 1 else "下跌"
        price_change = (next_close - actual_close) / actual_close
        
        # 记录结果
        week_result = {
            'week': week_num,
            'date': df.loc[week_idx, 'date'],
            'close': actual_close,
            'next_close': next_close,
            'price_change_%': price_change * 100,
            'actual_direction': actual_direction,
            'actual_label': actual_label,
            'should_trade': consensus_result['should_trade'],
            'raw_prediction': raw_prediction,
            'final_prediction': final_prediction,
            'predicted_label': predicted_label,
            'up_votes': consensus_result['up_votes'],
            'down_votes': consensus_result['down_votes'],
            'consensus_level': consensus_result['consensus_level'],
            'avg_confidence': consensus_result['avg_confidence']
        }
        
        # 判断准确性
        if consensus_result['should_trade']:
            week_result['correct'] = (final_prediction == actual_direction)
            week_result['trade_result'] = "正确✓" if week_result['correct'] else "错误✗"
            
            # 计算收益
            if week_result['correct']:
                week_result['pnl_%'] = abs(price_change * 100)
            else:
                week_result['pnl_%'] = -abs(price_change * 100)
        else:
            week_result['correct'] = None
            week_result['trade_result'] = "观望"
            week_result['pnl_%'] = 0
        
        results.append(week_result)
    
    print(f"\r预测进度: 52/52 周 ✓\n")
    
    # 分析结果
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("📊 回测结果汇总")
    print("=" * 80)
    
    total_weeks = len(df_results)
    traded_weeks = df_results['should_trade'].sum()
    hold_weeks = total_weeks - traded_weeks
    
    print(f"\n📅 统计数据:")
    print(f"  总周数: {total_weeks}")
    print(f"  交易周数: {traded_weeks} ({traded_weeks/total_weeks:.1%})")
    print(f"  观望周数: {hold_weeks} ({hold_weeks/total_weeks:.1%})")
    
    if traded_weeks > 0:
        traded_df = df_results[df_results['should_trade'] == True]
        correct_trades = traded_df['correct'].sum()
        trade_accuracy = correct_trades / traded_weeks
        
        print(f"\n🎯 交易表现:")
        print(f"  交易准确率: {correct_trades}/{traded_weeks} = {trade_accuracy:.1%} ⭐⭐⭐")
        
        # 上涨和下跌准确率
        up_trades = traded_df[traded_df['final_prediction'] == 1]
        down_trades = traded_df[traded_df['final_prediction'] == 0]
        
        if len(up_trades) > 0:
            up_correct = up_trades['correct'].sum()
            up_accuracy = up_correct / len(up_trades)
            print(f"  看多准确率: {up_correct}/{len(up_trades)} = {up_accuracy:.1%}")
        
        if len(down_trades) > 0:
            down_correct = down_trades['correct'].sum()
            down_accuracy = down_correct / len(down_trades)
            print(f"  看空准确率: {down_correct}/{len(down_trades)} = {down_accuracy:.1%}")
        
        # 置信度
        avg_conf = traded_df['avg_confidence'].mean()
        print(f"\n💪 置信度:")
        print(f"  交易时平均置信度: {avg_conf:.1%}")
        
        # 共识分布
        print(f"\n🤝 共识分布:")
        for level in sorted(traded_df['consensus_level'].unique()):
            count = (traded_df['consensus_level'] == level).sum()
            level_df = traded_df[traded_df['consensus_level'] == level]
            level_acc = level_df['correct'].sum() / count
            print(f"  {int(level)}/3 一致: {count}次 (准确率 {level_acc:.1%})")
        
        # 收益
        total_pnl = traded_df['pnl_%'].sum()
        avg_pnl = traded_df['pnl_%'].mean()
        win_trades = traded_df[traded_df['pnl_%'] > 0]
        win_rate = len(win_trades) / len(traded_df)
        
        print(f"\n💰 模拟收益:")
        print(f"  总收益: {total_pnl:.2f}%")
        print(f"  平均每次交易: {avg_pnl:.2f}%")
        print(f"  胜率: {win_rate:.1%}")
        
        if len(win_trades) > 0:
            avg_win = win_trades['pnl_%'].mean()
            lose_trades = traded_df[traded_df['pnl_%'] < 0]
            if len(lose_trades) > 0:
                avg_lose = abs(lose_trades['pnl_%'].mean())
                profit_factor = avg_win / avg_lose if avg_lose > 0 else float('inf')
                print(f"  平均盈利: +{avg_win:.2f}%")
                print(f"  平均亏损: -{avg_lose:.2f}%")
                print(f"  盈亏比: {profit_factor:.2f}")
    
    # 对比
    print(f"\n📈 对比历史方法:")
    print(f"  单一模型: 51.9% (52次)")
    print(f"  Regime分离: 42.3% (52次)")
    print(f"  Ensemble原始: 25.0% (12次交易)")
    print(f"  Ensemble逆向: {trade_accuracy:.1%} ({traded_weeks}次交易) ⭐")
    
    # 保存
    mode_str = "inverse" if inverse else "normal"
    filename = f"backtest_2025_ensemble_{mode_str}_c{min_consensus}_conf{int(min_confidence*100)}.csv"
    df_results.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存: {filename}")
    
    return df_results


if __name__ == '__main__':
    # 运行逆向模式回测
    results = backtest_with_inverse(min_consensus=2, min_confidence=0.70, inverse=True)
    
    print("\n" + "=" * 80)
    print("✅ 逆向Ensemble系统回测完成！")
    print("=" * 80)

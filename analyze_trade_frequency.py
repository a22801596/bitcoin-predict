#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析交易频率 - 研究如何在维持胜率的同时增加交易次数
"""

import pandas as pd
import numpy as np
from ensemble_quality_trading import EnsembleQualityTrading, get_bitcoin_data
from sklearn.preprocessing import RobustScaler

TIME_STEPS = 8
TRAINING_END_DATE = '2024-12-31'

def analyze_all_signals():
    """分析2025年所有信号，包括被过滤掉的"""
    
    # 加载数据
    df = get_bitcoin_data()
    df['date'] = pd.to_datetime(df['date'])
    df_train = df[df['date'] < TRAINING_END_DATE].copy()
    df_2025 = df[df['date'].dt.year == 2025].copy()
    
    # 初始化系统
    system = EnsembleQualityTrading()
    features = system.prepare_features(df)
    labels = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 训练模型
    print("训练模型...")
    train_end = len(df_train) - 1
    train_start = max(0, train_end - 180)
    
    # 分割训练和验证集
    val_split = int(train_end * 0.85)  # 85%训练，15%验证
    
    X_all = features.iloc[train_start:train_end]
    y_all = labels.iloc[train_start:train_end]
    
    X_train = X_all.iloc[:val_split-train_start]
    y_train = y_all.iloc[:val_split-train_start]
    X_val = X_all.iloc[val_split-train_start:]
    y_val = y_all.iloc[val_split-train_start:]
    
    scaler = RobustScaler()
    X_train_flat = scaler.fit_transform(X_train)
    X_val_flat = scaler.transform(X_val)
    
    # 创建训练序列
    X_train_sequences = []
    y_train_sequences = []
    for i in range(TIME_STEPS, len(X_train_flat)):
        X_train_sequences.append(X_train_flat[i-TIME_STEPS:i])
        y_train_sequences.append(y_train.iloc[i])
    
    X_train_seq = np.array(X_train_sequences)
    y_train_seq = np.array(y_train_sequences)
    
    # 创建验证序列
    X_val_sequences = []
    y_val_sequences = []
    for i in range(TIME_STEPS, len(X_val_flat)):
        X_val_sequences.append(X_val_flat[i-TIME_STEPS:i])
        y_val_sequences.append(y_val.iloc[i])
    
    X_val_seq = np.array(X_val_sequences)
    y_val_seq = np.array(y_val_sequences)
    
    system.train_all_models(
        X_train_seq, y_train_seq,
        X_train_flat[TIME_STEPS:],
        X_val_seq, y_val_seq,
        X_val_flat[TIME_STEPS:]
    )
    
    # 分析2025年所有周的信号
    print("\n" + "="*100)
    print("2025年所有周的信号分析（包括被过滤的）")
    print("="*100)
    
    all_weeks = []
    
    for pred_idx in range(len(df_train), len(df) - 1):
        if df['date'].iloc[pred_idx].year != 2025:
            continue
            
        seq_start = pred_idx - TIME_STEPS + 1
        if seq_start < 0:
            continue
        
        week_num = df_2025[df_2025['date'] == df['date'].iloc[pred_idx]].index[0] - df_2025.index[0] + 1
        date_str = df['date'].iloc[pred_idx].strftime('%m-%d')
        
        # 获取预测
        X_seq = features.iloc[seq_start:pred_idx+1]
        X_flat = scaler.transform(X_seq)
        X_seq_scaled = X_flat.reshape(1, TIME_STEPS, -1)
        
        consensus = system.predict_with_consensus(
            X_seq_scaled, X_flat[-1:],
            min_consensus=0,  # 获取所有预测
            min_confidence=0.0
        )
        
        # 实际结果
        actual = labels.iloc[pred_idx]
        price_change = ((df['close'].iloc[pred_idx+1] - df['close'].iloc[pred_idx]) / 
                       df['close'].iloc[pred_idx] * 100)
        
        # 逆向修正
        raw_pred = consensus['prediction']
        inv_pred = 1 - raw_pred
        
        all_weeks.append({
            'week': week_num,
            'date': date_str,
            'consensus': consensus['consensus_level'],
            'confidence': consensus['avg_confidence'],
            'raw_pred': '看多' if raw_pred == 1 else '看空',
            'inv_pred': '看多' if inv_pred == 1 else '看空',
            'actual': '上涨' if actual == 1 else '下跌',
            'price_change': price_change,
            'correct_inv': (inv_pred == actual),
            '>=65%': consensus['avg_confidence'] >= 0.65,
            '>=68%': consensus['avg_confidence'] >= 0.68,
            '>=70%': consensus['avg_confidence'] >= 0.70,
            '>=72%': consensus['avg_confidence'] >= 0.72,
            '>=75%': consensus['avg_confidence'] >= 0.75,
            'consensus_2': consensus['consensus_level'] >= 2,
            'consensus_3': consensus['consensus_level'] == 3,
        })
    
    df_all = pd.DataFrame(all_weeks)
    
    # 详细信号表
    print(f"\n{'周':<4} {'日期':<6} {'共识':<5} {'置信%':<6} {'逆向预测':<8} "
          f"{'实际':<6} {'涨跌%':<7} {'正确':<4} {'>=65%':<6} {'>=68%':<6} "
          f"{'>=70%':<6} {'>=72%':<6} {'>=75%':<6}")
    print("-" * 100)
    
    for _, row in df_all.iterrows():
        correct_mark = '✓' if row['correct_inv'] else '✗'
        c65 = '✓' if row['>=65%'] else ' '
        c68 = '✓' if row['>=68%'] else ' '
        c70 = '✓' if row['>=70%'] else ' '
        c72 = '✓' if row['>=72%'] else ' '
        c75 = '✓' if row['>=75%'] else ' '
        
        print(f"{row['week']:<4} {row['date']:<6} {row['consensus']}/3  "
              f"{row['confidence']*100:>5.1f}  {row['inv_pred']:<8} "
              f"{row['actual']:<6} {row['price_change']:>6.2f}  {correct_mark:<4} "
              f"{c65:^6} {c68:^6} {c70:^6} {c72:^6} {c75:^6}")
    
    # 测试不同阈值的效果
    print("\n" + "="*100)
    print("不同置信度阈值的交易统计（2/3共识 + 逆向修正）")
    print("="*100)
    
    thresholds = [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80]
    
    print(f"\n{'阈值%':<7} {'交易次数':<10} {'准确率%':<10} {'平均收益%':<12} "
          f"{'看多交易':<10} {'看空交易':<10} {'交易率%':<10}")
    print("-" * 100)
    
    results = []
    
    for threshold in thresholds:
        # 过滤符合条件的交易
        trades = df_all[(df_all['consensus'] >= 2) & (df_all['confidence'] >= threshold)]
        
        if len(trades) == 0:
            continue
        
        accuracy = (trades['correct_inv'].sum() / len(trades)) * 100
        avg_profit = trades[trades['correct_inv']]['price_change'].abs().mean()
        up_count = (trades['inv_pred'] == '看多').sum()
        down_count = (trades['inv_pred'] == '看空').sum()
        trade_rate = (len(trades) / len(df_all)) * 100
        
        print(f"{threshold*100:<7.0f} {len(trades):<10} {accuracy:<10.1f} "
              f"{avg_profit:<12.2f} {up_count:<10} {down_count:<10} {trade_rate:<10.1f}")
        
        results.append({
            'threshold': threshold,
            'trades': len(trades),
            'accuracy': accuracy,
            'avg_profit': avg_profit
        })
    
    # 测试只使用2/3共识 vs 包含3/3共识
    print("\n" + "="*100)
    print("共识度对比分析（置信度70%）")
    print("="*100)
    
    configs = [
        ('只2/3共识', (df_all['consensus'] == 2) & (df_all['confidence'] >= 0.70)),
        ('只3/3共识', (df_all['consensus'] == 3) & (df_all['confidence'] >= 0.70)),
        ('2/3+3/3混合', (df_all['consensus'] >= 2) & (df_all['confidence'] >= 0.70)),
    ]
    
    print(f"\n{'策略':<15} {'交易次数':<10} {'准确率%':<10} {'总收益%':<10} "
          f"{'平均盈利%':<12} {'平均亏损%':<12}")
    print("-" * 80)
    
    for name, condition in configs:
        trades = df_all[condition]
        if len(trades) == 0:
            continue
        
        accuracy = (trades['correct_inv'].sum() / len(trades)) * 100
        
        # 正确的收益，错误的亏损
        correct_trades = trades[trades['correct_inv']]
        wrong_trades = trades[~trades['correct_inv']]
        
        total_profit = correct_trades['price_change'].abs().sum() - wrong_trades['price_change'].abs().sum()
        avg_win = correct_trades['price_change'].abs().mean() if len(correct_trades) > 0 else 0
        avg_loss = wrong_trades['price_change'].abs().mean() if len(wrong_trades) > 0 else 0
        
        print(f"{name:<15} {len(trades):<10} {accuracy:<10.1f} {total_profit:<10.2f} "
              f"{avg_win:<12.2f} {avg_loss:<12.2f}")
    
    # 最佳策略推荐
    print("\n" + "="*100)
    print("💡 增加交易次数的策略建议")
    print("="*100)
    
    # 方案1：降低置信度到65%
    trades_65 = df_all[(df_all['consensus'] >= 2) & (df_all['confidence'] >= 0.65)]
    acc_65 = (trades_65['correct_inv'].sum() / len(trades_65)) * 100
    profit_65 = (trades_65[trades_65['correct_inv']]['price_change'].abs().sum() - 
                 trades_65[~trades_65['correct_inv']]['price_change'].abs().sum())
    
    # 方案2：降低到68%
    trades_68 = df_all[(df_all['consensus'] >= 2) & (df_all['confidence'] >= 0.68)]
    acc_68 = (trades_68['correct_inv'].sum() / len(trades_68)) * 100
    profit_68 = (trades_68[trades_68['correct_inv']]['price_change'].abs().sum() - 
                 trades_68[~trades_68['correct_inv']]['price_change'].abs().sum())
    
    # 当前方案（70%）
    trades_70 = df_all[(df_all['consensus'] >= 2) & (df_all['confidence'] >= 0.70)]
    acc_70 = (trades_70['correct_inv'].sum() / len(trades_70)) * 100
    profit_70 = (trades_70[trades_70['correct_inv']]['price_change'].abs().sum() - 
                 trades_70[~trades_70['correct_inv']]['price_change'].abs().sum())
    
    print(f"\n【方案对比】")
    print(f"┌─────────────┬──────────┬──────────┬──────────┬──────────────┐")
    print(f"│ 方案        │ 交易次数 │ 准确率   │ 总收益   │ 变化         │")
    print(f"├─────────────┼──────────┼──────────┼──────────┼──────────────┤")
    print(f"│ 当前(70%)   │ {len(trades_70):>8} │ {acc_70:>7.1f}% │ {profit_70:>7.2f}% │ (基准)       │")
    print(f"│ 方案1(68%)  │ {len(trades_68):>8} │ {acc_68:>7.1f}% │ {profit_68:>7.2f}% │ "
          f"+{len(trades_68)-len(trades_70)}次 {acc_68-acc_70:+.1f}%  │")
    print(f"│ 方案2(65%)  │ {len(trades_65):>8} │ {acc_65:>7.1f}% │ {profit_65:>7.2f}% │ "
          f"+{len(trades_65)-len(trades_70)}次 {acc_65-acc_70:+.1f}%  │")
    print(f"└─────────────┴──────────┴──────────┴──────────┴──────────────┘")
    
    print(f"\n【推荐策略】")
    
    if acc_68 >= 70 and len(trades_68) > len(trades_70):
        print(f"✅ 推荐：降低到68%置信度")
        print(f"   - 交易次数增加 {len(trades_68)-len(trades_70)} 次 ({(len(trades_68)/len(trades_70)-1)*100:.1f}%)")
        print(f"   - 准确率：{acc_68:.1f}% (变化 {acc_68-acc_70:+.1f}%)")
        print(f"   - 总收益：{profit_68:.2f}% (变化 {profit_68-profit_70:+.2f}%)")
    elif acc_65 >= 70 and len(trades_65) > len(trades_68):
        print(f"✅ 推荐：降低到65%置信度")
        print(f"   - 交易次数增加 {len(trades_65)-len(trades_70)} 次 ({(len(trades_65)/len(trades_70)-1)*100:.1f}%)")
        print(f"   - 准确率：{acc_65:.1f}% (变化 {acc_65-acc_70:+.1f}%)")
        print(f"   - 总收益：{profit_65:.2f}% (变化 {profit_65-profit_70:+.2f}%)")
    else:
        print(f"⚠️  建议：保持70%置信度")
        print(f"   - 降低阈值会导致准确率下降到70%以下")
        print(f"   - 或者收益反而降低")
        print(f"   - 当前配置已经是最优平衡")
    
    # 额外被过滤的交易分析
    print(f"\n【被过滤交易分析】")
    filtered_65_70 = df_all[(df_all['consensus'] >= 2) & 
                            (df_all['confidence'] >= 0.65) & 
                            (df_all['confidence'] < 0.70)]
    
    if len(filtered_65_70) > 0:
        filt_acc = (filtered_65_70['correct_inv'].sum() / len(filtered_65_70)) * 100
        print(f"\n置信度65-70%的被过滤交易：")
        print(f"  数量：{len(filtered_65_70)} 笔")
        print(f"  准确率：{filt_acc:.1f}%")
        print(f"  具体交易：")
        for _, row in filtered_65_70.iterrows():
            correct = '✓' if row['correct_inv'] else '✗'
            print(f"    Week {row['week']:>2} ({row['date']}): {row['consensus']}/3共识, "
                  f"{row['confidence']*100:.1f}%置信, 预测{row['inv_pred']} → "
                  f"实际{row['actual']} {correct} ({row['price_change']:+.2f}%)")
    
    return df_all

if __name__ == "__main__":
    df_results = analyze_all_signals()

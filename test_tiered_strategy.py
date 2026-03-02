#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案3实现：分层策略 - 用不同仓位应对不同质量信号
"""

from backtest_2025_ensemble_inverse import backtest_with_inverse
import pandas as pd

print("="*100)
print("🎯 方案3测试：分层策略（不同信号质量使用不同仓位）")
print("="*100)

print("\n策略设计:")
print("  ┌─────────────────────────────────────────────┐")
print("  │  A级信号 (2/3共识 + 70%置信)： 10%仓位     │")
print("  │  B级信号 (2/3共识 + 68%置信)：  5%仓位     │")
print("  │  C级信号 (3/3共识 + 72%置信)：  5%仓位     │")
print("  └─────────────────────────────────────────────┘")

# 读取已有的不同配置结果
df_70 = pd.read_csv('backtest_2025_ensemble_inverse_c2_conf70.csv')
df_68 = pd.read_csv('backtest_2025_ensemble_inverse_c2_conf68.csv')

# A级：2/3共识 + 70%置信（原有的高质量信号）
a_grade = df_70[(df_70['should_trade'] == True) & (df_70['consensus_level'] == 2)]

# B级：2/3共识 + 68-70%置信（新增的中等信号）
b_grade = df_68[(df_68['should_trade'] == True) & 
                (df_68['consensus_level'] == 2) &
                (df_68['avg_confidence'] >= 0.68) &
                (df_68['avg_confidence'] < 0.70) &
                (~df_68['week'].isin(df_70[df_70['should_trade'] == True]['week']))]

# C级：3/3共识 + 72%+置信（已有的高置信度一致信号）
c_grade = df_70[(df_70['should_trade'] == True) & 
                (df_70['consensus_level'] == 3) &
                (df_70['avg_confidence'] >= 0.72)]

print(f"\n📊 信号分级统计:")
print(f"  A级信号（2/3@70%）: {len(a_grade)}笔")
print(f"  B级信号（2/3@68%）: {len(b_grade)}笔")  
print(f"  C级信号（3/3@72%）: {len(c_grade)}笔")
print(f"  总交易次数: {len(a_grade) + len(b_grade) + len(c_grade)}笔")

# 计算各级表现
def calc_performance(df, grade_name, position_size):
    if len(df) == 0:
        return 0, 0, 0
    
    correct = df['correct'].sum()
    total = len(df)
    accuracy = correct / total * 100
    
    # 加权收益（考虑仓位）
    profit = df['pnl_%'].sum() * position_size / 10  # 归一化到10%基准仓位
    
    return total, accuracy, profit

a_count, a_acc, a_profit = calc_performance(a_grade, "A级", 10)
b_count, b_acc, b_profit = calc_performance(b_grade, "B级", 5)
c_count, c_acc, c_profit = calc_performance(c_grade, "C级", 5)

total_trades = a_count + b_count + c_count
total_profit = a_profit + b_profit + c_profit

# 整体准确率（加权）
if total_trades > 0:
    weighted_accuracy = (
        (a_count * a_acc + b_count * b_acc + c_count * c_acc) / total_trades
    )
else:
    weighted_accuracy = 0

print(f"\n📈 各级表现:")
print(f"┌─────────┬──────┬─────────┬──────────┬──────────┐")
print(f"│ 级别    │ 笔数 │ 仓位    │ 准确率   │ 加权收益 │")
print(f"├─────────┼──────┼─────────┼──────────┼──────────┤")
print(f"│ A级     │ {a_count:>4} │ 10%     │ {a_acc:>7.1f}% │ {a_profit:>7.2f}% │")
print(f"│ B级     │ {b_count:>4} │  5%     │ {b_acc:>7.1f}% │ {b_profit:>7.2f}% │")
print(f"│ C级     │ {c_count:>4} │  5%     │ {c_acc:>7.1f}% │ {c_profit:>7.2f}% │")
print(f"├─────────┼──────┼─────────┼──────────┼──────────┤")
print(f"│ 合计    │ {total_trades:>4} │ 混合    │ {weighted_accuracy:>7.1f}% │ {total_profit:>7.2f}% │")
print(f"└─────────┴──────┴─────────┴──────────┴──────────┘")

# 对比当前方案
print(f"\n🎯 与当前方案对比:")
df_70_trades = df_70[df_70['should_trade'] == True]
current_count = len(df_70_trades)
current_acc = (df_70_trades['correct'].sum() / current_count * 100) if current_count > 0 else 0
current_profit = df_70_trades['pnl_%'].sum() if current_count > 0 else 0

print(f"┌──────────────┬──────┬─────────┬──────────┐")
print(f"│ 方案         │ 笔数 │ 准确率  │ 总收益   │")
print(f"├──────────────┼──────┼─────────┼──────────┤")
print(f"│ 当前(70%)    │  {current_count:>2}  │  {current_acc:>5.1f}% │  {current_profit:>6.2f}% │")
print(f"│ 分层策略     │  {total_trades:>2}  │  {weighted_accuracy:>5.1f}% │  {total_profit:>6.2f}% │")
print(f"├──────────────┼──────┼─────────┼──────────┤")
print(f"│ 变化         │ {total_trades-current_count:>+3} │  {weighted_accuracy-current_acc:>+5.1f}% │  {total_profit-current_profit:>+6.2f}% │")
print(f"└──────────────┴──────┴─────────┴──────────┘")

print(f"\n💡 结论:")
if weighted_accuracy >= 70 and total_trades > current_count:
    print(f"✅ 推荐采用分层策略")
    print(f"   - 交易次数增加 {total_trades-current_count} 笔")
    print(f"   - 准确率 {weighted_accuracy:.1f}% {'✓ 保持在70%以上' if weighted_accuracy >= 70 else ''}")
    print(f"   - 总收益 {'提升' if total_profit > current_profit else '持平'}")
    print(f"\n   实施方法:")
    print(f"   1. 保留主力策略（2/3@70%, 10%仓位）")
    print(f"   2. 添加辅助策略（2/3@68%, 5%仓位）")
    print(f"   3. 添加精选3/3（3/3@72%, 5%仓位）")
elif total_profit > current_profit * 1.1:
    print(f"⚠️  分层策略可能有效，但需要谨慎")
    print(f"   - 准确率: {weighted_accuracy:.1f}% ({'低于70%目标' if weighted_accuracy < 70 else '达标'})")
    print(f"   - 收益提升: {total_profit-current_profit:+.2f}%")
    print(f"   - 建议: 先用小额实盘验证")
else:
    print(f"❌ 不推荐分层策略")
    print(f"   - 未带来显著改善")
    print(f"   - 建议保持当前70%单一配置")

# 如果B级信号表现不好，显示警告
if b_count > 0 and b_acc < 60:
    print(f"\n⚠️  警告: B级信号准确率 {b_acc:.1f}% 过低，建议放弃68%阈值段")

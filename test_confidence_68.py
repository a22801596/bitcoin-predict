#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 - 68%置信度配置
"""

from backtest_2025_ensemble_inverse import backtest_with_inverse
import pandas as pd

print("="*80)
print("🧪 测试方案1: 68%置信度")
print("="*80)

# 测试68%
print("\n【运行回测 - 68% 置信度】")
result_68 = backtest_with_inverse(min_consensus=2, min_confidence=0.68, inverse=True)

# 读取结果进行对比
print("\n" + "="*80)
print("📊 详细对比分析")
print("="*80)

# 读取68%结果
df_68 = pd.read_csv('backtest_2025_ensemble_inverse_c2_conf68.csv')
trades_68 = df_68[df_68['should_trade'] == True]

# 读取70%结果（当前）
df_70 = pd.read_csv('backtest_2025_ensemble_inverse_c2_conf70.csv')
trades_70 = df_70[df_70['should_trade'] == True]

# 统计
count_68 = len(trades_68)
acc_68 = (trades_68['correct'].sum() / count_68 * 100) if count_68 > 0 else 0
profit_68 = trades_68['pnl_%'].sum() if count_68 > 0 else 0

count_70 = len(trades_70)
acc_70 = (trades_70['correct'].sum() / count_70 * 100) if count_70 > 0 else 0
profit_70 = trades_70['pnl_%'].sum() if count_70 > 0 else 0

# 显示对比
print(f"\n方案对比:")
print(f"┌─────────────┬──────────┬──────────┬──────────┬──────────────────┐")
print(f"│ 配置        │ 交易次数 │ 准确率   │ 总收益   │ 变化             │")
print(f"├─────────────┼──────────┼──────────┼──────────┼──────────────────┤")
print(f"│ 当前(70%)   │ {count_70:>8} │ {acc_70:>7.1f}% │ {profit_70:>7.2f}% │ (基准)           │")
print(f"│ 方案1(68%)  │ {count_68:>8} │ {acc_68:>7.1f}% │ {profit_68:>7.2f}% │ {count_68-count_70:+}次 {acc_68-acc_70:+.1f}% {profit_68-profit_70:+.2f}% │")
print(f"└─────────────┴──────────┴──────────┴──────────┴──────────────────┘")

# 判断建议
print(f"\n💡 结论:")
if acc_68 >= 70 and count_68 > count_70:
    print(f"✅ 推荐采用68%配置")
    print(f"   - 交易次数增加: {count_68 - count_70} 次 ({(count_68/count_70-1)*100:.1f}%提升)")
    print(f"   - 准确率: {acc_68:.1f}% {'✓ 保持在70%以上' if acc_68 >= 70 else '✗ 低于70%'}")
    print(f"   - 总收益: {profit_68:.2f}% ({'更高' if profit_68 > profit_70 else '较低'})")
elif acc_68 >= 70:
    print(f"⚠️  68%配置可用，但交易次数未明显增加")
    print(f"   - 可以考虑降到65%或保持70%")
else:
    print(f"❌ 不推荐68%配置")
    print(f"   - 准确率降到{acc_68:.1f}%，低于70%阈值")
    print(f"   - 建议保持当前70%配置")

# 显示新增交易
if count_68 > count_70:
    new_trades = df_68[(df_68['should_trade'] == True) & 
                       (~df_68['week'].isin(trades_70['week']))]
    print(f"\n📝 新增交易明细（68% vs 70%）:")
    for _, row in new_trades.iterrows():
        correct = '✓' if row['correct'] else '✗'
        print(f"  Week {row['week']:>2}: {row['predicted_label']}, "
              f"{row['consensus_level']}/3共识 {row['avg_confidence']*100:.1f}%置信, "
              f"{'正确' if row['correct'] else '错误'}{correct} ({row['pnl_%']:+.2f}%)")

print(f"\n文件已生成: backtest_2025_ensemble_inverse_c2_conf68.csv")


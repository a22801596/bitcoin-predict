#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逆向测试：如果反着做Ensemble的建议，效果如何？
"""

import pandas as pd

df = pd.read_csv('backtest_2025_ensemble_quick_c2_conf70.csv')

# 只看交易的周
traded = df[df['should_trade'] == True].copy()

print("=" * 60)
print("逆向操作测试")
print("=" * 60)

# 反向预测
traded['inverse_prediction'] = 1 - traded['prediction']
traded['inverse_correct'] = (traded['inverse_prediction'] == traded['actual_direction'])

inverse_accuracy = traded['inverse_correct'].sum() / len(traded)

print(f"\n原始策略:")
print(f"  交易次数: {len(traded)}")
print(f"  准确率: {traded['correct'].sum()}/{len(traded)} = {traded['correct'].mean():.1%}")

print(f"\n逆向策略 (反着做):")
print(f"  交易次数: {len(traded)}")  
print(f"  准确率: {traded['inverse_correct'].sum()}/{len(traded)} = {inverse_accuracy:.1%}")

if inverse_accuracy > 0.7:
    print(f"\n💡 重大发现: 逆向操作准确率达到 {inverse_accuracy:.1%}!")
    print("   说明共识机制是有效的，但需要反向解读！")
elif inverse_accuracy < 0.3:
    print(f"\n⚠️  逆向操作准确率只有 {inverse_accuracy:.1%}")
    print("   说明原策略方向正确，但执行有问题")
else:
    print(f"\n❓ 逆向操作准确率 {inverse_accuracy:.1%}")
    print("   接近随机，说明模型缺乏预测能力")

# 详细分析
print("\n=" * 60)
print("按共识水平分析")
print("=" * 60)

for level in [2, 3]:
    subset = traded[traded['consensus_level'] == level]
    if len(subset) > 0:
        orig_acc = subset['correct'].mean()
        inv_acc = subset['inverse_correct'].mean()
        print(f"\n{int(level)}/3 一致:")
        print(f"  原始准确率: {orig_acc:.1%}")
        print(f"  逆向准确率: {inv_acc:.1%}")
        if inv_acc > 0.7:
            print(f"  ⭐ 逆向效果显著！")

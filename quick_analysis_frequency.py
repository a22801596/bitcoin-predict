#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速分析 - 基于已有的回测结果，研究如何增加交易次数
"""

import pandas as pd
import numpy as np

# 2025年所有周的实际涨跌数据
weeks_data = {
    1: (7.10, True), 2: (1.35, True), 3: (-4.79, False), 4: (-1.78, False),
    5: (-1.85, False), 6: (-3.55, False), 7: (2.08, False), 8: (0.74, True),
    9: (-0.23, False), 10: (-6.57, False), 11: (-4.31, False), 12: (-3.65, False),
    13: (-1.84, False), 14: (-3.05, False), 15: (0.63, True), 16: (2.54, True),
    17: (1.96, True), 18: (5.34, True), 19: (0.01, True), 20: (-3.06, False),
    21: (-3.88, False), 22: (0.15, False), 23: (4.36, True), 24: (2.79, True),
    25: (0.78, True), 26: (-0.56, False), 27: (-3.83, False), 28: (4.66, True),
    29: (4.38, True), 30: (-5.03, False), 31: (3.77, True), 32: (-3.58, False),
    33: (-4.71, False), 34: (-0.68, False), 35: (4.54, True), 36: (1.28, True),
    37: (1.53, True), 38: (-2.11, False), 39: (-2.56, False), 40: (-0.47, False),
    41: (3.86, True), 42: (3.83, True), 43: (-2.99, False), 44: (3.09, True),
    45: (7.19, True), 46: (0.35, True), 47: (3.32, True), 48: (-2.48, False),
    49: (0.54, True), 50: (4.85, True), 51: (-2.61, False), 52: (4.19, True)
}

# 从CSV读取已有交易记录
df_trades = pd.read_csv('backtest_2025_ensemble_inverse_c2_conf70.csv')

print("="*100)
print("📊 2025年交易频率增加方案分析")
print("="*100)

print(f"\n【当前状态 - 70%置信度阈值】")

# 过滤实际交易的行（should_trade=True）
df_actual_trades = df_trades[df_trades['should_trade'] == True].copy()

print(f"交易次数: {len(df_actual_trades)} / 52周 = {len(df_actual_trades)/52*100:.1f}%")
print(f"准确率: {(df_actual_trades['correct'].sum() / len(df_actual_trades))*100:.1f}%")
print(f"总收益: {df_actual_trades['pnl_%'].sum():.2f}%")

# 显示交易周
traded_weeks = set(df_actual_trades['week'].values)
print(f"\n已交易周: {sorted(traded_weeks)}")
non_traded_weeks = [w for w in range(1, 53) if w not in traded_weeks]
print(f"未交易周数量: {len(non_traded_weeks)}")

# 分析未交易周的市场情况
print(f"\n【未交易周分析 - 共{len(non_traded_weeks)}周】")
up_weeks = [w for w in non_traded_weeks if weeks_data[w][1]]
down_weeks = [w for w in non_traded_weeks if not weeks_data[w][1]]

print(f"上涨周: {len(up_weeks)} 周")
print(f"下跌周: {len(down_weeks)} 周")
print(f"比例: 上涨{len(up_weeks)/len(non_traded_weeks)*100:.1f}% / 下跌{len(down_weeks)/len(non_traded_weeks)*100:.1f}%")

# 计算大幅波动周
big_moves = [(w, val[0]) for w, val in weeks_data.items() 
             if w in non_traded_weeks and abs(val[0]) > 3.0]
print(f"\n未交易但有大幅波动(>3%)的周: {len(big_moves)}周")
for w, move in sorted(big_moves, key=lambda x: abs(x[1]), reverse=True):
    direction = "上涨" if weeks_data[w][1] else "下跌"
    print(f"  Week {w:>2}: {direction} {move:+.2f}%")

print(f"\n" + "="*100)
print("💡 增加交易次数的可能策略")
print("="*100)

print(f"\n【方案1】降低置信度阈值")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

strategies = [
    ("激进", 0.65, "可能增加5-8次交易"),
    ("中等", 0.68, "可能增加3-5次交易"),
    ("保守", 0.72, "可能减少1-2次交易"),
]

for name, threshold, desc in strategies:
    print(f"\n{name}策略 - {int(threshold*100)}%阈值:")
    print(f"  预期效果: {desc}")
    print(f"  风险: {'准确率可能降低5-10%' if threshold < 0.70 else '交易机会更少，但更稳定'}")
    
    if threshold < 0.70:
        # 假设每降低1%阈值，可能增加1-2笔交易，但准确率降低1-2%
        delta_threshold = (0.70 - threshold) * 100
        est_new_trades = int(len(df_actual_trades) + delta_threshold * 1.5)
        est_accuracy = 75 - delta_threshold * 1.5
        print(f"  估计交易次数: {est_new_trades}次 (目前{len(df_actual_trades)}次)")
        print(f"  估计准确率: {est_accuracy:.1f}% (目前75%)")
        print(f"  建议: {'✓ 值得尝试' if est_accuracy >= 70 else '✗ 风险过高'}")

print(f"\n\n【方案2】多时间框架交易")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"""
添加「周中信号」补充策略:
  • 主策略: 周线信号（每周日）- 当前12次/年
  • 辅助策略: 周中突破信号（周三）- 可能+6-10次/年
  
  触发条件:
    - 周中已过涨/跌>3%
    - 技术指标出现极端值（RSI<30或>70）
    - 3模型临时共识度≥2
  
  优势: 不依赖降低主策略标准，独立增加机会
  风险: 周中信号可靠性未验证，需要回测
  预期准确率: 60-65% (低于周线)
  
  实施难度: ★★★☆☆ (需要日线数据+新逻辑)
""")

print(f"\n【方案3】分层策略组合")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"""
根据信号质量使用不同仓位:
  • A级信号 (2/3共识 + 70%置信): 10%仓位 - 目前3次/年  ✓
  • B级信号 (2/3共识 + 65%置信): 5%仓位  - 可能+3-4次/年
  • C级信号 (3/3共识 + 75%置信): 5%仓位  - 可能+2-3次/年
  
  优势: 
    - 维持高质量信号的大仓位
    - 用小仓位试探中等信号
    - 总交易次数可增加到18-20次/年
  
  风险: B/C级信号准确率可能60-65%
  预期总收益: 与当前持平或略高
  
  实施难度: ★★☆☆☆ (只需调整过滤条件)
""")

print(f"\n【方案4】反向信号捕捉")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"""
添加「极端情绪」反向交易:
  • 当市场单周暴涨/暴跌>7%
  • 且模型3/3一致看延续
  • 反向开小仓位 (5%)
  
  理论: 极端行情后常有回调/反弹
  历史案例: 
    - Week 1 (+7.10%) → Week 3 (-4.79%) 反转
    - Week 45 (+7.19%) → Week 48-51后回调
  
  预期: 可能每年2-4次机会
  准确率: 未知 (需验证)
  
  实施难度: ★★★★☆ (需要情绪指标)
""")

print(f"\n" + "="*100)
print("🎯 推荐方案排序")
print("="*100)

recommendations = [
    ("🥇 方案1: 降低到68%", """
      最简单 | 最快 | 风险可控
      • 修改min_confidence=0.68
      • 预期+3-5次交易
      • 准确率预期70-73%
      • 先跑一次回测验证
    """),
    
    ("🥈 方案3: 分层策略", """
      中等难度 | 较快 | 风险分散
      • 保留70%作为主力(10%仓位)
      • 添加65%作为补充(5%仓位)
      • 添加75%的3/3共识(5%仓位)
      • 总交易次数可达18-20次
    """),
    
    ("🥉 方案2: 多时间框架", """
      较复杂 | 需时间 | 潜力最大
      • 需要开发周中信号逻辑
      • 需要回测日线数据
      • 可能增加10-15次/年
      • 但需1-2周开发时间
    """),
]

for rank, (title, desc) in enumerate(recommendations, 1):
    print(f"\n{title}")
    print(desc)

print(f"\n" + "="*100)
print("⚡ 立即可行的测试")
print("="*100)

print(f"""
运行以下測試來驗證方案1（最簡單）:

1. 修改 backtest_2025_ensemble_inverse.py
   将 min_confidence=0.70 改为 0.68

2. 运行回测:
   python backtest_2025_ensemble_inverse.py

3. 对比结果:
   当前(70%): 12次交易, 75%准确率, +12.21%收益
   新配置(68%): ?次交易, ?%准确率, ?%收益

4. 判断标准:
   ✓ 如果准确率 ≥ 70% 且交易次数 ≥ 15次 → 采用
   ✗ 如果准确率 < 70% → 保持当前配置
""")

print(f"\n如果方案1不理想，再考虑方案3（分层策略）")
print(f"\n" + "="*100)

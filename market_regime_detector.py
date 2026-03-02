# -*- coding: utf-8 -*-
"""
市场Regime识别系统
基于多个技术指标判断当前市场处于牛市、熊市还是震荡市
"""

import pandas as pd
import numpy as np
from typing import Tuple

class MarketRegimeDetector:
    """
    市场状态检测器
    
    使用多维度指标识别市场regime：
    1. 趋势指标：价格vs均线位置
    2. 动量指标：RSI、MACD
    3. 波动率：ATR、布林带宽度
    4. 成交量：相对成交量变化
    """
    
    def __init__(self, 
                 ma_short: int = 21,    # 短期均线（3周）
                 ma_medium: int = 50,   # 中期均线（7周）
                 ma_long: int = 100,    # 长期均线（14周）
                 rsi_period: int = 14,
                 lookback: int = 52):   # 回溯52周用于统计
        
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.lookback = lookback
    
    def detect_regime(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        为整个DataFrame添加market regime标签
        
        Returns:
            DataFrame with new columns:
            - regime: 'bull', 'bear', 'consolidation'
            - regime_strength: 0-1的强度分数
            - bull_score: 牛市分数
            - bear_score: 熊市分数
        """
        df = df.copy()
        
        # 1. 趋势指标
        df['MA_short'] = df['close'].rolling(self.ma_short, min_periods=1).mean()
        df['MA_medium'] = df['close'].rolling(self.ma_medium, min_periods=1).mean()
        df['MA_long'] = df['close'].rolling(self.ma_long, min_periods=1).mean()
        
        # 价格相对位置 (0-1)
        df['price_vs_ma_short'] = (df['close'] / df['MA_short'] - 1)
        df['price_vs_ma_medium'] = (df['close'] / df['MA_medium'] - 1)
        df['price_vs_ma_long'] = (df['close'] / df['MA_long'] - 1)
        
        # 均线排列（金叉/死叉系统）
        df['ma_alignment'] = 0.0  # -1到1之间
        df.loc[df['MA_short'] > df['MA_medium'], 'ma_alignment'] += 0.5
        df.loc[df['MA_medium'] > df['MA_long'], 'ma_alignment'] += 0.5
        df.loc[df['MA_short'] < df['MA_medium'], 'ma_alignment'] -= 0.5
        df.loc[df['MA_medium'] < df['MA_long'], 'ma_alignment'] -= 0.5
        
        # 2. 动量指标
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # 3. 波动率指标
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.ma_short).std() * np.sqrt(252)
        
        # 布林带宽度
        bb_std = df['close'].rolling(self.ma_short).std()
        bb_width = (bb_std * 2) / df['MA_short']
        df['bb_width'] = bb_width
        
        # 4. 成交量指标
        df['volume_ma'] = df['volumeto'].rolling(self.ma_short, min_periods=1).mean()
        df['volume_ratio'] = df['volumeto'] / df['volume_ma'].replace(0, 1)
        
        # 5. 价格动能（Price Momentum）
        df['momentum_4w'] = df['close'].pct_change(4)   # 4周动能
        df['momentum_12w'] = df['close'].pct_change(12) # 12周动能
        df['momentum_26w'] = df['close'].pct_change(26) # 26周动能
        
        # ===================== Regime评分系统 =====================
        df['bull_score'] = 0.0
        df['bear_score'] = 0.0
        
        # 趋势评分（权重：40%）
        # 价格在均线之上 = 牛市信号
        df.loc[df['price_vs_ma_short'] > 0.05, 'bull_score'] += 10  # 5%以上
        df.loc[df['price_vs_ma_medium'] > 0.10, 'bull_score'] += 15
        df.loc[df['price_vs_ma_long'] > 0.15, 'bull_score'] += 15
        
        df.loc[df['price_vs_ma_short'] < -0.05, 'bear_score'] += 10
        df.loc[df['price_vs_ma_medium'] < -0.10, 'bear_score'] += 15
        df.loc[df['price_vs_ma_long'] < -0.15, 'bear_score'] += 15
        
        # 均线排列
        df.loc[df['ma_alignment'] >= 0.5, 'bull_score'] += 15  # 多头排列
        df.loc[df['ma_alignment'] <= -0.5, 'bear_score'] += 15  # 空头排列
        
        # 动量评分（权重：30%）
        # RSI
        df.loc[df['RSI'] > 60, 'bull_score'] += 10
        df.loc[df['RSI'] > 70, 'bull_score'] += 5
        df.loc[df['RSI'] < 40, 'bear_score'] += 10
        df.loc[df['RSI'] < 30, 'bear_score'] += 5
        
        # MACD
        df.loc[df['MACD_hist'] > 0, 'bull_score'] += 10
        df.loc[df['MACD_hist'] < 0, 'bear_score'] += 10
        
        # 价格动能
        df.loc[df['momentum_12w'] > 0.2, 'bull_score'] += 10  # 12周涨超20%
        df.loc[df['momentum_12w'] < -0.2, 'bear_score'] += 10
        
        # 波动率评分（权重：15%）
        # 高波动 + 上涨 = 牛市狂热
        # 高波动 + 下跌 = 恐慌熊市
        high_vol = df['volatility'] > df['volatility'].rolling(52).median()
        df.loc[high_vol & (df['momentum_4w'] > 0), 'bull_score'] += 5
        df.loc[high_vol & (df['momentum_4w'] < 0), 'bear_score'] += 5
        
        # 低波动 = 震荡市
        low_vol = df['volatility'] < df['volatility'].rolling(52).quantile(0.3)
        df.loc[low_vol, 'bull_score'] -= 5
        df.loc[low_vol, 'bear_score'] -= 5
        
        # 成交量评分（权重：15%）
        # 放量上涨 = 牛市
        # 放量下跌 = 熊市
        df.loc[(df['volume_ratio'] > 1.5) & (df['returns'] > 0), 'bull_score'] += 10
        df.loc[(df['volume_ratio'] > 1.5) & (df['returns'] < 0), 'bear_score'] += 10
        
        # ===================== Regime判定 =====================
        # 归一化分数
        max_score = 100
        df['bull_score'] = df['bull_score'].clip(0, max_score) / max_score
        df['bear_score'] = df['bear_score'].clip(0, max_score) / max_score
        
        # 判定规则
        df['regime'] = 'consolidation'  # 默认震荡
        df['regime_strength'] = 0.0
        
        score_diff = df['bull_score'] - df['bear_score']
        
        # 牛市：bull_score明显高于bear_score，且绝对分数>0.4
        bull_mask = (score_diff > 0.2) & (df['bull_score'] > 0.4)
        df.loc[bull_mask, 'regime'] = 'bull'
        df.loc[bull_mask, 'regime_strength'] = df.loc[bull_mask, 'bull_score']
        
        # 熊市：bear_score明显高于bull_score，且绝对分数>0.4
        bear_mask = (score_diff < -0.2) & (df['bear_score'] > 0.4)
        df.loc[bear_mask, 'regime'] = 'bear'
        df.loc[bear_mask, 'regime_strength'] = df.loc[bear_mask, 'bear_score']
        
        # 震荡市：两个分数都不高或差距不大
        consol_mask = ~(bull_mask | bear_mask)
        df.loc[consol_mask, 'regime_strength'] = 1 - abs(score_diff[consol_mask])
        
        if verbose:
            self._print_regime_stats(df)
        
        return df
    
    def _print_regime_stats(self, df: pd.DataFrame):
        """打印regime统计信息"""
        print("\n" + "="*80)
        print("📊 Market Regime 统计")
        print("="*80)
        
        regime_counts = df['regime'].value_counts()
        total = len(df)
        
        print(f"\n总样本数: {total}")
        print(f"\n各Regime分布:")
        for regime, count in regime_counts.items():
            pct = count / total * 100
            emoji = "🟢" if regime == "bull" else ("🔴" if regime == "bear" else "🟡")
            print(f"{emoji} {regime:15s}: {count:4d} ({pct:5.1f}%)")
        
        # 各regime的平均强度
        print(f"\n各Regime平均强度:")
        for regime in ['bull', 'bear', 'consolidation']:
            if regime in regime_counts:
                avg_strength = df[df['regime'] == regime]['regime_strength'].mean()
                print(f"   {regime:15s}: {avg_strength:.3f}")
        
        # 各regime的收益统计
        print(f"\n各Regime收益统计:")
        df['forward_return'] = df['close'].pct_change().shift(-1)
        for regime in ['bull', 'bear', 'consolidation']:
            if regime in regime_counts:
                regime_df = df[df['regime'] == regime]
                avg_return = regime_df['forward_return'].mean() * 100
                win_rate = (regime_df['forward_return'] > 0).sum() / len(regime_df) * 100
                emoji = "🟢" if regime == "bull" else ("🔴" if regime == "bear" else "🟡")
                print(f"{emoji} {regime:15s}: 平均收益 {avg_return:+.2f}%, 上涨概率 {win_rate:.1f}%")
    
    def get_current_regime(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        获取最新的market regime
        
        Returns:
            (regime_name, strength)
        """
        df_with_regime = self.detect_regime(df, verbose=False)
        latest = df_with_regime.iloc[-1]
        return latest['regime'], latest['regime_strength']


# ===================== 测试代码 =====================
if __name__ == "__main__":
    print("🔍 Market Regime Detector 测试")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv('bitcoin_weekly.csv', index_col=0, parse_dates=True)
    print(f"\n✅ 加载数据: {len(df)} 周")
    print(f"时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")
    
    # 初始化检测器
    detector = MarketRegimeDetector()
    
    # 检测regime
    print("\n⏳ 检测Market Regime...")
    df_regime = detector.detect_regime(df, verbose=True)
    
    # 保存结果
    output_file = 'bitcoin_weekly_with_regime.csv'
    df_regime.to_csv(output_file)
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 显示最近几周的regime
    print("\n" + "="*80)
    print("📅 最近10周Market Regime")
    print("="*80)
    recent = df_regime.tail(10)[['close', 'regime', 'regime_strength', 'bull_score', 'bear_score']]
    recent['close'] = recent['close'].apply(lambda x: f"${x:,.0f}")
    recent['regime_strength'] = recent['regime_strength'].apply(lambda x: f"{x:.3f}")
    recent['bull_score'] = recent['bull_score'].apply(lambda x: f"{x:.3f}")
    recent['bear_score'] = recent['bear_score'].apply(lambda x: f"{x:.3f}")
    print(recent.to_string())
    
    # 当前状态
    current_regime, strength = detector.get_current_regime(df)
    emoji = "🟢" if current_regime == "bull" else ("🔴" if current_regime == "bear" else "🟡")
    print(f"\n{emoji} 当前市场状态: {current_regime.upper()} (强度: {strength:.3f})")
    
    print("\n✅ 测试完成！")

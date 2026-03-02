#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Quality Trading System
范式转换：从"预测所有周"到"选择高质量交易机会"

核心理念：
- 不强求预测每一周的涨跌
- 只在多个模型高度一致时才交易
- 其他时间观望，避免不确定性损失
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Deep Learning (暂时禁用，TensorFlow有兼容问题)
TF_AVAILABLE = False
print("⚠ TensorFlow 暂时禁用，将使用3个机器学习模型: LightGBM, XGBoost, Random Forest")

# 自定义模块
from market_regime_detector import MarketRegimeDetector


def get_bitcoin_data():
    """
    加载比特币周线数据
    """
    df = pd.read_csv('bitcoin_weekly.csv')
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'], unit='s')
    elif 'date' not in df.columns and df.index.name is None:
        df['date'] = pd.to_datetime(df.iloc[:, 0])
    return df

# 全局配置
TIME_STEPS = 8  # 使用8周数据预测下一周
TRAINING_WEEKS = 180  # 180周训练窗口（约3.5年）
MIN_CONSENSUS = 2  # 3个模型，至少2个一致才交易
MIN_CONFIDENCE = 0.70  # 最低平均置信度70%


# FocalLoss 仅在使用深度学习模型时需要
# class FocalLoss(keras.losses.Loss):
#     """
#     Focal Loss for imbalanced classification
#     """
#     def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss'):
#         super().__init__(name='focal_loss')
#         self.gamma = gamma
#         self.alpha = alpha
#     
#     def call(self, y_true, y_pred):
#         import tensorflow as tf
#         epsilon = tf.keras.backend.epsilon()
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
#         
#         cross_entropy = -y_true * tf.math.log(y_pred)
#         weight = self.alpha * y_true * tf.pow((1 - y_pred), self.gamma)
#         weight = self.alpha * y_true * tf.pow((1 - y_pred), self.gamma)
#         
#         loss = weight * cross_entropy
#         return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
#     
#     def get_config(self):
#         return {'gamma': self.gamma, 'alpha': self.alpha}


class EnsembleQualityTrading:
    """
    Ensemble Quality Trading System
    集成5个不同类型的模型，只在高共识时交易
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.regime_detector = MarketRegimeDetector()
        self.feature_names = []
        
    def prepare_features(self, df, use_simple_regime=True):
        """
        准备81+特征（与现有系统兼容）
        use_simple_regime: True时使用简化版regime计算（快10倍）
        """
        print("  [1/6] 基础价格特征...")
        features = pd.DataFrame()
        
        # 基础价格比率（5个）
        features['close_open_ratio'] = df['close'] / df['open']
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_prev_close_ratio'] = df['close'] / df['close'].shift(1)
        features['volume_prev_volume_ratio'] = df['volumeto'] / df['volumeto'].shift(1)
        features['high_close_ratio'] = df['high'] / df['close']
        
        # RSI家族（多周期：7, 14, 21, 28）
        print("  [2/6] RSI指标家族...")
        for period in [7, 14, 21, 28]:
            features[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # MACD变种（5种组合）
        print("  [3/6] MACD变种指标...")
        for fast, slow, signal in [(12, 26, 9), (5, 13, 5), (19, 39, 9)]:
            macd_name = f'macd_{fast}_{slow}_{signal}'
            features[macd_name], features[f'{macd_name}_signal'], features[f'{macd_name}_hist'] = \
                self._calculate_macd(df['close'], fast, slow, signal)
        
        # 布林带（多周期：13, 21, 34）
        print("  [4/6] 布林带与波动率...")
        for period in [13, 21, 34]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], period)
            features[f'bb_{period}_upper'] = bb_upper
            features[f'bb_{period}_lower'] = bb_lower
            features[f'bb_{period}_width'] = (bb_upper - bb_lower) / bb_middle
            features[f'bb_{period}_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Williams %R（多周期）
        for period in [14, 21, 28]:
            features[f'williams_r_{period}'] = self._calculate_williams_r(df, period)
        
        # 动量指标
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            features[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # 波动率指标
        for period in [7, 14, 21]:
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            features[f'atr_{period}'] = self._calculate_atr(df, period)
        
        # 成交量指标
        features['volume_ma_ratio_5'] = df['volumeto'] / df['volumeto'].rolling(5).mean()
        features['volume_ma_ratio_10'] = df['volumeto'] / df['volumeto'].rolling(10).mean()
        features['volume_std_20'] = df['volumeto'].rolling(20).std() / df['volumeto'].rolling(20).mean()
        
        # 市场状态特征（regime scores）
        print("  [5/6] 市场状态特征（简化版）...")
        
        if use_simple_regime:
            # 简化版：使用MA和RSI快速计算
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean()
            rsi_14 = self._calculate_rsi(df['close'], 14)
            
            # Bull score: 价格>MA且RSI>50
            features['bull_score'] = ((df['close'] > ma_20).astype(int) * 0.5 + 
                                     (ma_20 > ma_50).astype(int) * 0.3 + 
                                     (rsi_14 > 50).astype(int) * 0.2)
            # Bear score: 相反
            features['bear_score'] = 1 - features['bull_score']
            features['regime_diff'] = features['bull_score'] - features['bear_score']
        else:
            # 完整版：使用MarketRegimeDetector（慢）
            regime_scores = []
            print(f"    计算精确regime scores (0/{len(df)})", end='', flush=True)
            for idx in range(len(df)):
                if idx < 50:  # 需要足够历史数据
                    regime_scores.append({'bull_score': 0.5, 'bear_score': 0.5})
                else:
                    hist_data = df.iloc[max(0, idx-100):idx+1]
                    regime_info = self.regime_detector.detect_regime(hist_data)
                    regime_scores.append({
                        'bull_score': regime_info['bull_score'],
                        'bear_score': regime_info['bear_score']
                    })
                if idx % 50 == 0:
                    print(f"\r    计算精确regime scores ({idx}/{len(df)})", end='', flush=True)
            print(f"\r    计算精确regime scores ({len(df)}/{len(df)}) ✓")
            
            features['bull_score'] = [s['bull_score'] for s in regime_scores]
            features['bear_score'] = [s['bear_score'] for s in regime_scores]
            features['regime_diff'] = features['bull_score'] - features['bear_score']
        
        # 移除NaN
        print("  [6/6] 清理数据...")
        features = features.bfill().fillna(0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_williams_r(self, df, period=14):
        """计算Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_atr(self, df, period=14):
        """计算ATR（真实波幅）"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def create_sequences(self, features, labels, time_steps=TIME_STEPS):
        """
        创建时间序列（用于LSTM/GRU）
        """
        X, y = [], []
        for i in range(len(features) - time_steps):
            X.append(features.iloc[i:(i + time_steps)].values)
            y.append(labels.iloc[i + time_steps])
        return np.array(X), np.array(y)
    
    # def build_bilstm_model(self, input_shape):
    #     """
    #     Model 1: Bi-LSTM（双向LSTM）
    #     """
    #     model = keras.Sequential([
    #         layers.Bidirectional(layers.LSTM(64, return_sequences=True), 
    #                             input_shape=input_shape),
    #         layers.Dropout(0.3),
    #         layers.Bidirectional(layers.LSTM(32)),
    #         layers.Dropout(0.3),
    #         layers.Dense(16, activation='relu'),
    #         layers.Dense(1, activation='sigmoid')
    #     ], name='BiLSTM')
    #     
    #     model.compile(
    #         optimizer=Adam(learning_rate=0.001),
    #         loss=FocalLoss(gamma=2.0, alpha=0.25),
    #         metrics=['accuracy']
    #     )
    #     return model
    
    # def build_gru_model(self, input_shape):
    #     """
    #     Model 2: GRU（门控循环单元）
    #     """
    #     model = keras.Sequential([
    #         layers.GRU(64, return_sequences=True, input_shape=input_shape),
    #         layers.Dropout(0.3),
    #         layers.GRU(32),
    #         layers.Dropout(0.3),
    #         layers.Dense(16, activation='relu'),
    #         layers.Dense(1, activation='sigmoid')
    #     ], name='GRU')
    #     
    #     model.compile(
    #         optimizer=Adam(learning_rate=0.001),
    #         loss=FocalLoss(gamma=2.0, alpha=0.25),
    #         metrics=['accuracy']
    #     )
    #     return model
    
    def train_all_models(self, X_train_seq, y_train, X_train_flat, X_val_seq, y_val, X_val_flat):
        """
        训练所有模型（TensorFlow可用时5个，否则3个）
        """
        print("=" * 80)
        total_models = 5 if TF_AVAILABLE else 3
        print(f"开始训练 Ensemble 系统 ({total_models}个模型)")
        print("=" * 80)
        
        val_accuracies = []
        
        # Model 1: Bi-LSTM (需要TensorFlow)
        if TF_AVAILABLE:
            print("\n[1/5] 训练 Bi-LSTM 模型...")
            self.models['bilstm'] = self.build_bilstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history_bilstm = self.models['bilstm'].fit(
                X_train_seq, y_train,
                validation_data=(X_val_seq, y_val),
                epochs=100, batch_size=16,
                callbacks=[early_stop],
                verbose=0
            )
            val_acc_bilstm = max(history_bilstm.history['val_accuracy'])
            val_accuracies.append(val_acc_bilstm)
            print(f"✓ Bi-LSTM 验证准确率: {val_acc_bilstm:.1%}")
        else:
            print("\n[跳过] Bi-LSTM (TensorFlow不可用)")
        
        # Model 2: GRU (需要TensorFlow)
        if TF_AVAILABLE:
            print("\n[2/5] 训练 GRU 模型...")
            self.models['gru'] = self.build_gru_model((X_train_seq.shape[1], X_train_seq.shape[2]))
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history_gru = self.models['gru'].fit(
                X_train_seq, y_train,
                validation_data=(X_val_seq, y_val),
                epochs=100, batch_size=16,
                callbacks=[early_stop],
                verbose=0
            )
            val_acc_gru = max(history_gru.history['val_accuracy'])
            val_accuracies.append(val_acc_gru)
            print(f"✓ GRU 验证准确率: {val_acc_gru:.1%}")
        else:
            print("\n[跳过] GRU (TensorFlow不可用)")
        
        # Model 3: LightGBM
        model_num = 3 if TF_AVAILABLE else 1
        print(f"\n[{model_num}/{total_models}] 训练 LightGBM 模型...")
        self.models['lightgbm'] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.models['lightgbm'].fit(X_train_flat, y_train)
        val_acc_lgb = self.models['lightgbm'].score(X_val_flat, y_val)
        val_accuracies.append(val_acc_lgb)
        print(f"✓ LightGBM 验证准确率: {val_acc_lgb:.1%}")
        
        # Model 4: XGBoost
        model_num = 4 if TF_AVAILABLE else 2
        print(f"\n[{model_num}/{total_models}] 训练 XGBoost 模型...")
        self.models['xgboost'] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.models['xgboost'].fit(X_train_flat, y_train, verbose=False)
        val_acc_xgb = self.models['xgboost'].score(X_val_flat, y_val)
        val_accuracies.append(val_acc_xgb)
        print(f"✓ XGBoost 验证准确率: {val_acc_xgb:.1%}")
        
        # Model 5: Random Forest
        model_num = 5 if TF_AVAILABLE else 3
        print(f"\n[{model_num}/{total_models}] 训练 Random Forest 模型...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train_flat, y_train)
        val_acc_rf = self.models['random_forest'].score(X_val_flat, y_val)
        val_accuracies.append(val_acc_rf)
        print(f"✓ Random Forest 验证准确率: {val_acc_rf:.1%}")
        
        print("\n" + "=" * 80)
        print("所有模型训练完成！")
        print(f"平均验证准确率: {np.mean(val_accuracies):.1%}")
        print("=" * 80)
    
    def predict_with_consensus(self, X_seq, X_flat, min_consensus=MIN_CONSENSUS, min_confidence=MIN_CONFIDENCE):
        """
        使用共识机制进行预测
        只有当足够多模型一致 + 置信度足够高时才交易
        """
        predictions = {}
        confidences = {}
        
        # Bi-LSTM (如果可用)
        if 'bilstm' in self.models:
            pred_bilstm = self.models['bilstm'].predict(X_seq, verbose=0)
            predictions['bilstm'] = (pred_bilstm > 0.5).astype(int).flatten()[0]
            confidences['bilstm'] = pred_bilstm[0][0] if predictions['bilstm'] == 1 else (1 - pred_bilstm[0][0])
        
        # GRU (如果可用)
        if 'gru' in self.models:
            pred_gru = self.models['gru'].predict(X_seq, verbose=0)
            predictions['gru'] = (pred_gru > 0.5).astype(int).flatten()[0]
            confidences['gru'] = pred_gru[0][0] if predictions['gru'] == 1 else (1 - pred_gru[0][0])
        
        # LightGBM
        pred_lgb = self.models['lightgbm'].predict_proba(X_flat)
        predictions['lightgbm'] = self.models['lightgbm'].predict(X_flat)[0]
        confidences['lightgbm'] = pred_lgb[0][predictions['lightgbm']]
        
        # XGBoost
        pred_xgb = self.models['xgboost'].predict_proba(X_flat)
        predictions['xgboost'] = self.models['xgboost'].predict(X_flat)[0]
        confidences['xgboost'] = pred_xgb[0][predictions['xgboost']]
        
        # Random Forest
        pred_rf = self.models['random_forest'].predict_proba(X_flat)
        predictions['random_forest'] = self.models['random_forest'].predict(X_flat)[0]
        confidences['random_forest'] = pred_rf[0][predictions['random_forest']]
        
        # 计算共识
        pred_values = list(predictions.values())
        up_votes = sum(pred_values)
        down_votes = len(pred_values) - up_votes
        avg_confidence = np.mean(list(confidences.values()))
        
        # 动态调整共识要求（如果只有3个模型）
        total_models = len(predictions)
        adjusted_min_consensus = min_consensus if total_models >= 5 else max(2, int(total_models * 0.67))
        
        # 决策逻辑
        should_trade = False
        final_prediction = None
        consensus_level = max(up_votes, down_votes)
        
        if consensus_level >= adjusted_min_consensus and avg_confidence >= min_confidence:
            should_trade = True
            final_prediction = 1 if up_votes > down_votes else 0
        
        return {
            'should_trade': should_trade,
            'prediction': final_prediction,
            'up_votes': up_votes,
            'down_votes': down_votes,
            'consensus_level': consensus_level,
            'avg_confidence': avg_confidence,
            'total_models': total_models,
            'adjusted_min_consensus': adjusted_min_consensus,
            'individual_predictions': predictions,
            'individual_confidences': confidences
        }


def main():
    """
    主函数：演示系统训练
    """
    print("Ensemble Quality Trading System - 训练演示")
    print("=" * 80)
    
    # 加载数据
    print("\n加载比特币数据...")
    df = get_bitcoin_data()
    print(f"数据范围: {df['date'].iloc[0]} 到 {df['date'].iloc[-1]}")
    print(f"总周数: {len(df)}")
    
    # 初始化系统
    system = EnsembleQualityTrading()
    
    # 准备特征
    print("\n准备特征...")
    features = system.prepare_features(df)
    labels = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 使用最近180周作为训练集演示
    train_end_idx = len(df) - 20
    train_start_idx = train_end_idx - TRAINING_WEEKS
    
    X_train = features.iloc[train_start_idx:train_end_idx]
    y_train = labels.iloc[train_start_idx:train_end_idx]
    
    # 分割验证集
    val_split = int(len(X_train) * 0.8)
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]
    X_train = X_train.iloc[:val_split]
    y_train = y_train.iloc[:val_split]
    
    print(f"训练集: {len(X_train)} 周")
    print(f"验证集: {len(X_val)} 周")
    print(f"特征数: {len(features.columns)}")
    
    # 标准化
    print("\n标准化特征...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    # 为不同模型准备数据
    # 序列数据（LSTM/GRU）
    X_train_seq, y_train_seq = system.create_sequences(X_train_scaled, y_train)
    X_val_seq, y_val_seq = system.create_sequences(X_val_scaled, y_val)
    
    # 扁平数据（LightGBM/XGBoost/RF）- 使用最近一周的数据
    X_train_flat = X_train_scaled.iloc[TIME_STEPS:].values
    X_val_flat = X_val_scaled.iloc[TIME_STEPS:].values
    
    print(f"\n序列数据形状: {X_train_seq.shape}")
    print(f"扁平数据形状: {X_train_flat.shape}")
    
    # 训练所有模型
    system.train_all_models(
        X_train_seq, y_train_seq, X_train_flat,
        X_val_seq, y_val_seq, X_val_flat
    )
    
    print("\n✓ 系统训练完成！")
    print("\n下一步：运行回测测试共识机制效果")


if __name__ == '__main__':
    main()

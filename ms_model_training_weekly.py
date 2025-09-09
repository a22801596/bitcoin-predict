# -*- coding: utf-8 -*-
"""
Train an LSTM on weekly BTC data and predict next week's close.

Run order:
    python ms_collect_his_data.py   # 準備 hourly & weekly CSV
    python ms_model_training_weekly.py

Outputs:
    - Adds a new row with 'predicted_close_next_week' into bitcoin_weekly.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import pandas_ta as ta

# 路徑（與收集腳本一致：預設寫到當前 repo 資料夾）
OUTPUT_DIR = "."
WEEKLY_CSV = os.path.join(OUTPUT_DIR, "bitcoin_weekly.csv")

if not os.path.exists(WEEKLY_CSV):
    raise FileNotFoundError(f"{WEEKLY_CSV} not found. Run ms_collect_his_data.py first.")

# 讀取 weekly 資料
df = pd.read_csv(WEEKLY_CSV, index_col="timestamp", parse_dates=True).sort_index()

# 技術指標（以 weekly close 為基礎）
df["SMA_8"]  = ta.sma(df["close"], length=8)
df["EMA_8"]  = ta.ema(df["close"], length=8)

bb = ta.bbands(df["close"], length=8)
df["BBU"] = bb["BBU_8_2.0"]
df["BBM"] = bb["BBM_8_2.0"]
df["BBL"] = bb["BBL_8_2.0"]

macd = ta.macd(df["close"], fast=6, slow=13, signal=5)
df["MACD"]  = macd["MACD_6_13_5"]
df["MACDs"] = macd["MACDs_6_13_5"]

# 以每週 high/low 推算 Fibonacci retracement（週資料）
hi = df["high"]; lo = df["low"]; diff = (hi - lo).replace(0, np.nan)
df["fib_0.236"] = hi - diff * 0.236
df["fib_0.382"] = hi - diff * 0.382
df["fib_0.5"]   = hi - diff * 0.5
df["fib_0.618"] = hi - diff * 0.618
df["fib_0.786"] = hi - diff * 0.786

# K 線幾何特徵（週）
df["candle_body"]  = (df["close"] - df["open"]).abs()
df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

# 清理 NaN/Inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 特徵欄位
features = [
    "SMA_8","EMA_8","BBU","BBM","BBL",
    "MACD","MACDs",
    "fib_0.236","fib_0.382","fib_0.5","fib_0.618","fib_0.786",
    "candle_body","upper_shadow","lower_shadow"
]

# 標準化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df[features])

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(df["close"].values.reshape(-1, 1))

# 建序列（以 time_steps 週預測下一週）
time_steps = 8
def make_seq(X, y, steps):
    xs, ys = [], []
    for i in range(steps, len(X)):
        xs.append(X[i-steps:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

X_seq, y_seq = make_seq(X_scaled, y_scaled, time_steps)

# 切分資料（不打亂時間順序）
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=False
)

# 建 LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="rmsprop", loss="mean_squared_error")

# 訓練
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=60, batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[es], verbose=1
)

# 用最後 time_steps 週做「下一週」預測
last_window_X = X_scaled[-time_steps:]
last_window_X = last_window_X.reshape(1, last_window_X.shape[0], last_window_X.shape[1])
pred_scaled = model.predict(last_window_X)
pred_close = scaler_y.inverse_transform(pred_scaled)[0][0]

# 把預測結果加到 df（時間為最後一列下一週的 label 起始）
next_week_idx = df.index[-1] + pd.Timedelta(weeks=1)
df.loc[next_week_idx, "predicted_close_next_week"] = pred_close

# 儲存回 weekly CSV（不覆蓋原 OHLC；新欄位會在最後）
df.to_csv(WEEKLY_CSV)
print(f"Predicted next-week close @ {next_week_idx.date()}: {pred_close:.2f}")
print(f"Saved back to {WEEKLY_CSV}")

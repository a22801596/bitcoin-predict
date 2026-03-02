# -*- coding: utf-8 -*-
"""
Collect hourly BTC/USD from CryptoCompare and maintain a growing CSV.
Also produces a weekly-aggregated CSV for model training.

Usage (local/Colab):
    pip install -r requirements.txt
    python ms_collect_his_data.py

Default output directory is the current repo folder (OUTPUT_DIR=".").
If you want persistence on Colab Google Drive:
    from google.colab import drive; drive.mount('/content/drive')
    # then change OUTPUT_DIR below, e.g.:
    OUTPUT_DIR = "/content/drive/My Drive/bitcoin-project"

Optionally set an API key for higher rate limits:
    export CRYPTOCOMPARE_API_KEY=xxxx
"""

import os
import time
import random
import requests
import pandas as pd
from datetime import datetime, timezone

# ===================== Config =====================
OUTPUT_DIR = "." 
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOURLY_CSV = os.path.join(OUTPUT_DIR, "bitcoin_hourly.csv")
WEEKLY_CSV = os.path.join(OUTPUT_DIR, "bitcoin_weekly.csv")

FSYM = "BTC"
TSYM = "USD"
API = "https://min-api.cryptocompare.com/data/v2/histohour"
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
HEADERS = {"authorization": f"Apikey {API_KEY}"} if API_KEY else {}

# 初次抓取設定
INITIAL_START = "2020-04-09 03:00:00"  # 起始時間（UTC）
BATCH_LIMIT = 2000                     # CryptoCompare histohour 一次最多 2000（實務 2000-1）
SLEEP_SEC = 0.2                        # 基礎節流
# ==================================================


def _utc_ts(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _now_utc_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        df.index = idx.tz_localize("UTC")
    else:
        df.index = idx.tz_convert("UTC")
    return df


def fetch_hist_hour_chunk(to_ts: int, limit: int = 1999, retries: int = 3) -> pd.DataFrame:
    """
    抓一個批次（結束時間為 to_ts，含），回傳 DataFrame（UTC DatetimeIndex）。
    內含簡單退避重試與 429 處理。
    """
    params = dict(fsym=FSYM, tsym=TSYM, limit=limit, toTs=to_ts)
    for i in range(retries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                backoff = SLEEP_SEC * (2 ** i) + random.uniform(0, 0.3)
                print(f"[RateLimit] 429. Backoff {backoff:.2f}s ...")
                time.sleep(backoff)
                continue
            r.raise_for_status()
            payload = r.json()
            data = payload.get("Data", {}).get("Data", [])
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("timestamp").sort_index()
            cols = ["open", "high", "low", "close", "volumefrom", "volumeto"]
            return df[cols].astype("float64")
        except requests.RequestException as e:
            backoff = SLEEP_SEC * (2 ** i) + random.uniform(0, 0.3)
            print(f"[FetchError] {e}. Retry in {backoff:.2f}s ...")
            time.sleep(backoff)
    print("[FetchError] Failed after retries.")
    return pd.DataFrame()


def initial_full_fetch(start_dt: str, hourly_csv: str):
    """從 start_dt 以 2000 小時一批往 '現在' 抓，直到補滿。"""
    os.makedirs(os.path.dirname(hourly_csv) or ".", exist_ok=True)

    start_ts = _utc_ts(start_dt)
    end_ts = _now_utc_ts()
    all_chunks = []
    cursor = start_ts + 3600 * BATCH_LIMIT  # 第一個 toTs

    print(f"[Init] from={datetime.fromtimestamp(start_ts, timezone.utc)} to now (UTC)")

    while cursor <= end_ts + 3600:  # 稍微多抓一點，API會自動截斷
        df_chunk = fetch_hist_hour_chunk(to_ts=cursor, limit=BATCH_LIMIT - 1)
        if df_chunk.empty:
            print("[Init] Empty chunk, stop.")
            break
        all_chunks.append(df_chunk)
        last_ts = int(df_chunk.index[-1].timestamp())
        print(f"[Init] fetched up to {df_chunk.index[-1]}  rows={len(df_chunk)}")
        cursor = last_ts + 3600 * BATCH_LIMIT  # 前進下一批
        time.sleep(SLEEP_SEC)

    if not all_chunks:
        print("[Init] No data fetched.")
        return

    df = pd.concat(all_chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(hourly_csv, index=True)
    print(f"[Init] Saved hourly → {hourly_csv}  rows={len(df)}")


def incremental_update(hourly_csv: str):
    """從既有 CSV 的最後一筆往 '現在' 追加。"""
    if not os.path.exists(hourly_csv):
        print("[Inc] No previous data found. Run initial_full_fetch() first.")
        return

    df_exist = pd.read_csv(hourly_csv, index_col="timestamp", parse_dates=True)
    df_exist = _ensure_utc_index(df_exist)
    df_exist = df_exist[~df_exist.index.duplicated(keep="first")].sort_index()

    last_idx = df_exist.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    last_ts = int(last_idx.timestamp())
    now_ts = _now_utc_ts()

    print(f"[Inc] from last={last_idx} -> now(UTC)")

    all_chunks = []
    cursor = min(last_ts + 3600 * BATCH_LIMIT, now_ts)

    while cursor <= now_ts:
        df_chunk = fetch_hist_hour_chunk(to_ts=cursor, limit=BATCH_LIMIT - 1)
        if df_chunk.empty:
            print("[Inc] Empty chunk, stop.")
            break
        # 過濾掉已存在的 index
        df_chunk = df_chunk[~df_chunk.index.isin(df_exist.index)]
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
            print(f"[Inc] fetched up to {df_chunk.index[-1]}  new_rows={len(df_chunk)}")
        # 前進
        last_ts = int(df_chunk.index[-1].timestamp())
        cursor = min(last_ts + 3600 * BATCH_LIMIT, now_ts)
        time.sleep(SLEEP_SEC)

    if all_chunks:
        df_new = pd.concat(all_chunks).sort_index()
        df_all = pd.concat([df_exist, df_new]).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep="first")]
        df_all.to_csv(hourly_csv, index=True)
        print(f"[Inc] Appended rows={len(df_new)}  Total={len(df_all)} → {hourly_csv}")
    else:
        print("[Inc] No new data.")


def make_weekly_from_hourly(hourly_csv: str, weekly_csv: str):
    """由小時資料聚合出 weekly OHLC 與必要欄位（週一為起始）。"""
    if not os.path.exists(hourly_csv):
        print("[Weekly] Hourly CSV not found.")
        return
    df_h = pd.read_csv(hourly_csv, index_col="timestamp", parse_dates=True)
    df_h = _ensure_utc_index(df_h).sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volumefrom": "sum",
        "volumeto": "sum",
    }
    df_w = df_h.resample("W-MON", label="left", closed="left").agg(agg).dropna()
    df_w.to_csv(weekly_csv, index=True)
    print(f"[Weekly] Saved → {weekly_csv}  rows={len(df_w)}")


if __name__ == "__main__":
    # 第一次使用請先跑初始化；之後就只需要跑 incremental_update
    if not os.path.exists(HOURLY_CSV):
        initial_full_fetch(INITIAL_START, HOURLY_CSV)
    else:
        incremental_update(HOURLY_CSV)

    make_weekly_from_hourly(HOURLY_CSV, WEEKLY_CSV)
    print(f"[Done] Outputs at: {os.path.abspath(OUTPUT_DIR)}")

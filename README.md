# Bitcoin Price Prediction (Weekly LSTM)

A clean, reproducible pipeline:
- Collect **hourly** BTC/USD from CryptoCompare (initial load + incremental updates)
- Aggregate to **weekly** OHLC
- Engineer features (SMA/EMA, Bollinger Bands, MACD, Fibonacci levels, candle geometry)
- Train an **LSTM** on weekly sequences to predict **next-week close**

## Repo Structure

ms_collect_his_data.py # hourly fetch (init + incremental) + weekly aggregation
ms_model_training_weekly.py # LSTM training & next-week prediction (uses weekly CSV)
requirements.txt

## Quickstart (local or Colab)
```bash
pip install -r requirements.txt
python ms_collect_his_data.py
python ms_model_training_weekly.py

After the first run, you'll find:

bitcoin_hourly.csv — hourly OHLCV (UTC index)

bitcoin_weekly.csv — weekly OHLC with volume sums (UTC index)

The training script appends predicted_close_next_week to the next week's row.

Colab Tips (save to Google Drive)

By default, data is saved in the repo folder (OUTPUT_DIR=".").

If you want persistence across sessions:

  from google.colab import drive
  drive.mount('/content/drive')

Then in ms_collect_his_data.py, set:
  OUTPUT_DIR = "/content/drive/My Drive/bitcoin-project"


Design Choices

Weekly horizon avoids mixing 168 hourly points per week and yields a smoother series.

We deduplicate by timestamp and resample with W-MON (weeks start Monday).

CryptoCompare API batching with basic backoff; optional API key via CRYPTOCOMPARE_API_KEY.

Troubleshooting

429 rate limit: the collector auto-backs off and retries.

Timezone: indexes are stored in UTC; on reload we normalize to UTC.

Colab restarts: if saving to the repo folder, outputs reset; save to Drive for persistence.

This project is for educational/demo purposes only and not financial advice.

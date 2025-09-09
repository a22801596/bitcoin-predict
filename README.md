# Bitcoin Price Prediction (Weekly LSTM)

A clean, reproducible pipeline:
- Collect **hourly** BTC/USD from CryptoCompare (initial load + incremental updates)
- Aggregate to **weekly** OHLC
- Engineer features (SMA/EMA, Bollinger Bands, MACD, Fibonacci levels, candle geometry)
- Train an **LSTM** on weekly sequences to predict **next-week close**

## Repo Structure

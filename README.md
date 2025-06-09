# Bitcoin Price Prediction using LSTM (Weekly Forecast)

This project demonstrates a complete AI pipeline for Bitcoin price forecasting based on historical data, technical indicators, and deep learning models. It collects hourly price data using CryptoCompare API, processes and aggregates the data into weekly format, applies technical feature engineering, and trains an LSTM model to predict future weekly closing prices.

## 📌 Features

- ✅ Hourly data collection from CryptoCompare API  
- ✅ Data cleaning and missing timestamp handling  
- ✅ Weekly high/low aggregation and Fibonacci retracement levels  
- ✅ Technical indicators (SMA, EMA, Bollinger Bands, MACD)  
- ✅ LSTM model training with weekly forecasting  
- ✅ Model evaluation with training/validation loss  
- ✅ Prediction saved directly to time-series dataset for analysis  

## 📁 Project Structure

bitcoin-predictor/
├── ms_collect_his_data.py # Data collection and weekly aggregation
├── ms_model_training_weekly.py # LSTM model training and prediction
├── bitcoin_data_and_prediction_weekly.csv # Weekly dataset
└── README.md # Project Documentation

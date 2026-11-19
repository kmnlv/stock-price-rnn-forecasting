# Stock Price Forecasting with LSTM and GRU Recurrent Networks

> Course: Deep Learning · Semester 2

## Overview

Time-series forecasting of daily stock prices with recurrent neural networks. The main notebook benchmarks LSTM and GRU on Amazon prices using 30-day and 50-day input windows; an earlier script applies the same idea to IBM data.

## Repository structure

- `notebooks/stock-price-rnn.ipynb` - AMZN experiment with four trained models.
- `scripts/stock-price-rnn.py` - earlier IBM variant of the recurrent forecaster.

## Method

- `MinMaxScaler` normalization and sliding windows of 30 or 50 days over the `Low` price.
- Four stacked recurrent layers with 50 units and `Dropout(0.2)`, followed by a linear `Dense(1)`.
- Adam, MSE plus MAE monitoring, up to 100 epochs, batch size 16, `EarlyStopping(patience=20)`.
- Train on data up to 2016, evaluate on 2017 data; RMSE reported per configuration.

## Results

Printed RMSE values in the notebook: **GRU-50 = 22.13**, **GRU-30 = 49.00**, **LSTM-50 = 55.73**, **LSTM-30 = 58.82**. The longer window and the GRU cell give the best result. The notebook's final markdown summary contains older RMSE figures that do not match the executed cells; the values above are the ones produced by the code.

## Data

The original CSV comes from a public Kaggle stock-price dataset (`AMZN_2006-01-01_to_2018-01-01.csv`, `IBM_...`). The compressed third-party dataset is **not** redistributed here. Download it and place the AMZN file in `data/` (or update `path_file` in the notebook) before running.

## Requirements

`tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `jupyter`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/stock-price-rnn.ipynb
```

---
Author: Karla Maria Monroy Alvarez

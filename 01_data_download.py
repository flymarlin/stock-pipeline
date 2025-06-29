%%writefile 01_data_download.py
import os, pathlib, time, json, requests, pandas as pd, yfinance as yf, 00_settings as cfg
from datetime import date, timedelta

# --- フォルダ作成 -----------------------------------------------------------
pathlib.Path(cfg.DATA_DIR, "fundamentals").mkdir(parents=True, exist_ok=True)
pathlib.Path(cfg.DATA_DIR, "sentiment").mkdir(parents=True, exist_ok=True)

tickers = cfg.UNIVERSE_US

# --- 価格・出来高 -----------------------------------------------------------
print("▶ Price / Volume")
prices = yf.download(tickers, start=cfg.BACKTEST_START, end=cfg.BACKTEST_END, group_by="column")
prices["Adj Close"].to_csv(f"{cfg.DATA_DIR}/close.csv")
prices["Volume"].to_csv(f"{cfg.DATA_DIR}/volume.csv")

# --- ファンダメンタル -------------------------------------------------------
print("▶ Fundamentals (FMP)")
for t in tickers:
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{t}?period=quarter&limit=1&apikey={cfg.FMP_API_KEY}"
    open(f"{cfg.DATA_DIR}/fundamentals/{t}.json", "w").write(requests.get(url, timeout=30).text)
    time.sleep(0.3)

# --- ニュース & センチメント ----------------------------------------------
print("▶ News (Marketaux Basic)")
token = cfg.MARKETAUX_KEY
to_ = date.today(); frm_ = to_ - timedelta(days=3)

for t in tickers:
    url = (
        "https://api.marketaux.com/v1/news/all"
        f"?symbols={t}&filter_entities=true&language=en"
        f"&from={frm_}&to={to_}&limit=20&api_token={token}"
    )
    open(f"{cfg.DATA_DIR}/sentiment/{t}.json", "w").write(requests.get(url, timeout=30).text)
    time.sleep(0.05)
print("✅ Data download complete")

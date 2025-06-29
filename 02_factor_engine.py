%%writefile 02_factor_engine.py
import os, json, pathlib, numpy as np, pandas as pd, vectorbt as vbt, 00_settings as cfg

pathlib.Path(cfg.FACTOR_DIR).mkdir(exist_ok=True)
close  = pd.read_csv(f"{cfg.DATA_DIR}/close.csv",  index_col=0, parse_dates=True)
volume = pd.read_csv(f"{cfg.DATA_DIR}/volume.csv", index_col=0, parse_dates=True)

# --- テクニカル -------------------------------------------------------------
fast   = vbt.MA.run(close, window=50).ma
slow   = vbt.MA.run(close, window=200).ma
rsi    = vbt.RSI.run(close, window=14).rsi
vol_trend = volume.rolling(30).mean()

tech_score = (
    (fast > slow).astype(int) +
    (rsi.between(45,70)).astype(int) +
    (volume > vol_trend).astype(int)
)

# --- ファンダメンタル -------------------------------------------------------
fund_cols = ["roe","revenueGrowth","evToEbitda"]
fund_df   = pd.DataFrame(index=tech_score.columns, columns=fund_cols, dtype=float)
for t in tech_score.columns:
    f = pathlib.Path(cfg.DATA_DIR,"fundamentals",f"{t}.json")
    if f.exists():
        j = json.loads(open(f).read())
        if j:
            row = j[0]
            fund_df.loc[t] = [row.get(c) for c in fund_cols]

fund_z = fund_df.apply(lambda x: (x - x.mean())/x.std(), axis=0)
fund_score = (fund_z > 0).astype(int)

# --- ニュースセンチメント ---------------------------------------------------
news_score = pd.Series(0, index=tech_score.columns, dtype=int)
for t in tech_score.columns:
    f = pathlib.Path(cfg.DATA_DIR,"sentiment",f"{t}.json")
    if f.exists():
        arts = json.loads(open(f).read()).get("data", [])
        pos = sum(1 for a in arts if any(e.get("sentiment")=="Bullish"  for e in a["entities"]))
        neg = sum(1 for a in arts if any(e.get("sentiment")=="Bearish" for e in a["entities"]))
        news_score[t] = pos - neg

# --- 合計スコア (週次) ------------------------------------------------------
total = tech_score.resample("W-FRI").last()
latest_fund = fund_score.sum(axis=1)
total = total.add(latest_fund, axis=1).add(news_score, axis=1)
total.to_parquet(f"{cfg.FACTOR_DIR}/scores.parquet")
print("✅ Factor engine complete")

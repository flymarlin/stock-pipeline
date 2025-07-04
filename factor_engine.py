# factor_engine.py  ── 完全版
import os, json, pathlib, numpy as np, pandas as pd, vectorbt as vbt, settings as cfg

# ------------------------------------------------------------------------------
# 0. フォルダ準備
pathlib.Path(cfg.FACTOR_DIR).mkdir(exist_ok=True)

# ------------------------------------------------------------------------------
# 1. 価格・出来高データ読み込み
close  = pd.read_csv(f"{cfg.DATA_DIR}/close.csv",  index_col=0, parse_dates=True)
volume = pd.read_csv(f"{cfg.DATA_DIR}/volume.csv", index_col=0, parse_dates=True)

# ------------------------------------------------------------------------------
# 2. テクニカル・スコア
# ① ゴールデンクロス
fast   = close.rolling(50).mean()
slow   = close.rolling(200).mean()
golden = (fast > slow).astype(int)

# ② RSI バンド（45〜70 が 1）
rsi_full = vbt.RSI.run(close, window=14).rsi
if isinstance(rsi_full.columns, pd.MultiIndex):
    rsi_full.columns = rsi_full.columns.get_level_values(0)
rsi_band = ((rsi_full >= 45) & (rsi_full <= 70)).astype(int)

# ③ 出来高トレンド（直近 30 日平均より大きければ 1）
vol_trend = volume.rolling(30).mean()
vol_up = (volume > vol_trend).astype(int)

# ④ 合算
tech_score = (golden + rsi_band + vol_up).fillna(0)

# ------------------------------------------------------------------------------
# 3. ファンダメンタル・スコア
fund_cols = ["roe", "revenueGrowth", "evToEbitda", "freeCashFlowMargin"]
fund_df   = pd.DataFrame(index=tech_score.columns, columns=fund_cols, dtype=float)

for tkr in tech_score.columns:
    f = pathlib.Path(cfg.DATA_DIR, "fundamentals", f"{tkr}.json")
    if not f.exists():
        continue
    data = json.load(open(f))
    if isinstance(data, list) and data:
        row = data[0]
    elif isinstance(data, dict):
        row = data
    else:
        continue
    fund_df.loc[tkr] = [row.get(c) for c in fund_cols]

def _scale(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

scaled = fund_df.apply(_scale, axis=0)
scaled["evToEbitda"] = 1 - scaled["evToEbitda"]        # 低いほど良いので反転
fund_score = scaled.mean(axis=1).fillna(0)             # 0〜1 の連続値

# ------------------------------------------------------------------------------
# 4. ニュース・センチメント・スコア
news_score = pd.Series(0.0, index=tech_score.columns)
for tkr in tech_score.columns:
    f = pathlib.Path(cfg.DATA_DIR, "sentiment", f"{tkr}.json")
    if f.exists():
        arts = json.load(open(f)).get("data", [])
        pos = sum(any(e.get("sentiment") == "Bullish"  for e in a["entities"]) for a in arts)
        neg = sum(any(e.get("sentiment") == "Bearish" for e in a["entities"]) for a in arts)
        news_score[tkr] = pos - neg                     # プラス = 強気

# ------------------------------------------------------------------------------
# 5. 合計スコア（週次）
total = tech_score.resample("W-FRI").last()

#   価格データに存在するティッカーだけ残す（“14” 列などを除去）
valid_cols = close.columns
total = total.loc[:, total.columns.intersection(valid_cols)]

#   ファンダ & ニュースを加算
total = total.add(fund_score, axis=1).add(news_score, axis=1)

#   ★ 欠損は 0 扱いに統一
total = total.fillna(0)

# ------------------------------------------------------------------------------
# 6. 保存
total.to_parquet(f"{cfg.FACTOR_DIR}/scores.parquet")
print("✅ Factor engine complete")


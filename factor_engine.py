# factor_engine.py  ── 完全版 (ML統合)
import os, json, pathlib, numpy as np, pandas as pd, vectorbt as vbt, settings as cfg
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
# 5. ML予測スコア
def generate_ml_features(close, volume):
    """ML用の特徴量を生成 (train_ml_model.pyと同じ処理)"""
    features = pd.DataFrame(index=close.index)
    
    for ticker in close.columns:
        if ticker not in volume.columns:
            continue
            
        price = close[ticker]
        vol = volume[ticker]
        
        # Technical indicators
        sma_5 = price.rolling(5).mean()
        sma_10 = price.rolling(10).mean()
        sma_20 = price.rolling(20).mean()
        sma_50 = price.rolling(50).mean()
        sma_200 = price.rolling(200).mean()
        
        # Price features
        features[f'{ticker}_price_ratio_5_10'] = sma_5 / sma_10
        features[f'{ticker}_price_ratio_10_20'] = sma_10 / sma_20
        features[f'{ticker}_price_ratio_20_50'] = sma_20 / sma_50
        features[f'{ticker}_price_ratio_50_200'] = sma_50 / sma_200
        
        # RSI
        rsi = vbt.RSI.run(price, window=14).rsi
        if isinstance(rsi, pd.DataFrame):
            rsi = rsi.iloc[:, 0]
        features[f'{ticker}_rsi'] = rsi
        
        # Volume features
        vol_sma_10 = vol.rolling(10).mean()
        vol_sma_30 = vol.rolling(30).mean()
        features[f'{ticker}_volume_ratio'] = vol / vol_sma_30
        features[f'{ticker}_volume_trend'] = vol_sma_10 / vol_sma_30
        
        # Volatility
        returns = price.pct_change()
        features[f'{ticker}_volatility_10'] = returns.rolling(10).std()
        features[f'{ticker}_volatility_30'] = returns.rolling(30).std()
        
        # Momentum
        features[f'{ticker}_momentum_5'] = price / price.shift(5)
        features[f'{ticker}_momentum_10'] = price / price.shift(10)
        features[f'{ticker}_momentum_20'] = price / price.shift(20)
        
    return features

def get_ml_predictions(close, volume):
    """ML予測を取得"""
    model_path = f"{cfg.DATA_DIR}/ml_model.joblib"
    feature_names_path = f"{cfg.DATA_DIR}/feature_names.json"
    
    if not os.path.exists(model_path) or not os.path.exists(feature_names_path):
        print("Warning: ML model not found. Returning zero scores.")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    
    # Load model and feature names
    model = joblib.load(model_path)
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    # Generate features
    features = generate_ml_features(close, volume)
    
    # ML predictions for each ticker
    ml_scores = pd.DataFrame(index=close.index, columns=close.columns)
    
    for ticker in close.columns:
        ticker_feature_cols = [col for col in feature_names if col.startswith(ticker + '_')]
        
        if not ticker_feature_cols:
            ml_scores[ticker] = 0.0
            continue
            
        ticker_features = features[ticker_feature_cols]
        
        # Handle missing columns
        missing_cols = set(ticker_feature_cols) - set(ticker_features.columns)
        for col in missing_cols:
            ticker_features[col] = 0.0
        
        # Reorder columns to match training
        ticker_features = ticker_features[ticker_feature_cols]
        
        # Make predictions
        valid_mask = ~ticker_features.isna().any(axis=1)
        predictions = np.zeros(len(ticker_features))
        
        if valid_mask.sum() > 0:
            X_valid = ticker_features[valid_mask].fillna(0)
            # Predict probability of price increase
            proba = model.predict_proba(X_valid)[:, 1]
            predictions[valid_mask] = proba
        
        ml_scores[ticker] = predictions
    
    return ml_scores.fillna(0)

# ML予測を取得
ml_predictions = get_ml_predictions(close, volume)

# ------------------------------------------------------------------------------
# 6. 合計スコア（週次）
total = tech_score.resample("W-FRI").last()

#   価格データに存在するティッカーだけ残す（“14” 列などを除去）
valid_cols = close.columns
total = total.loc[:, total.columns.intersection(valid_cols)]

#   ファンダ & ニュースを加算
total = total.add(fund_score, axis=1).add(news_score, axis=1)

#   ML予測スコアを加算 (週次にリサンプル)
ml_weekly = ml_predictions.resample("W-FRI").last()
ml_weekly = ml_weekly.loc[:, ml_weekly.columns.intersection(valid_cols)]
# ML予測に重みを付ける (0-1の確率を2倍して重要度を上げる)
total = total.add(ml_weekly * 2, axis=1)

#   ★ 欠損は 0 扱いに統一
total = total.fillna(0)

# ------------------------------------------------------------------------------
# 7. 保存
total.to_parquet(f"{cfg.FACTOR_DIR}/scores.parquet")
print("✅ Factor engine complete (with ML integration)")


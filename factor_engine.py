# factor_engine.py  ── 完全版（ML統合）
import os, json, pathlib, numpy as np, pandas as pd, vectorbt as vbt, settings as cfg
import joblib
from sklearn.exceptions import NotFittedError

# ------------------------------------------------------------------------------
# 0. フォルダ準備
pathlib.Path(cfg.FACTOR_DIR).mkdir(exist_ok=True)

# ------------------------------------------------------------------------------
# 1. 価格・出来高データ読み込み
close  = pd.read_csv(f"{cfg.DATA_DIR}/close.csv",  index_col=0, parse_dates=True)
volume = pd.read_csv(f"{cfg.DATA_DIR}/volume.csv", index_col=0, parse_dates=True)

# ------------------------------------------------------------------------------
# 1.5. ML モデル読み込み
def load_ml_model():
    """ML モデルと特徴量名を読み込み"""
    try:
        model_path = f"{cfg.DATA_DIR}/ml_model.joblib"
        feature_names_path = f"{cfg.DATA_DIR}/feature_names.json"
        
        if not os.path.exists(model_path) or not os.path.exists(feature_names_path):
            print(f"⚠️  ML モデルファイルが見つかりません: {model_path}")
            return None, None
        
        model = joblib.load(model_path)
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        
        print(f"✅ ML モデル読み込み完了: {len(feature_names)} 特徴量")
        return model, feature_names
    except Exception as e:
        print(f"❌ ML モデル読み込みエラー: {e}")
        return None, None

def generate_ml_features(close_prices, volume_data, ticker):
    """
    指定された銘柄のML特徴量を生成（train_ml_model.pyと同じ）
    """
    try:
        if ticker not in close_prices.columns or ticker not in volume_data.columns:
            return pd.DataFrame()
        
        price = close_prices[ticker].dropna()
        volume = volume_data[ticker].dropna()
        
        if len(price) < 250:
            return pd.DataFrame()
        
        # 特徴量生成
        feature_data = pd.DataFrame(index=price.index)
        
        # 1. 価格関連の特徴量
        feature_data['price_ratio_5'] = price / price.shift(5)
        feature_data['price_ratio_10'] = price / price.shift(10)
        feature_data['price_ratio_20'] = price / price.shift(20)
        
        # 2. 移動平均からの乖離
        sma_20 = price.rolling(20).mean()
        sma_50 = price.rolling(50).mean()
        feature_data['price_vs_sma20'] = price / sma_20
        feature_data['price_vs_sma50'] = price / sma_50
        feature_data['sma20_vs_sma50'] = sma_20 / sma_50
        
        # 3. RSI
        rsi = vbt.RSI.run(price, window=14).rsi
        if isinstance(rsi, pd.DataFrame):
            rsi = rsi.iloc[:, 0]
        feature_data['rsi'] = rsi
        
        # 4. 出来高関連
        vol_sma_20 = volume.rolling(20).mean()
        feature_data['volume_ratio'] = volume / vol_sma_20
        
        # 5. ボラティリティ
        returns = price.pct_change()
        feature_data['volatility_20'] = returns.rolling(20).std()
        
        # 6. モメンタム
        feature_data['momentum_10'] = (price / price.shift(10) - 1) * 100
        feature_data['momentum_20'] = (price / price.shift(20) - 1) * 100
        
        # 週次リサンプリング
        weekly_features = feature_data.resample('W-FRI').last()
        weekly_features = weekly_features.fillna(0)
        
        return weekly_features
    except Exception as e:
        print(f"❌ 特徴量生成エラー ({ticker}): {e}")
        return pd.DataFrame()

# ML モデルと特徴量名を読み込み
ml_model, ml_feature_names = load_ml_model()

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
# 5. ML スコア（週次）
ml_scores = pd.DataFrame()

if ml_model is not None and ml_feature_names is not None:
    print("🤖 ML 予測スコア生成中...")
    
    # 週次インデックスを作成
    weekly_index = pd.date_range(start=close.index[0], end=close.index[-1], freq='W-FRI')
    ml_scores = pd.DataFrame(index=weekly_index)
    
    for ticker in close.columns:
        if ticker in ["14"]:  # 無効な列をスキップ
            continue
            
        # ML特徴量生成
        ml_features = generate_ml_features(close, volume, ticker)
        
        if ml_features.empty:
            ml_scores[ticker] = 0.0
            continue
        
        try:
            # 特徴量名の順序を合わせる
            features_aligned = ml_features.reindex(columns=ml_feature_names, fill_value=0)
            
            # 予測実行
            probabilities = ml_model.predict(features_aligned.values)
            
            # 確率をスコアに変換（0-1の確率を0-2のスコアに変換で重み付け）
            ml_score_series = pd.Series(probabilities * 2, index=features_aligned.index)
            
            # インデックスを合わせて代入
            ml_scores[ticker] = ml_score_series.reindex(ml_scores.index, fill_value=0)
            
        except Exception as e:
            print(f"❌ ML予測エラー ({ticker}): {e}")
            ml_scores[ticker] = 0.0
    
    print("✅ ML スコア生成完了")
else:
    print("⚠️  ML モデルが利用できません - ML スコアは 0 に設定")
    # 週次インデックスを作成
    weekly_index = pd.date_range(start=close.index[0], end=close.index[-1], freq='W-FRI')
    ml_scores = pd.DataFrame(index=weekly_index)
    for ticker in close.columns:
        ml_scores[ticker] = 0.0

# ------------------------------------------------------------------------------
# 6. 合計スコア（週次）
total = tech_score.resample("W-FRI").last()

#   価格データに存在するティッカーだけ残す（“14” 列などを除去）
valid_cols = close.columns
total = total.loc[:, total.columns.intersection(valid_cols)]

#   ファンダ & ニュース & ML スコアを加算
total = total.add(fund_score, axis=1).add(news_score, axis=1)

# ML スコアを追加（インデックスを合わせる）
ml_scores_aligned = ml_scores.reindex(index=total.index, columns=total.columns, fill_value=0)
total = total.add(ml_scores_aligned, fill_value=0)

#   ★ 欠損は 0 扱いに統一
total = total.fillna(0)

# ------------------------------------------------------------------------------
# 7. 保存
total.to_parquet(f"{cfg.FACTOR_DIR}/scores.parquet")
print("✅ Factor engine complete (ML統合版)")


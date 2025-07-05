#!/usr/bin/env python3
"""
train_ml_model.py - 機械学習モデルの訓練
翌週の株価上昇確率を予測するLightGBMモデルを訓練
"""

import os
import json
import pathlib
import numpy as np
import pandas as pd
import vectorbt as vbt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
import settings as cfg


def create_features(close_prices, volume_data):
    """
    ML用の特徴量を生成
    
    Args:
        close_prices: 終値データ (DataFrame)
        volume_data: 出来高データ (DataFrame)
    
    Returns:
        features_df: 特徴量DataFrame
    """
    features_list = []
    
    for ticker in close_prices.columns:
        if ticker not in volume_data.columns:
            continue
            
        price = close_prices[ticker].dropna()
        volume = volume_data[ticker].dropna()
        
        # 最低限のデータ数チェック
        if len(price) < 250:  # 約1年分のデータ
            continue
            
        # 特徴量生成
        feature_data = pd.DataFrame(index=price.index)
        
        # 1. 価格関連の特徴量
        feature_data['price_ratio_5'] = price / price.shift(5)  # 5日前比
        feature_data['price_ratio_10'] = price / price.shift(10)  # 10日前比
        feature_data['price_ratio_20'] = price / price.shift(20)  # 20日前比
        
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
        
        # 7. ターゲット作成（翌週の価格上昇判定）
        # 週次リサンプリング
        weekly_price = price.resample('W-FRI').last()
        target = (weekly_price.shift(-1) > weekly_price).astype(int)
        
        # 特徴量も週次リサンプリング
        weekly_features = feature_data.resample('W-FRI').last()
        
        # ターゲットとインデックスを合わせる
        aligned_target = target.reindex(weekly_features.index)
        
        # 欠損値を除去
        valid_idx = weekly_features.dropna().index.intersection(aligned_target.dropna().index)
        if len(valid_idx) < 50:  # 最低50週間のデータ
            continue
            
        # データを追加
        for date in valid_idx:
            row = weekly_features.loc[date].copy()
            row['ticker'] = ticker
            row['date'] = date
            row['target'] = aligned_target.loc[date]
            features_list.append(row)
    
    if not features_list:
        raise ValueError("特徴量データが生成されませんでした")
    
    return pd.DataFrame(features_list)


def train_model():
    """
    機械学習モデルの訓練
    """
    print("🚀 ML モデル訓練開始...")
    
    # データディレクトリの存在確認
    pathlib.Path(cfg.DATA_DIR).mkdir(exist_ok=True)
    
    # 価格・出来高データ読み込み
    close_path = f"{cfg.DATA_DIR}/close.csv"
    volume_path = f"{cfg.DATA_DIR}/volume.csv"
    
    if not os.path.exists(close_path) or not os.path.exists(volume_path):
        print(f"❌ データファイルが見つかりません: {close_path}, {volume_path}")
        print("まずdata_download.pyを実行してください")
        return
    
    close_prices = pd.read_csv(close_path, index_col=0, parse_dates=True)
    volume_data = pd.read_csv(volume_path, index_col=0, parse_dates=True)
    
    print(f"📊 データ読み込み完了: {len(close_prices)} 日分, {len(close_prices.columns)} 銘柄")
    
    # 特徴量生成
    print("🔧 特徴量生成中...")
    features_df = create_features(close_prices, volume_data)
    
    print(f"✅ 特徴量生成完了: {len(features_df)} サンプル")
    
    # 特徴量とターゲットの分離
    feature_cols = [col for col in features_df.columns if col not in ['ticker', 'date', 'target']]
    X = features_df[feature_cols]
    y = features_df['target']
    
    # 欠損値の最終チェック
    X = X.fillna(0)
    
    print(f"📈 特徴量数: {len(feature_cols)}, サンプル数: {len(X)}")
    print(f"📊 ターゲット分布: {y.value_counts().to_dict()}")
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # LightGBMモデル訓練
    print("🎯 LightGBM訓練中...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    # モデル評価
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = (y_pred_binary == y_test).mean()
    
    print(f"📊 モデル評価結果:")
    print(f"   AUC Score: {auc_score:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    # 分類レポート
    print("\n📋 分類レポート:")
    print(classification_report(y_test, y_pred_binary))
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 特徴量重要度 (Top 10):")
    print(feature_importance.head(10))
    
    # モデル保存
    model_path = f"{cfg.DATA_DIR}/ml_model.joblib"
    joblib.dump(model, model_path)
    print(f"💾 モデル保存完了: {model_path}")
    
    # 特徴量名を保存
    feature_names_path = f"{cfg.DATA_DIR}/feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_cols, f)
    print(f"📝 特徴量名保存完了: {feature_names_path}")
    
    print("✅ ML モデル訓練完了!")
    return model


if __name__ == "__main__":
    train_model()
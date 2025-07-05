import os
import json
import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import vectorbt as vbt
import settings as cfg

# Ensure directories exist
pathlib.Path(cfg.DATA_DIR).mkdir(exist_ok=True)
pathlib.Path("models").mkdir(exist_ok=True)

def load_data():
    """Load price and volume data"""
    close = pd.read_csv(f"{cfg.DATA_DIR}/close.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(f"{cfg.DATA_DIR}/volume.csv", index_col=0, parse_dates=True)
    
    # Handle MultiIndex columns if present
    if isinstance(close.columns, pd.MultiIndex):
        close = close.xs('Adj Close', level=1, axis=1)
    if isinstance(volume.columns, pd.MultiIndex):
        volume = volume.xs('Volume', level=1, axis=1)
    
    return close, volume

def generate_features(close, volume):
    """Generate technical features for ML model"""
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

def generate_targets(close, lookforward_days=7):
    """Generate target: 1 if price increases in next week, 0 otherwise"""
    targets = pd.DataFrame(index=close.index)
    
    for ticker in close.columns:
        price = close[ticker]
        future_price = price.shift(-lookforward_days)
        targets[f'{ticker}_target'] = (future_price > price).astype(int)
    
    return targets

def prepare_ml_data(features, targets):
    """Prepare data for ML training"""
    # Stack data for all tickers
    X_list = []
    y_list = []
    
    for ticker in targets.columns:
        ticker_name = ticker.replace('_target', '')
        
        # Get feature columns for this ticker
        feature_cols = [col for col in features.columns if col.startswith(ticker_name + '_')]
        
        if not feature_cols:
            continue
            
        ticker_features = features[feature_cols]
        ticker_target = targets[ticker]
        
        # Remove rows with NaN values
        valid_mask = ~(ticker_features.isna().any(axis=1) | ticker_target.isna())
        valid_features = ticker_features[valid_mask]
        valid_target = ticker_target[valid_mask]
        
        if len(valid_features) > 0:
            X_list.append(valid_features)
            y_list.append(valid_target)
    
    if not X_list:
        raise ValueError("No valid data found for training")
    
    X = pd.concat(X_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)
    
    return X, y

def train_model():
    """Train LightGBM model"""
    print("Loading data...")
    close, volume = load_data()
    
    print("Generating features...")
    features = generate_features(close, volume)
    
    print("Generating targets...")
    targets = generate_targets(close)
    
    print("Preparing ML data...")
    X, y = prepare_ml_data(features, targets)
    
    print(f"Training data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.1,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model AUC: {auc:.4f}")
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = f"{cfg.DATA_DIR}/ml_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature names for later use
    feature_names = list(X.columns)
    with open(f"{cfg.DATA_DIR}/feature_names.json", 'w') as f:
        json.dump(feature_names, f)
    
    return model, auc, accuracy

if __name__ == "__main__":
    try:
        model, auc, accuracy = train_model()
        print("✅ ML model training complete")
        print(f"   AUC: {auc:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise
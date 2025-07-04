import numpy as np
import pandas as pd
from pathlib import Path
import settings as cfg   # ← settings.py に合わせる

# ──────────────────────────────────────────────
def _load_prices(path: str) -> pd.DataFrame:
    """close.csv を読み込み，列を 1 段（ティッカーのみ）に整形"""
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # ❶ MultiIndex（ticker, field）の場合 → Close / Adj Close を抽出
    if isinstance(df.columns, pd.MultiIndex):
        # 'Adj Close' があれば優先，なければ 'Close'
        for fld in ["Adj Close", "Close"]:
            if fld in df.columns.get_level_values(1):
                df = df.xs(fld, level=1, axis=1)
                break
        else:  # 見つからなければエラー
            raise ValueError("Close 列が見つかりません")

    # ❷ 列名を str に統一
    df.columns = df.columns.astype(str)

    return df.sort_index()

# ──────────────────────────────────────────────
def run_backtest(start_date=None, end_date=None):
    """CAGR / MDD / Sharpe を返す"""
    close  = _load_prices(f"{cfg.DATA_DIR}/close.csv")
    score  = pd.read_parquet(f"{cfg.FACTOR_DIR}/scores.parquet")

    # -------- インデックス合わせ --------
    if start_date:  close = close.loc[start_date:]
    if end_date:    close = close.loc[:end_date]
    score = score.loc[score.index.intersection(close.index)]

    # -------- スコア → ウェイト（softmax）--------
    w_exp   = np.exp(score)
    weights = w_exp.div(w_exp.sum(axis=1), axis=0)

    # 週次 → 日次
    weights = (
        weights.reindex(close.index)  # 日次インデックスに揃える
               .ffill()
               .fillna(0)
    )

    # -------- ポートフォリオ算出 --------
    daily_ret = close.pct_change(fill_method=None).fillna(0)
    port_ret  = (weights.shift().fillna(0) * daily_ret).sum(axis=1)
    equity    = (1 + port_ret).cumprod()

    # -------- 指標 --------
    days   = len(equity)
    cagr   = (equity.iloc[-1] / equity.iloc[0]) ** (252 / days) - 1
    mdd    = (equity / equity.cummax() - 1).min()
    std    = port_ret.std(ddof=0)
    sharpe = 0 if std == 0 else (port_ret.mean() * 252) / (std * np.sqrt(252))

    # -------- 保存 --------
    Path(cfg.REPORT_DIR).mkdir(exist_ok=True)
    pd.DataFrame({"CAGR": [cagr], "MDD": [mdd], "Sharpe": [sharpe]}
                 ).to_csv(f"{cfg.REPORT_DIR}/backtest_stats.csv", index=False)

    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe}

# ──────────────────────────────────────────────
if __name__ == "__main__":
    stats = run_backtest()
    print("✅ Backtest complete →", f"{cfg.REPORT_DIR}/backtest_stats.csv")


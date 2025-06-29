%%writefile 03_backtest.py
import pandas as pd, vectorbt as vbt, 00_settings as cfg
from pathlib import Path

close  = pd.read_csv(f"{cfg.DATA_DIR}/close.csv", index_col=0, parse_dates=True)
score  = pd.read_parquet(f"{cfg.FACTOR_DIR}/scores.parquet")

# --- スコアを割合に変換（負値は 0） ---------------------------------------
weights = score.clip(lower=0)
weights = weights.div(weights.sum(axis=1), axis=0)
weights = weights.reindex(close.loc[weights.index].index)

pf = vbt.Portfolio.from_weights(close, weights, freq="W-FRI", init_cash=1_000_000)
Path(cfg.REPORT_DIR).mkdir(exist_ok=True)
pf.stats().to_csv(f"{cfg.REPORT_DIR}/backtest_stats.csv")
print("✅ Backtest complete →", f"{cfg.REPORT_DIR}/backtest_stats.csv")

%%writefile 00_settings.py
import os, datetime
FMP_API_KEY     = os.getenv("FMP_API_KEY")
MARKETAUX_KEY   = os.getenv("MARKETAUX_KEY")

BACKTEST_START  = "2020-01-01"
BACKTEST_END    = datetime.date.today().isoformat()

UNIVERSE_US = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL",
    "META","TSLA","BRK.B","JNJ","V"
]                       # ← まず 10 銘柄で検証

DATA_DIR    = "data"
FACTOR_DIR  = "factors"
REPORT_DIR  = "reports"

# Stock Pipeline

A comprehensive stock market data pipeline for downloading, processing, and backtesting stock market data.

## Features

- **Stock Data Download**: Automatically downloads stock price and volume data using yfinance
- **Fundamental Data Collection**: Fetches fundamental data from Financial Modeling Prep API
- **Sentiment Analysis**: Collects market sentiment data
- **Backtesting Engine**: Implements backtesting capabilities using vectorbt
- **Factor Analysis**: Analyzes various stock factors for strategy development
- **Hyperparameter Optimization**: Optimizes trading strategies using hyperopt

## Project Structure

```
stock-pipeline/
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── settings.py               # Configuration settings
├── data_download.py          # Data download and collection
├── backtest.py              # Backtesting engine
├── factor_engine.py         # Factor analysis engine
├── hyperopt.py              # Hyperparameter optimization
├── config/                  # Configuration files
│   └── config_20250704.json
└── data/                   # Data storage (created automatically)
    ├── fundamentals/       # Fundamental data
    └── sentiment/         # Sentiment data
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   export FMP_API_KEY="your_financial_modeling_prep_api_key"
   export MARKETAUX_KEY="your_marketaux_api_key"
   ```

3. **Configure settings**:
   Edit `settings.py` to customize:
   - Stock universe (currently set to 10 major US stocks)
   - Backtest date range
   - Data directories

## Usage

### Download Data
```bash
python data_download.py
```
This will download:
- Stock prices and volumes
- Fundamental data from Financial Modeling Prep
- Market sentiment data

### Run Backtests
```bash
python backtest.py
```

### Factor Analysis
```bash
python factor_engine.py
```

### Hyperparameter Optimization
```bash
python hyperopt.py
```

## Current Stock Universe

The pipeline currently analyzes these 10 major US stocks:
- AAPL, MSFT, NVDA, AMZN, GOOGL
- META, TSLA, BRK.B, JNJ, V

## Configuration

- **Backtest Period**: 2020-01-01 to present
- **Data Sources**: Yahoo Finance, Financial Modeling Prep, MarketAux
- **Storage**: Local CSV and JSON files

## Dependencies

- pandas: Data manipulation and analysis
- yfinance: Yahoo Finance data download
- requests: HTTP requests for API calls
- vectorbt: Backtesting and analysis framework

## License

This project is for educational and research purposes.
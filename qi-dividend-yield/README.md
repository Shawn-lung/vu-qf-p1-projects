# QI Part 2 - Dividend yield forecasting and portfolio allocation (Fama and French, 1988)

This project uses dividend yields to forecast market returns. CAPM is then used to estimate stock returns and build a portfolio.

**Team project** (three students). **My part:** the data preparation (`lib/data_preparation.py`) and the two return models: the dividend-yield market model (`lib/market_model.py`) and the CAPM mapping to stock returns (`lib/capm_model.py`). Teammates wrote the covariance and portfolio optimisation and the backtest.

## How to run

Use Python 3.12. Open a terminal in this project folder, where `main.py` is saved.

1. Install the packages:

```bash
python -m pip install -r requirements.txt
```

2. Run the portfolio model:

```bash
python main.py
```

This prepares the data, estimates the models and creates the final portfolio. The input files must be in `data` first (see Data below).

3. To also run the 2021-2025 monthly backtest:

```bash
python -m lib.backtest
```

Run `main.py` first. The backtest uses its output files and may take several minutes.

## Output files

Results are saved in `outputs`:

- `Hand_in_sheet.xlsx`: final portfolio in Part II, column K. Needs the course's blank template (see Data).
- `holdings.csv`: stocks held, whole shares and invested amounts.
- `portfolio_summary.csv`: expected portfolio return, risk and risky share.
- `expected_return.csv`: stock return forecasts for 2021-2025.
- `expected_return_forecast.csv`: stock return forecasts for January 2026.
- `backtest_*.csv`: backtest results, created by the backtest command.

Running the commands again overwrites these output files.

## Results

Team project; numbers from the submitted summary.

- **Predictability:** the dividend yield barely predicts next month's market return (slope p = 0.463, adjusted R² 0.001), but the slopes for two to four years are significant at 5%, with adjusted R² rising to 0.086 at four years. White and Breusch-Godfrey tests show heteroskedasticity and autocorrelation, so all inference uses HAC standard errors. The long-horizon pattern resembles Fama and French (1988), with weaker explanatory power.
- **Portfolio (January 2026):** CAPM with Blume-adjusted betas turns the market forecast into stock returns; a Ledoit-Wolf covariance from 2023-2025 daily returns feeds a long-only maximum-Sharpe portfolio with a 10% cap per stock. Result: 51 stocks, expected return 6.42% a year at 9.92% volatility.
- **Backtest 2021-2025:** with monthly rebalancing on lagged forecasts, the strategy returned 9.30% a year against 14.57% for the market, with volatility 8.88% against 15.14% and a maximum drawdown of 6.87% against 23.67%. Beta is 0.40; the alpha of 2.82% a year is not significant (p = 0.378).
- **Sensitivity:** in April 2023, when March 2020 left the three-year covariance window, the share invested in stocks rose from 0.52 to 0.86 while expected returns barely changed. The covariance estimate, not the return forecast, drove the allocation.
- **Limitations:** the backtest ignores transaction costs and whole-share rounding.

## Data

Not included. The CRSP extracts are licensed through WRDS; download them with your own access. `main.py` reads these files from `data`:

| File | Content |
|---|---|
| `stock_monthly_raw.csv`, `stock_monthly_returns.csv` | CRSP monthly stock data and returns |
| `stock_daily_returns.csv` | CRSP daily stock returns, for the covariance matrix |
| `sp500_universe_monthly_raw.csv` | CRSP value-weighted index of the S&P 500 universe, the market proxy |
| `rf_monthly_raw.csv` | CRSP 30-day T-bill returns, the monthly risk-free rate |
| `stock_otc_prices_raw.csv` | OTC prices for stocks that left their exchange |
| `portfolio_template_part1.xlsx` | The course's blank hand-in sheet, for the final export only |

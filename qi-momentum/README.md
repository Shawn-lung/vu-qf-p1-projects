# QI Part 1 - Momentum (Jegadeesh and Titman, 1993)

**Team project** (three students). **My part:** everything except the original replication in `lib/momentum.py`, which a teammate wrote. I extended that file so that stocks with a recorded delisting are not used in new formations, a missing holding-period return stops the run instead of being dropped silently, and delisted or OTC-exited holdings are kept as cash (`lib/exit_handling.py`); I also added the pipeline function and the outputs for the portfolio step. I wrote the data preparation, the minimum-variance portfolio and backtest (`lib/optimization.py`), the export and `main.py`.

- Run `python main.py` from the project folder.
- Install the packages: `python -m pip install -r requirements.txt`.
- A fresh copy has no saved results. Create an empty `outputs` folder and set `prepare_csv = True` in `main.py` to build the winner baskets, expected returns and optimized weights.
- To also save updated returns and performance statistics, set `prepare_csv=True` in the `organize_output()` call.
- The final allocation is written to `outputs/Hand_in_sheet.xlsx`. This step needs the course's blank hand-in sheet as `data/Hand_in_sheet.xlsx`, which is not included.

## Results

Team project; numbers from the submitted summary.

- **Replication:** stocks are sorted into deciles on their past six-month return and held for six months. Over 114 months, past winners (decile 10) earned 2.11% per month (t = 3.93), the highest of the ten deciles. The winner-minus-loser spread, 0.42% per month, is not significant (t = 0.99); since short selling is not allowed, only the long leg matters.
- **Portfolio:** every stock in the winner decile gets the same expected return, so a long-only minimum-variance portfolio (two years of daily returns, at most 10% per stock) decides the weights. One-sixth of the capital enters a new cohort each month.
- **Backtest 2021-2025:** EUR 10m grows to EUR 20.3m: 15.2% a year at 15.2% volatility (Sharpe ratio 0.95), against 14.3% at 15.9% for an equally weighted portfolio of the same stocks.
- **Limitation:** with equal expected returns inside the decile, the covariance matrix alone sets the weights; the momentum signal only chooses which stocks are eligible.

## Data

The input data are not included: the CRSP extracts are licensed through WRDS, and the daily file supplied by the lecturer cannot be redistributed. [`DATA.md`](DATA.md) describes each file and its columns.

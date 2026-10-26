# VU Quantitative Finance — Period 1 Projects

Six empirical projects from the first period of the MSc Finance, Honours Programme in Quantitative Finance at Vrije Universiteit Amsterdam (autumn 2026), by Hsiang-I (Shawn) Lung and teammates. All six are team projects; each folder states which part I wrote and has its own README with results and limitations. The two Econometrics projects are described without code or detailed results, at the request of the course coordinator.

**Tools:** Python (pandas, NumPy, SciPy, statsmodels, scikit-learn), DuckDB SQL, WRDS (CRSP, Compustat, Financial Ratios Suite), Git.

## One lesson from the three Quantitative Investing projects

The three projects use the same long-only mean-variance setup with three different return models: momentum, dividend yields with CAPM, and machine learning. Each time, the covariance matrix rather than the return forecast ended up deciding the portfolio:

- **Momentum:** every stock in the winner decile gets the same expected return, so the covariance matrix alone sets the weights.
- **Dividend yield:** when March 2020 left the three-year covariance window in April 2023, the share invested in stocks jumped from 0.52 to 0.86 while expected returns barely changed.
- **Machine learning:** the random forest gave the best forecasts, but they differ little across stocks; the covariance matrix drove the weights and the portfolio turned defensive (beta 0.68), returning 13.1% a year against 23.0% for the market.

With return forecasts this noisy, how the risk is estimated (window, shrinkage, treatment of missing data) matters as much as the forecast itself.

## Projects

### [Momentum and a minimum-variance winner portfolio](qi-momentum)
*Quantitative Investing, Part 1; replicates Jegadeesh and Titman (1993).*
- **Question:** do past winners keep winning, and how should a long-only investor hold them?
- **Result:** past winners earned 2.11% a month (t = 3.93), but the winner-minus-loser spread is not significant. From 2021 to 2025 the strategy returned 15.2% a year at 15.2% volatility, against 14.3% at 15.9% for an equally weighted portfolio.
- **My part:** data preparation, delisting and exit handling, the portfolio optimisation and backtest.

### [Dividend-yield forecasts and a CAPM portfolio](qi-dividend-yield)
*Quantitative Investing, Part 2; adapts Fama and French (1988).*
- **Question:** does the dividend yield predict market returns, and what does that imply for a stock portfolio?
- **Result:** predictive power only at two- to four-year horizons. The Ledoit-Wolf mean-variance portfolio returned 9.30% a year against 14.57% for the market, with a maximum drawdown of 6.87% against 23.67% and no significant alpha.
- **My part:** data preparation, the dividend-yield market model and the CAPM mapping to stock returns.

### [Machine-learning return forecasts](qi-ml-forecasts)
*Quantitative Investing, Part 3; adapts Gu, Kelly and Xiu (2020).*
- **Question:** does a random forest forecast stock returns better than linear models, and does that help a portfolio?
- **Result:** the random forest has the best out-of-sample R² (1.31% against 0.82% for OLS-3) and the only useful stock ranking, but its mean-variance portfolio lagged the market.
- **My part:** data preparation: choosing and building the 12 stock characteristics.

### [The overreaction hypothesis in ETFs](eqf-overreaction)
*Econometrics for Quantitative Finance, Assignment 1; follows De Bondt and Thaler (1985) and Clare and Thomas (1995). Description only.*
- **Question:** do past losers outperform past winners among ETFs listed on Borsa Italiana?
- **Result:** no evidence of overreaction in any specification, including controls for market risk and liquidity.
- **My part:** the whole pipeline behind the main results, including the data cleaning and corporate-action corrections.

### [Volatility and correlation models](eqf-volatility)
*Econometrics for Quantitative Finance, Assignment 2. Description only.*
- **Question:** how should the volatility and correlation of two European-listed equity ETFs be modelled?
- **Result:** EGARCH fits best: negative shocks raise volatility more than positive ones. The correlation between the two markets is far from constant and rises in periods of market stress.
- **My part:** GARCH, EWMA and RiskMetrics, the DCC correlation model with robust standard errors, and a revision of the EGARCH model selection.

### [Dynamic delta hedging](spf-dynamic-hedging)
*Stochastic Processes: the Fundamentals, computer assignment.*
- **Question:** how well does delta hedging replicate a short position in S&P 500 calls, and what drives the error?
- **Result:** the standard deviation of the hedging error falls from USD 222,000 with monthly rebalancing to USD 26,000 with four rebalances a day, roughly with 1/√n; what remains comes from gamma.
- **My part:** the Black-Scholes pricing, the path simulation and the hedging simulations.

## Reproducibility

On 24 October 2026 every project with code in this repository was rerun from that code with the original input data. The outputs match the submitted results; in the momentum project some portfolio weights differ by up to 0.07 percentage points because of optimizer tolerance, with the same performance figures.

## Data

This repository contains no market data. The Quantitative Investing projects use CRSP and Compustat data through WRDS, which are licensed and may not be redistributed; you need your own WRDS access. The dynamic hedging project uses the S&P 500 index from FRED. Each folder's README lists the files it needs.

## Use of AI

- **Quantitative Investing:** as declared in our submissions, we used AI to check the WRDS monthly returns and write a script that finds missing holding-period returns; to write the code that handles delisted stocks (the OTC closing price as exit price where no delisting return exists, with the proceeds held as cash until the holding period ends); for suggestions on adjusted betas and Ledoit-Wolf shrinkage; for debugging and the Excel export; and to improve the wording of the summaries and READMEs.
- **Econometrics and Stochastic Processes:** we used AI mainly to check the logic of our methods and to help debug the code, not to write the code.
- **This repository:** the READMEs and result summaries were drafted with an AI assistant (Claude) from our reports and code, and checked by me; the rerun of all projects was also done with its help. Claude also assembled the repository: it copied the code from our original submissions without data, made the commits and pushed them, which is why it appears as a co-author of the commits. The project code itself is the code we submitted for the courses.

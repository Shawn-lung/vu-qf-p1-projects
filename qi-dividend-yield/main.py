from pathlib import Path
import pandas as pd
from lib.capm_model import CAPMModel
from lib.covariance_model import CovarianceModel
from lib.data_preparation import index_data_prep, stock_data_prep
from lib.export_portfolio import export_to_excel
from lib.market_model import DividendYieldModel
from lib.portfolio import MeanVariancePortfolio


CAPITAL = 10_000_000
RF_MONTHLY = 1.01 ** (1 / 12) - 1   # assignment risk-free rate: 1% per year
CAP = 0.10                          # max weight per stock in the tangency portfolio
GAMMA = 4                           # risk aversion for the stock / risk-free split


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    index_data_prep(data_dir, outputs_dir)
    stock_data_prep(data_dir, outputs_dir)

    period = 1

    # Evaluation: estimate on data up to 2020 and predict 2021-2025 out of sample
    market_model = DividendYieldModel(outputs_dir, period=period, yield_type='end')
    market_return = market_model.simple_prediction

    capm_model = CAPMModel(outputs_dir, blume=True)
    expected_return = capm_model.predict(market_return=market_return)
    capm_model.results[['nobs', 'eligibility']].to_csv(str(outputs_dir / "stock_eligibility.csv"))
    expected_return.to_csv(str(outputs_dir / "expected_return.csv"), index=False)

    # Portfolio inputs: use all information available on 31-12-2025
    final_market_model = DividendYieldModel(
        outputs_dir, period=period, yield_type='end', train_end='2025-12-01'
    )
    market_return_forecast = final_market_model.simple_return_forecast()

    final_capm_model = CAPMModel(
        outputs_dir, train_start='2021-01-01', train_end='2025-12-01', blume=True
    )
    expected_return_forecast = final_capm_model.predict(
        market_return=market_return_forecast, rf=RF_MONTHLY
    )
    expected_return_forecast.to_csv(str(outputs_dir / "expected_return_forecast.csv"), index=False)

    template = data_dir / "portfolio_template_part1.xlsx"
    sheet_permnos = pd.read_excel(template, header=19, usecols='A')['PERMNO'].dropna().astype(int)
    mu = expected_return_forecast.set_index('PERMNO')['expected_return']
    mu = mu.loc[mu.index.isin(sheet_permnos)]

    cov_model = CovarianceModel(data_dir, mu.index)

    portfolio = MeanVariancePortfolio(mu, cov_model.cov, rf=RF_MONTHLY, cap=CAP, gamma=GAMMA)
    summary = portfolio.summary()
    summary.to_csv(str(outputs_dir / "portfolio_summary.csv"))

    holdings = export_to_excel(
        portfolio.weights, CAPITAL, template, outputs_dir / "Hand_in_sheet.xlsx"
    )
    holdings.loc[holdings['shares'] > 0].to_csv(str(outputs_dir / "holdings.csv"), index=False)

    print(f"Market forecast Jan 2026: {market_return_forecast.iloc[0]:.4%} per month")
    print(f"Stocks with a forecast: {len(mu)}, with full covariance data: {len(cov_model.cov)}")
    print(f"Ledoit-Wolf shrinkage intensity: {cov_model.shrinkage:.3f}")
    print(summary.round(4))
    print(f"Invested in stocks: {holdings['investment'].sum():,.2f}")
    print(f"Invested at risk-free: {CAPITAL - holdings['investment'].sum():,.2f}")


if __name__ == "__main__":
    main()

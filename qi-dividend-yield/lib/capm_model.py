from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


class CAPMModel:
    """
    Purpose:
        Estimate betas and predict stock returns

    Inputs:
        outputs_dir    folder with prepared CSV files
        train_start    first month of the beta estimation window
        train_end      last month of the beta estimation window
        blume          shrink betas toward 1 with the Blume adjustment

    Outputs:
        Stock estimates and return predictions
    """

    def __init__(
        self,
        outputs_dir: Path,
        train_start: str = '2016-01-01',
        train_end: str = '2020-12-01',
        blume: bool = False
    ):
        self.outputs_dir = outputs_dir
        self.blume = blume
        self.df = self.get_stock_data()
        self.rf_forecast = self.get_rf_forecast()
        self.train = self.df.loc[
            (self.df.index >= train_start)
            & (self.df.index <= train_end)
        ]
        self.test = self.df.loc[self.df.index > train_end]
        self.results = self.fit()

    def get_stock_data(self):
        """
        Purpose:
            Read stock data

        Returns:
            DataFrame indexed by Month
        """
        df = pd.read_csv(
            str(self.outputs_dir / "stock_monthly.csv"),
            index_col='Month',
            usecols=['Month', 'PERMNO', 'stock_excess', 'market_excess', 'Return_flag', 'Rf'],
            parse_dates=True
        )
        return df

    def get_rf_forecast(self):
        """
        Purpose:
            Use the last monthly RF to forecast the next month

        Returns:
            RF Series indexed by target month
        """
        rf_data = self.df[['Rf']].reset_index()
        if (rf_data.groupby('Month')['Rf'].nunique(dropna=False) > 1).any():
            raise ValueError('RF values must be identical within each month')

        rf = (
            rf_data.drop_duplicates(subset=['Month'])
            .set_index('Month')['Rf']
            .sort_index()
            .asfreq('MS')
        )
        next_month = rf.index[-1] + pd.DateOffset(months=1)
        rf.loc[next_month] = np.nan
        return rf.shift(1).rename('rf_forecast')

    def fit(self):
        """
        Purpose:
            Estimate alpha and beta

        Returns:
            Estimates and eligibility by PERMNO
        """
        train = self.train.loc[self.train['Return_flag'] == 'CR']
        results = []
        for permno in self.train['PERMNO'].unique():
            train_permno = train.loc[train['PERMNO'] == permno]
            nobs = len(train_permno)
            if nobs < 60:
                results.append({
                    'PERMNO': permno,
                    'nobs': nobs,
                    'eligibility': False
                })
                continue
            x = train_permno['market_excess']
            Y = train_permno['stock_excess']
            X = sm.add_constant(x)
            model = sm.OLS(Y, X)
            result = model.fit()
            results.append(
                {
                    'PERMNO': permno,
                    'alpha': result.params['const'],
                    'beta': result.params['market_excess'],
                    'alpha_pvalue': result.pvalues['const'],
                    'beta_pvalue': result.pvalues['market_excess'],
                    'nobs': int(result.nobs),
                    'eligibility': True
                }
            )
        results = pd.DataFrame(results).set_index('PERMNO')
        results['beta_blume'] = 0.33 + 0.67 * results['beta']
        return results

    def predict(self, market_return: pd.Series, rf: float | None = None):
        """
        Purpose:
            Predict stock returns with CAPM

        Inputs:
            market_return    monthly simple return forecasts
            rf               fixed monthly risk-free rate; last T-bill if None

        Returns:
            Stock return DataFrame in decimals
        """
        if rf is None:
            rf = self.rf_forecast.reindex(market_return.index)
            if rf.isna().any():
                raise ValueError('Missing RF forecasts for the requested months')
        else:
            rf = pd.Series(rf, index=market_return.index)

        beta_col = 'beta_blume' if self.blume else 'beta'
        betas = self.results[beta_col].dropna()
        market_excess_return = market_return - rf
        expected_returns = []
        for permno, beta in betas.items():
            stock_prediction = rf + beta * market_excess_return
            stock_result = pd.DataFrame(
                {
                    'PERMNO': permno,
                    'expected_return': stock_prediction
                }
            )

            stock_result = stock_result.rename_axis('Month').reset_index()
            expected_returns.append(stock_result)
        return pd.concat(expected_returns, ignore_index=True)
    
    
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    outputs_dir = Path(__file__).resolve().parent.parent / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    testing = CAPMModel(outputs_dir)
    print(testing.results)

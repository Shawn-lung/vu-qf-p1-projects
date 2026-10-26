from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class CovarianceModel:
    """
    Purpose:
        Estimate the monthly covariance matrix of stock returns

    Inputs:
        data_dir         folder with the daily return CSV
        permnos          stocks to include
        start            first day of the estimation window
        end              last day of the estimation window
        days_per_month   trading days used to scale daily to monthly
        panel            optional date x PERMNO table of daily returns; when
                         given it is sliced instead of re-reading the CSV
        daily            preloaded daily returns; read from disk when None

    Outputs:
        Shrunk monthly covariance matrix indexed by PERMNO
    """

    def __init__(
        self,
        data_dir: Path,
        permnos,
        start: str = '2023-01-01',
        end: str = '2025-12-31',
        days_per_month: int = 21,
        panel: pd.DataFrame | None = None
    ):
        self.days_per_month = days_per_month
        self.returns = self.get_returns(data_dir, permnos, start, end, panel)
        self.shrinkage = None
        self.cov = self.fit()

    def get_returns(
        self, data_dir: Path, permnos, start: str, end: str,
        panel: pd.DataFrame | None = None
    ):
        """
        Purpose:
            Build a date x stock return table without missing values

        Returns:
            DataFrame indexed by date with one column per PERMNO
        """
        if panel is not None:
            # The backtest re-estimates this every month; slicing a panel that
            # was pivoted once avoids re-reading and re-pivoting a 55 MB file.
            returns = panel.loc[start:end, panel.columns.intersection(permnos)]
        else:
            df = pd.read_csv(
                str(data_dir / 'stock_daily_returns.csv'),
                usecols=['PERMNO', 'DlyCalDt', 'DlyRet'],
                parse_dates=['DlyCalDt']
            )
            df = df.loc[
                (df['DlyCalDt'] >= start)
                & (df['DlyCalDt'] <= end)
                & (df['PERMNO'].isin(permnos))
            ]
            returns = df.pivot(index='DlyCalDt', columns='PERMNO', values='DlyRet')
        # Ledoit-Wolf needs a full panel, so stocks with any missing day are dropped
        return returns.loc[:, returns.notna().all()]

    def fit(self):
        """
        Purpose:
            Estimate the covariance with Ledoit-Wolf shrinkage

        Returns:
            Monthly covariance DataFrame
        """
        lw = LedoitWolf().fit(self.returns.to_numpy())
        self.shrinkage = lw.shrinkage_
        cov = lw.covariance_ * self.days_per_month
        return pd.DataFrame(cov, index=self.returns.columns, columns=self.returns.columns)

    def condition_numbers(self):
        """
        Purpose:
            Compare the stability of the sample and shrunk matrices

        Returns:
            Series with both condition numbers
        """
        sample = np.cov(self.returns.to_numpy(), rowvar=False)
        return pd.Series({
            'sample': np.linalg.cond(sample),
            'ledoit_wolf': np.linalg.cond(self.cov.to_numpy())
        })


if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    permnos = pd.read_csv(data_dir / 'stock_daily_returns.csv', usecols=['PERMNO'])['PERMNO'].unique()
    model = CovarianceModel(data_dir, permnos)
    print(f'stocks: {model.cov.shape[0]}, shrinkage: {model.shrinkage:.3f}')
    print(model.condition_numbers())

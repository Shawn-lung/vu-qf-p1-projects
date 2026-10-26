from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


class DividendYieldModel:
    """
    Purpose:
        Predict market returns using dividend yields

    Inputs:
        outputs_dir    folder with prepared CSV files
        period         forecast horizon in months
        alpha          test significance level
        bg_lags        number of lags for the BG test
        yield_type     start or end yield
        train_end      last month of the estimation sample

    Outputs:
        Fitted models and return predictions
    """

    def __init__(
        self, 
        outputs_dir: Path,
        period: int = 1,
        alpha: float = 0.05,
        bg_lags: int = 6,
        yield_type: str = 'end',
        train_end: str = '2020-12-01'
    ):
        match period:
            case 1 | 3 | 12 | 24 | 36 | 48:
                self.period = period
            case _:
                raise ValueError('period must be 1, 3, 12, 24, 36 or 48')

        match yield_type:
            case 'start' | 'end':
                self.yield_type = yield_type
            case _:
                raise ValueError('yield_type must be start or end')

        self.alpha = alpha
        self.bg_lags = bg_lags
        self.df = self.get_data(outputs_dir, period)
        self.train = self.df.loc[:train_end]
        self.test = self.df.loc[
            (self.df.index > self.train.index[-1])
            & (self.df['forecast_origin'] >= self.train.index[-1])
        ]
        self.result = self.fit()
        self.prediction = self.predict()

        self.simple_result = self.fit_simple_return()
        self.simple_prediction = self.simple_return_predict()

    def get_data(self, outputs_dir: Path, period: int = 1):
        """
        Purpose:
            Prepare returns and lagged yields

        Inputs:
            outputs_dir    folder with prepared CSV files
            period         forecast horizon in months

        Returns:
            DataFrame indexed by Month
        """
        df = pd.read_csv(
            str(outputs_dir / 'sp500_monthly.csv'), 
            index_col='Month', 
            usecols=['Month', 'Total_ret', 'yield_end', 'yield_start'],
            parse_dates=True
        )
        monthly_log_return = np.log1p(df['Total_ret'])

        df['log_return'] = monthly_log_return.rolling(
            window=period,
            min_periods=period
        ).sum()
        df['Total_period_return'] = np.expm1(df['log_return'])

        df['yield_start_lag'] = df['yield_start'].shift(period)
        df['yield_end_lag'] = df['yield_end'].shift(period)
        df['forecast_origin'] = df.index - pd.DateOffset(months=period)
        df = df.dropna()

        match period:
            case 3:
                df = df.loc[df.index.month.isin([3, 6, 9, 12])]
            case 12 | 24 | 36 | 48:
                df = df.loc[df.index.month == 12]

        return df
    
    def fit(self):
        """
        Purpose:
            Fit the log return model

        Returns:
            OLS results
        """
        Y = self.train['log_return']
        x = self.train[f'yield_{self.yield_type}_lag']
        X = sm.add_constant(x)  
        model = sm.OLS(Y, X)
        result = model.fit()
        self.w_test = self.white_test(x, result)
        self.bg_test = self.autocorrelation_test(x, result)

        match (self.w_test, self.bg_test):
            case (True, False) if self.period <= 12:
                return result.get_robustcov_results(cov_type='HC3')
            case (False, False) if self.period <= 12:
                return result
            case _:
                maxlags = max(
                    int(4 * (result.nobs / 100) ** (2 / 9)),
                    self.period // 12 - 1
                )
                return result.get_robustcov_results(
                    cov_type='HAC',
                    maxlags=maxlags,
                    kernel='bartlett',
                    use_correction=True
                )

    def white_test(self, x, fitted_result):
        """
        Purpose:
            Test for changing error variance

        Inputs:
            x                dividend yield
            fitted_result    fitted OLS results

        Returns:
            True if constant variance is rejected
        """
        residuals = fitted_result.resid
        Y = residuals ** 2
        x_square = (x ** 2).rename(f'{x.name}_sq')
        X = pd.concat([x, x_square], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        result = model.fit()
        lm = result.nobs * result.rsquared
        critical_value = chi2.ppf(1 - self.alpha, df=2)
        return bool(lm > critical_value)

    def autocorrelation_test(self, x, fitted_result):
        """
        Purpose:
            Test for serial correlation

        Inputs:
            x                dividend yield
            fitted_result    fitted OLS results

        Returns:
            True if no serial correlation is rejected
        """
        residual = fitted_result.resid
        aux_data = pd.concat(
            [
                residual.rename('residual'),
                x.rename('yield_lag')
            ],
            axis=1
        )
        for lag in range(1, self.bg_lags + 1):
            aux_data[f'residual_lag{lag}'] = residual.shift(lag)

        aux_data = aux_data.dropna()
        Y = aux_data['residual']
        X = aux_data.drop(columns=['residual'])
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        result = model.fit()
        lm = result.nobs * result.rsquared
        critical_value = chi2.ppf(1 - self.alpha, df=self.bg_lags)
        return bool(lm > critical_value)

    def predict(self):
        """
        Purpose:
            Predict log returns for the test sample

        Returns:
            Prediction Series indexed by target month
        """
        x = self.test[f'yield_{self.yield_type}_lag']
        X = sm.add_constant(x, has_constant='add')
        prediction = self.result.predict(X)
        return pd.Series(
            prediction,
            index=self.test.index,
            name='predicted_log_return'
        )
    
    def fit_simple_return(self):
        """
        Purpose:
            Fit the simple return model

        Returns:
            OLS results
        """
        Y = self.train['Total_period_return']
        x = self.train[f'yield_{self.yield_type}_lag']
        X = sm.add_constant(x)  
        model = sm.OLS(Y, X)
        result = model.fit()
        self.w_test_simple = self.white_test(x, result)
        self.bg_test_simple = self.autocorrelation_test(x, result)

        match (self.w_test_simple, self.bg_test_simple):
            case (True, False) if self.period <= 12:
                return result.get_robustcov_results(cov_type='HC3')
            case (False, False) if self.period <= 12:
                return result
            case _:
                maxlags = max(
                    int(4 * (result.nobs / 100) ** (2 / 9)),
                    self.period // 12 - 1
                )
                return result.get_robustcov_results(
                    cov_type='HAC',
                    maxlags=maxlags,
                    kernel='bartlett',
                    use_correction=True
                )

    def simple_return_predict(self):
        """
        Purpose:
            Predict simple returns for the test sample

        Returns:
            Prediction Series in decimals
        """
        x = self.test[f'yield_{self.yield_type}_lag']
        X = sm.add_constant(x, has_constant='add')        
        prediction = self.simple_result.predict(X)

        return pd.Series(
            prediction,
            index=self.test.index,
            name='predicted_simple_return'
        )
    
    def simple_return_forecast(self):
        """
        Purpose:
            Forecast January 2026 return using December 2025 yield

        Returns:
            Monthly simple return Series
        """
        if self.period != 1:
            raise ValueError('This forecast method only supports period=1')
        target_month = pd.to_datetime('2026-01-01')
        x = self.df[f'yield_{self.yield_type}'].loc[['2025-12-01']]

        x.index = [target_month]
        X = sm.add_constant(x, has_constant='add')        
        prediction = self.simple_result.predict(X)

        return pd.Series(
            prediction,
            index=x.index,
            name='predicted_simple_return'
        ) 

        
if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    outputs_dir = Path(__file__).resolve().parent.parent / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    models = {}
    for period in [1, 3, 12, 24, 36, 48]:
        model = DividendYieldModel(outputs_dir, period=period, yield_type='end')
        models[period] = model
        print(f'\nForecast horizon: {period} months')
        print(model.simple_result.summary())

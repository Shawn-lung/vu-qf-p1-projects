"""Out-of-sample backtest of the dividend-yield / CAPM mean-variance portfolio.

The portfolio is rebuilt every month from 2021-01 to 2025-12 on information
available at that date only: the CAPM forecasts in outputs/expected_return.csv
(regression on 1927-2020, betas on 2016-2020) and a Ledoit-Wolf covariance from
the daily returns ending the day before the rebalance month.  Delisted stocks
earn their terminal return in the exit month and leave the universe after it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm

if __package__ in (None, ''):
   
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.covariance_model import CovarianceModel
from lib.portfolio import MeanVariancePortfolio

MONTHS_PER_YEAR = 12


class MaxUtilityPortfolio:
    """Maximise mean-variance utility directly under the no-borrowing constraint.

    `MeanVariancePortfolio` finds the tangency portfolio and only then scales it
    by y = (E[r] - rf) / (gamma * var), clipping y at 1.  Where the
    unconstrained y exceeds 1 the clipped result need not be the
    highest-utility portfolio available, so this solves

        max_w  w'(mu - rf) - (gamma / 2) w' Sigma w
        s.t.   0 <= w_i <= cap,  sum(w) <= 1

    which is concave and therefore gives the true constrained optimum.  A
    comparison policy for the backtest, not a replacement.  `weights` already
    includes the stock / risk-free split.
    """

    def __init__(
        self,
        expected_return: pd.Series,
        cov: pd.DataFrame,
        rf: float,
        cap: float = 0.10,
        gamma: float = 4.0
    ):
        permnos = expected_return.index.intersection(cov.index)
        self.mu = expected_return.loc[permnos]
        self.cov = cov.loc[permnos, permnos]
        self.rf = rf
        self.cap = cap
        self.gamma = gamma
        self.weights = self.fit()
        self.risky_share = float(self.weights.sum())
        self.tangency = (
            self.weights / self.risky_share if self.risky_share > 0
            else self.weights
        )

    def fit(self):
        """Solve the constrained utility maximisation."""
        excess = (self.mu - self.rf).to_numpy()
        cov = self.cov.to_numpy()
        n_assets = len(excess)
        if n_assets * self.cap < 1:
            raise ValueError('cap too small for the number of stocks')

        def neg_utility(w):
            value = w @ excess - 0.5 * self.gamma * (w @ cov @ w)
            grad = -(excess - self.gamma * (cov @ w))
            return -value, grad

        res = opt.minimize(
            fun=neg_utility,
            x0=np.full(n_assets, min(self.cap, 1.0 / n_assets)),
            jac=True,
            method='SLSQP',
            bounds=[(0, self.cap)] * n_assets,
            constraints={'type': 'ineq', 'fun': lambda w: 1.0 - w.sum()},
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        if not res.success:
            raise RuntimeError(res.message)

        w = np.where(res.x < 1e-6, 0, res.x)
        return pd.Series(w, index=self.mu.index, name='weight')


class Backtest:
    """Rebuild the portfolio every month out of sample and measure what it earned.

    `cov_years` is the length of the rolling covariance window; `policy` selects
    'tangency' (the team's portfolio module) or 'utility' (the class above).
    """

    def __init__(
        self,
        data_dir: Path,
        outputs_dir: Path,
        rf: float,
        capital: float = 10_000_000,
        cap: float = 0.10,
        gamma: float = 4.0,
        cov_years: int = 3,
        policy: str = 'tangency',
        start: str = '2021-01-01',
        end: str = '2025-12-01'
    ):
        if policy not in ('tangency', 'utility'):
            raise ValueError("policy must be 'tangency' or 'utility'")
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path(outputs_dir)
        self.rf = rf
        self.capital = capital
        self.cap = cap
        self.gamma = gamma
        self.cov_years = cov_years
        self.policy = policy
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)

        self.forecasts = self.get_forecasts()
        self.returns, self.market, self.exit_month = self.get_realised()
        self.daily = self.get_daily()
        self.months = [
            m for m in self.forecasts.index.get_level_values('Month').unique()
            if self.start <= m <= self.end
        ]
        self.weights = {}
        self.diagnostics = []

    # inputs
    def get_forecasts(self):
        """Expected returns by (Month, PERMNO); Month is the month being predicted."""
        df = pd.read_csv(
            str(self.outputs_dir / 'expected_return.csv'), parse_dates=['Month']
        )
        return df.set_index(['Month', 'PERMNO'])['expected_return'].sort_index()

    def get_realised(self):
        """Realised stock returns, the market benchmark and each stock's exit month."""
        df = pd.read_csv(
            str(self.outputs_dir / 'stock_monthly.csv'),
            usecols=['PERMNO', 'Month', 'Stock_ret', 'Exit_flag', 'Market_ret'],
            parse_dates=['Month']
        )
        returns = df.pivot(index='Month', columns='PERMNO', values='Stock_ret')
        market = df.groupby('Month')['Market_ret'].first()
        exits = df.loc[df['Exit_flag'].astype(bool)]
        if exits['PERMNO'].duplicated().any():
            raise ValueError('A stock has more than one exit month.')
        exit_month = exits.set_index('PERMNO')['Month']
        return returns, market, exit_month

    def get_daily(self):
        """Daily returns as a date x PERMNO panel, read and pivoted once so the
        sixty covariance estimates slice it instead of re-reading 55 MB."""
        df = pd.read_csv(
            str(self.data_dir / 'stock_daily_returns.csv'),
            usecols=['PERMNO', 'DlyCalDt', 'DlyRet'],
            parse_dates=['DlyCalDt']
        )
        return df.pivot(index='DlyCalDt', columns='PERMNO', values='DlyRet').sort_index()

    def tradeable(self, month):
        """Stocks with a forecast for `month` that have not already been delisted.

        Excluded when the exit month is strictly earlier than `month`; a stock
        leaving during `month` is still held and earns its terminal return."""
        permnos = self.forecasts.loc[month].index
        gone = self.exit_month.reindex(permnos)
        return permnos[~(gone < month)]

    def weights_for(self, month):
        """Optimal weights for `month`, estimated on data ending before it."""
        permnos = self.tradeable(month)
        window_start = month - pd.DateOffset(years=self.cov_years)
        window_end = month - pd.Timedelta(days=1)

        cov_model = CovarianceModel(
            self.data_dir,
            permnos,
            start=str(window_start.date()),
            end=str(window_end.date()),
            panel=self.daily
        )
        mu = self.forecasts.loc[month].reindex(cov_model.cov.index)
        if mu.isna().any():
            raise ValueError(f'{month:%Y-%m}: missing forecasts after the covariance filter.')

        builder = MeanVariancePortfolio if self.policy == 'tangency' else MaxUtilityPortfolio
        portfolio = builder(mu, cov_model.cov, rf=self.rf, cap=self.cap, gamma=self.gamma)
        weights = portfolio.weights[portfolio.weights > 0]

        self.diagnostics.append({
            'Month': month,
            'n_forecast': len(permnos),
            'n_estimable': len(cov_model.cov),
            'n_held': int(len(weights)),
            'risky_share': float(weights.sum()),
            'max_weight': float(weights.max()) if len(weights) else 0.0,
            'shrinkage': float(cov_model.shrinkage)
        })
        return weights

    #the backtest
    def run(self, verbose: bool = True):
        """Realised monthly returns of the strategy and of both benchmarks.

        Re-optimising sixty times takes a few minutes, so progress is printed."""
        self.weights = {}
        self.diagnostics = []
        records = []
        previous = pd.Series(dtype=float)
        previous_month = None

        for i, month in enumerate(self.months, start=1):
            weights = self.weights_for(month)
            self.weights[month] = weights

            realised = self.returns.loc[month].reindex(weights.index)
            if realised.isna().any():
                missing = realised[realised.isna()].index.tolist()
                raise ValueError(
                    f'{month:%Y-%m}: held stocks without a realised return: {missing}. '
                    'Investigate before averaging -- a silently dropped holding '
                    'overstates the strategy return.'
                )

            risky_share = float(weights.sum())
            stock_return = float(weights @ realised)
            cash_return = (1.0 - risky_share) * self.rf
            total = stock_return + cash_return

            turnover = self.turnover(weights, previous, previous_month)

            universe = self.tradeable(month)
            benchmark = self.returns.loc[month].reindex(universe)
            if benchmark.isna().any():
                raise ValueError(
                    f'{month:%Y-%m}: {int(benchmark.isna().sum())} stocks in the '
                    'equal-weight universe have no realised return.'
                )
            equal_weight = float(benchmark.mean())

            records.append({
                'Month': month,
                'strategy_return': total,
                'stock_contribution': stock_return,
                'cash_contribution': cash_return,
                'risky_share': risky_share,
                'n_held': int(len(weights)),
                'turnover': turnover,
                'sp500_return': float(self.market.loc[month]),
                'equal_weight_return': equal_weight
            })
            previous = weights
            previous_month = month
            if verbose:
                print(
                    f'  [{i:>2}/{len(self.months)}] {month:%Y-%m}  '
                    f'held {len(weights):>3}  y {risky_share:5.3f}  '
                    f'return {total:+7.2%}',
                    flush=True
                )

        results = pd.DataFrame(records).set_index('Month')
        results['wealth'] = self.capital * (1 + results['strategy_return']).cumprod()
        results['sp500_wealth'] = self.capital * (1 + results['sp500_return']).cumprod()
        results['cumulative_return'] = results['wealth'] / self.capital - 1
        self.results = results
        return results

    def turnover(self, weights, previous, previous_month):
        """One-way turnover at this rebalance, cash included.

        Last month's holdings are grown by LAST month's returns and the idle
        cash at rf, then divided by the resulting wealth.  The risky share
        drifts with them, so it must not be renormalised back to its old level.
        The cash leg counts too: moving wealth between stocks and cash is a
        trade.  The first month buys the whole risky share out of cash.
        """
        if previous_month is None:
            return float(weights.sum())
        realised = self.returns.loc[previous_month].reindex(previous.index).fillna(0)
        grown = previous * (1 + realised)
        wealth = grown.sum() + (1 - previous.sum()) * (1 + self.rf)
        drifted = grown / wealth
        both = weights.index.union(drifted.index)
        stock_leg = (
            weights.reindex(both, fill_value=0) - drifted.reindex(both, fill_value=0)
        ).abs().sum()
        cash_leg = abs(weights.sum() - drifted.sum())
        return float((stock_leg + cash_leg) / 2)

   #reporting
    def performance(self, returns: pd.Series):
        """Annualised statistics.  Each Sharpe ratio sits next to the excess
        return that produced it, so either can be checked by dividing two
        printed numbers."""
        n_months = len(returns)
        geometric = (1 + returns).prod() ** (MONTHS_PER_YEAR / n_months) - 1
        arithmetic = returns.mean() * MONTHS_PER_YEAR
        volatility = returns.std() * np.sqrt(MONTHS_PER_YEAR)
        rf_geometric = (1 + self.rf) ** MONTHS_PER_YEAR - 1
        rf_arithmetic = MONTHS_PER_YEAR * self.rf
        wealth = (1 + returns).cumprod()
        return pd.Series({
            'annualised_return_geometric': geometric,
            'annualised_return_arithmetic': arithmetic,
            'annualised_volatility': volatility,
            'excess_return_geometric': geometric - rf_geometric,
            'excess_return_arithmetic': arithmetic - rf_arithmetic,
            'sharpe_geometric': (geometric - rf_geometric) / volatility,
            'sharpe_arithmetic': (arithmetic - rf_arithmetic) / volatility,
            'max_drawdown': float((wealth / wealth.cummax() - 1).min()),
            'best_month': returns.max(),
            'worst_month': returns.min(),
            'positive_months': float((returns > 0).mean()),
            'n_months': n_months
        })

    def summary(self):
        """Strategy against the S&P 500 and an equally weighted universe."""
        table = pd.DataFrame({
            'strategy': self.performance(self.results['strategy_return']),
            'sp500': self.performance(self.results['sp500_return']),
            'equal_weight': self.performance(self.results['equal_weight_return'])
        })
        active = self.results['strategy_return'] - self.results['sp500_return']
        table.loc['tracking_error', 'strategy'] = active.std() * np.sqrt(MONTHS_PER_YEAR)
        table.loc['information_ratio', 'strategy'] = (
            active.mean() * MONTHS_PER_YEAR / (active.std() * np.sqrt(MONTHS_PER_YEAR))
        )
        table.loc['average_turnover', 'strategy'] = self.results['turnover'].mean()
        table.loc['average_risky_share', 'strategy'] = self.results['risky_share'].mean()
        return table

    def market_regression(self):
        """CAPM regression of strategy on market excess returns: did it earn more
        than its market exposure alone explains?  HC3 standard errors."""
        y = self.results['strategy_return'] - self.rf
        x = sm.add_constant(self.results['sp500_return'] - self.rf)
        fit = sm.OLS(y, x).fit(cov_type='HC3')
        alpha, beta = fit.params.iloc[0], fit.params.iloc[1]
        return pd.Series({
            'alpha_monthly': alpha,
            'alpha_annualised': (1 + alpha) ** MONTHS_PER_YEAR - 1,
            'alpha_tstat': fit.tvalues.iloc[0],
            'alpha_pvalue': fit.pvalues.iloc[0],
            'beta': beta,
            'beta_tstat': fit.tvalues.iloc[1],
            'r_squared': fit.rsquared,
            'n_months': int(fit.nobs)
        })

    def save(self, outputs_dir: Path | None = None, suffix: str = ''):
        """Write the monthly series, the summary, the regression and the weights."""
        outputs_dir = Path(outputs_dir or self.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(str(outputs_dir / f'backtest_returns{suffix}.csv'))
        self.summary().to_csv(str(outputs_dir / f'backtest_summary{suffix}.csv'))
        self.market_regression().to_csv(
            str(outputs_dir / f'backtest_market_regression{suffix}.csv'),
            header=['value']
        )
        pd.DataFrame(self.diagnostics).set_index('Month').to_csv(
            str(outputs_dir / f'backtest_diagnostics{suffix}.csv')
        )
        weights = pd.concat(
            [w.rename('weight').rename_axis('PERMNO').reset_index().assign(Month=month)
             for month, w in self.weights.items()],
            ignore_index=True
        )
        weights[['Month', 'PERMNO', 'weight']].to_csv(
            str(outputs_dir / f'backtest_weights{suffix}.csv'), index=False
        )


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    rf_monthly = 1.01 ** (1 / 12) - 1
    backtest = Backtest(root / 'data', root / 'outputs', rf=rf_monthly)
    backtest.run()
    backtest.save()
    print(backtest.summary().round(4))
    print()
    print(backtest.market_regression().round(4))

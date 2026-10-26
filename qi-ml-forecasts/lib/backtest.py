"""
Out-of-sample backtest of portfolios built from the OLS-3, OLS and RF return forecasts.

Every month from Jan 2023 to Dec 2025 the portfolio is rebuilt using only information available at the
start of that month:
    - expected returns: the test-set forecasts in outputs/test_predictions.csv. The models were estimated on
      target months up to Dec 2022 and predict month t+1 from characteristics observed in month t.
    - covariance: Ledoit-Wolf on the 3 years of daily returns that end the day before the holding month.
    - risk-free: the Fama-French T-bill return for the holding month (known at the start of the month).

Two ways to turn forecasts into a portfolio, both long-only:
    1. Mean-variance (main strategy): tangency portfolio with a 10% cap per stock, then the risky share
       y = (E[r] - rf) / (gamma * var), as in Part II.
    2. GKX decile sorts: each month sort stocks into 10 groups on the forecast. Decile 10 (highest forecast)
       held equally weighted is a simple long-only alternative. The 10-1 spread needs shorting, so it is only
       reported as a check of how well the forecasts rank stocks, not as a strategy we could hold.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib.covariance_model import load_daily_panel, estimate_covariance
from lib.portfolio import MeanVariancePortfolio

MODELS = {'OLS-3': 'pred_ols3', 'OLS': 'pred_ols', 'Random Forest': 'pred_rf'}
CAP = 0.10
GAMMA = 4
COV_YEARS = 3
N_DECILES = 10


def load_inputs(outputs_dir):
    """
    Test-set forecasts with the realized total return, market cap at formation and the monthly
    market and risk-free returns
    """
    predictions = pd.read_csv(outputs_dir / 'test_predictions.csv', parse_dates = ['Month', 'Target_month'])
    stocks = pd.read_csv(
        outputs_dir / 'stock_monthly.csv',
        usecols = ['PERMNO', 'Month', 'MthCap', 'market_excess', 'Rf'],
        parse_dates = ['Month']
    )
    market = stocks.groupby('Month')[['market_excess', 'Rf']].first()

    # market cap at the end of the formation month, used for value weights
    predictions = predictions.merge(
        stocks[['PERMNO', 'Month', 'MthCap']], on = ['PERMNO', 'Month'], how = 'left', validate = 'one_to_one'
    )
    predictions['Rf'] = predictions['Target_month'].map(market['Rf'])
    if predictions['Rf'].isna().any():
        raise ValueError('Missing risk-free rate for a holding month')
    # realized total return in the holding month
    predictions['Realized_return'] = predictions['Target_excess_return'] + predictions['Rf']
    return predictions, market


def turnover(weights, previous, previous_returns, rf):
    """
    One-way turnover at a rebalance, cash included.

    Last month's holdings are first grown by last month's returns (and the cash by rf), so a stock that
    went up is not counted as a trade. The first month buys the whole risky share out of cash.
    """
    if previous is None:
        return float(weights.sum())
    grown = previous * (1 + previous_returns.reindex(previous.index).fillna(0))
    wealth = grown.sum() + (1 - previous.sum()) * (1 + rf)
    drifted = grown / wealth
    both = weights.index.union(drifted.index)
    stock_leg = (weights.reindex(both, fill_value = 0) - drifted.reindex(both, fill_value = 0)).abs().sum()
    cash_leg = abs(weights.sum() - drifted.sum())
    return float((stock_leg + cash_leg) / 2)


def mean_variance_backtest(predictions, panel, column, cap = CAP, gamma = GAMMA, cov_years = COV_YEARS, verbose = True):
    """
    Rebuild the long-only mean-variance portfolio every month from one model's forecasts.

    Returns:
        monthly results, held weights per month, diagnostics per month
    """
    records, weights_out, diagnostics = [], [], []
    previous, previous_returns, previous_rf = None, None, None

    for month, group in predictions.groupby('Target_month', sort = True):
        group = group.set_index('PERMNO')
        rf = float(group['Rf'].iloc[0])

        window_start = month - pd.DateOffset(years = cov_years)
        window_end = month - pd.Timedelta(days = 1)
        cov, shrinkage = estimate_covariance(panel, group.index, window_start, window_end)

        # the models forecast excess returns, so add rf to get the expected total return
        mu = (group[column] + rf).reindex(cov.index)
        portfolio = MeanVariancePortfolio(mu, cov, rf = rf, cap = cap, gamma = gamma)
        weights = portfolio.weights[portfolio.weights > 0]

        realized = group['Realized_return'].reindex(weights.index)
        if realized.isna().any():
            raise ValueError(f'{month:%Y-%m}: held stocks without a realized return')

        risky_share = float(weights.sum())
        stock_part = float(weights @ realized)
        total = stock_part + (1 - risky_share) * rf

        records.append({
            'Month': month,
            'return': total,
            'excess_return': total - rf,
            'risky_share': risky_share,
            'n_held': len(weights),
            'turnover': turnover(weights, previous, previous_returns, previous_rf),
        })
        diagnostics.append({
            'Month': month,
            'n_forecast': len(group),
            'n_with_covariance': len(cov),
            'n_held': len(weights),
            'risky_share': risky_share,
            'max_weight': float(weights.max()),
            'ex_ante_excess_return': float(portfolio.stats(portfolio.tangency)[0] - rf),
            'ex_ante_volatility': float(portfolio.stats(portfolio.tangency)[1]),
            'shrinkage': float(shrinkage),
        })
        weights_out.append(weights.rename('weight').rename_axis('PERMNO').reset_index().assign(Month = month))

        previous, previous_returns, previous_rf = weights, group['Realized_return'], rf
        if verbose:
            print(f'  {month:%Y-%m}  held {len(weights):>3}  y {risky_share:5.3f}  return {total:+7.2%}', flush = True)

    return (
        pd.DataFrame(records).set_index('Month'),
        pd.concat(weights_out, ignore_index = True)[['Month', 'PERMNO', 'weight']],
        pd.DataFrame(diagnostics).set_index('Month'),
    )


def assign_deciles(predictions, column):
    """
    Decile 1 = lowest forecast, 10 = highest, re-sorted every month
    """
    ranks = predictions.groupby('Target_month')[column].rank(method = 'first')
    sizes = predictions.groupby('Target_month')[column].transform('size')
    return np.ceil(ranks / sizes * N_DECILES).astype(int)


def decile_returns(predictions, column):
    """
    Equal- and value-weighted realized return of every forecast decile in every month (GKX Table 7 style)
    """
    df = predictions.copy()
    df['Decile'] = assign_deciles(df, column)
    df['Forecast'] = df[column]
    df['vw'] = df['MthCap'] * df['Realized_return']

    grouped = df.groupby(['Target_month', 'Decile'])
    out = pd.DataFrame({
        'Forecast': grouped['Forecast'].mean(),
        'EW_return': grouped['Realized_return'].mean(),
        'VW_return': grouped['vw'].sum() / grouped['MthCap'].sum(),
        'Rf': grouped['Rf'].first(),
        'N': grouped.size(),
    })
    return out.reset_index()


def rank_ic(predictions, column):
    """
    Monthly Spearman correlation between forecast and realized return: does the model rank stocks well?
    """
    return predictions.groupby('Target_month').apply(
        lambda g: g[column].corr(g['Target_excess_return'], method = 'spearman')
    )


def performance(excess, market_excess, total = None):
    """
    Annualized statistics of a monthly return series plus a CAPM regression on the market (HC3 errors).
    For a zero-cost spread (10-1) the excess return is the return itself and total is left out.
    """
    stats = {
        'mean_excess_return': excess.mean() * 12,
        'volatility': excess.std() * np.sqrt(12),
        'sharpe_ratio': excess.mean() / excess.std() * np.sqrt(12),
        't_stat_mean': excess.mean() / (excess.std() / np.sqrt(len(excess))),
    }
    if total is not None:
        wealth = (1 + total).cumprod()
        stats['geometric_return'] = wealth.iloc[-1] ** (12 / len(total)) - 1
        stats['max_drawdown'] = float((wealth / wealth.cummax() - 1).min())
    fit = sm.OLS(excess, sm.add_constant(market_excess.reindex(excess.index))).fit(cov_type = 'HC3')
    stats.update({
        'alpha_annual': fit.params.iloc[0] * 12,
        'alpha_t_stat': fit.tvalues.iloc[0],
        'beta': fit.params.iloc[1],
        'n_months': len(excess),
    })
    return pd.Series(stats)


def run_backtest(outputs_dir: Path, data_dir: Path, verbose = True):
    """
    Backtest all three models and the benchmarks over the test period, and save the tables.

    Returns:
        summary table (strategies in rows) and decile table
    """
    predictions, market = load_inputs(outputs_dir)
    # no filter on the 2025 firm list here: stocks that dropped out before 2025 (e.g. WW, bankrupt) stay in,
    # otherwise the backtest would only contain survivors
    panel = load_daily_panel(data_dir)
    months = sorted(predictions['Target_month'].unique())
    mkt_excess = market['market_excess'].reindex(months)
    rf = market['Rf'].reindex(months)

    returns = pd.DataFrame(index = pd.DatetimeIndex(months, name = 'Month'))
    returns['Market'] = mkt_excess + rf
    # equal weight over the same universe the models choose from
    returns['Equal weight universe'] = predictions.groupby('Target_month')['Realized_return'].mean()

    market_row = performance(mkt_excess, mkt_excess, returns['Market'])
    # regressing the market on itself gives alpha 0 and a meaningless t-stat
    market_row[['alpha_annual', 'alpha_t_stat']] = np.nan
    summary = {
        'Market': market_row,
        'Equal weight universe': performance(returns['Equal weight universe'] - rf, mkt_excess, returns['Equal weight universe']),
    }
    extra = {}
    decile_tables, all_weights, all_diagnostics = [], [], []

    for name, column in MODELS.items():
        if verbose:
            print(f'{name}: mean-variance backtest')
        results, weights, diagnostics = mean_variance_backtest(predictions, panel, column, verbose = verbose)
        returns[f'{name} MV'] = results['return']
        summary[f'{name} MV'] = performance(results['excess_return'], mkt_excess, results['return'])
        extra[f'{name} MV'] = {'avg_turnover': results['turnover'].mean(), 'avg_risky_share': results['risky_share'].mean(),
                               'avg_n_held': results['n_held'].mean()}
        all_weights.append(weights.assign(Model = name))
        all_diagnostics.append(diagnostics.assign(Model = name))

        deciles = decile_returns(predictions, column)
        top = deciles.loc[deciles['Decile'] == N_DECILES].set_index('Target_month')
        bottom = deciles.loc[deciles['Decile'] == 1].set_index('Target_month')
        returns[f'{name} top decile EW'] = top['EW_return']
        summary[f'{name} top decile EW'] = performance(top['EW_return'] - top['Rf'], mkt_excess, top['EW_return'])
        for weighting in ['EW', 'VW']:
            spread = top[f'{weighting}_return'] - bottom[f'{weighting}_return']
            returns[f'{name} 10-1 {weighting}'] = spread
            summary[f'{name} 10-1 {weighting} (needs shorting)'] = performance(spread, mkt_excess)

        ic = rank_ic(predictions, column)
        table = deciles.groupby('Decile')[['Forecast', 'EW_return', 'VW_return']].mean() * 12
        table['EW_sharpe'] = deciles.assign(x = deciles['EW_return'] - deciles['Rf']).groupby('Decile')['x'].agg(
            lambda x: x.mean() / x.std() * np.sqrt(12))
        table = table.reset_index().assign(Model = name, Mean_rank_IC = ic.mean(), Rank_IC_t_stat = ic.mean() / ic.std() * np.sqrt(len(ic)))
        decile_tables.append(table)

    summary = pd.DataFrame(summary).T
    for name, values in extra.items():
        for key, value in values.items():
            summary.loc[name, key] = value
    deciles = pd.concat(decile_tables, ignore_index = True)

    summary.to_csv(outputs_dir / 'backtest_summary.csv')
    returns.to_csv(outputs_dir / 'backtest_returns.csv')
    deciles.to_csv(outputs_dir / 'backtest_deciles.csv', index = False)
    pd.concat(all_weights, ignore_index = True).to_csv(outputs_dir / 'backtest_weights.csv', index = False)
    pd.concat(all_diagnostics).to_csv(outputs_dir / 'backtest_diagnostics.csv')
    plot_wealth(returns, outputs_dir / 'backtest_wealth.png')
    return summary, deciles


def plot_wealth(returns, path):
    """
    Growth of 1 euro for the long-only strategies and the benchmarks
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    columns = ['Market', 'Equal weight universe', 'OLS-3 MV', 'OLS MV', 'Random Forest MV', 'Random Forest top decile EW']
    styles = {'Market': ('#1f1f1f', '-'), 'Equal weight universe': ('#8c8c8c', '--'), 'OLS-3 MV': ('#4c78a8', '-'),
              'OLS MV': ('#72b7b2', '-'), 'Random Forest MV': ('#e45756', '-'), 'Random Forest top decile EW': ('#e45756', ':')}
    wealth = (1 + returns[columns]).cumprod()
    start = wealth.index[0] - pd.DateOffset(months = 1)
    wealth.loc[start] = 1.0
    wealth = wealth.sort_index()

    fig, ax = plt.subplots(figsize = (8, 4.5))
    for column in columns:
        color, line = styles[column]
        ax.plot(wealth.index, wealth[column], color = color, linestyle = line, linewidth = 1.6, label = column)
    ax.set_ylabel('Value of 1 invested in Jan 2023')
    ax.set_title('Out-of-sample backtest, Jan 2023 - Dec 2025 (long-only)')
    ax.grid(alpha = 0.3)
    ax.legend(frameon = False, fontsize = 8)
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    summary, deciles = run_backtest(root / 'outputs', root / 'data')
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string())
    print(deciles.round(4).to_string(index = False))

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

DAYS_PER_MONTH = 21

def load_daily_panel(data_dir: Path):
    """
    Daily returns as a date x PERMNO table (read and pivoted once, then sliced every month)
    """
    df = pd.read_csv(
        data_dir / 'crsp_daily_2016_2025_reused.csv',
        usecols = ['PERMNO', 'DlyCalDt', 'DlyRet'],
        parse_dates = ['DlyCalDt']
    )
    return df.pivot(index = 'DlyCalDt', columns = 'PERMNO', values = 'DlyRet').sort_index()

def estimate_covariance(panel, permnos, start, end):
    """
    Monthly covariance matrix with Ledoit-Wolf shrinkage, same as in Part II.

    Uses daily returns between start and end. Ledoit-Wolf needs a full panel, so stocks with any
    missing day in the window are dropped. Daily covariance is scaled by 21 trading days.

    Returns:
        Covariance DataFrame indexed by PERMNO and the shrinkage intensity
    """
    returns = panel.loc[start:end, panel.columns.intersection(permnos)]
    returns = returns.loc[:, returns.notna().all()]
    lw = LedoitWolf().fit(returns.to_numpy())
    cov = pd.DataFrame(lw.covariance_ * DAYS_PER_MONTH, index = returns.columns, columns = returns.columns)
    return cov, lw.shrinkage_

def condition_numbers(panel, permnos, start, end):
    """
    Compare how stable the sample and the shrunk covariance matrices are
    """
    returns = panel.loc[start:end, panel.columns.intersection(permnos)]
    returns = returns.loc[:, returns.notna().all()]
    sample = np.cov(returns.to_numpy(), rowvar = False) * DAYS_PER_MONTH
    shrunk, _ = estimate_covariance(panel, permnos, start, end)
    return pd.Series({
        'sample': np.linalg.cond(sample),
        'ledoit_wolf': np.linalg.cond(shrunk.to_numpy())
    })

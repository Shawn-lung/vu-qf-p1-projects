import numpy as np
import pandas as pd
import scipy.optimize as opt


class MeanVariancePortfolio:
    """
    Long-only tangency portfolio, then split wealth between it and the risk-free asset.
    Same construction as Part II so the three parts are comparable.

    Inputs:
        expected_return    monthly expected simple returns indexed by PERMNO
        cov                monthly covariance matrix indexed by PERMNO
        rf                 monthly risk-free rate
        cap                maximum weight per stock inside the tangency portfolio
        gamma              risk aversion used for the risky share

    Outputs:
        tangency weights, risky share y and final weights y * tangency
    """

    def __init__(self, expected_return: pd.Series, cov: pd.DataFrame, rf: float, cap: float = 0.10, gamma: float = 4.0):
        permnos = expected_return.index.intersection(cov.index)
        self.mu = expected_return.loc[permnos]
        self.cov = cov.loc[permnos, permnos]
        self.rf = rf
        self.cap = cap
        self.gamma = gamma

        self.tangency = self.fit_tangency()
        self.risky_share = self.fit_risky_share()
        self.weights = (self.risky_share * self.tangency).rename('weight')

    @staticmethod
    def neg_sharpe(w, excess, cov):
        """
        Negative Sharpe ratio and its gradient for the optimizer
        """
        mean = w @ excess
        vol = np.sqrt(w @ cov @ w)
        grad = -(excess / vol - mean * (cov @ w) / vol ** 3)
        return -mean / vol, grad

    def fit_tangency(self):
        """
        Maximize the Sharpe ratio with no short sales (w >= 0) and a cap per stock
        """
        excess = (self.mu - self.rf).to_numpy()
        cov = self.cov.to_numpy()
        n_assets = len(excess)
        if n_assets * self.cap < 1:
            raise ValueError('cap too small for the number of stocks')

        res = opt.minimize(
            fun = self.neg_sharpe,
            x0 = np.ones(n_assets) / n_assets,
            args = (excess, cov),
            jac = True,
            method = 'SLSQP',
            bounds = [(0, self.cap)] * n_assets,
            constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1},
            options = {'ftol': 1e-12, 'maxiter': 1000}
        )
        if not res.success:
            raise RuntimeError(res.message)

        # tiny weights are numerical noise
        w = np.where(res.x < 1e-6, 0, res.x)
        return pd.Series(w / w.sum(), index = self.mu.index, name = 'tangency_weight')

    def stats(self, w: pd.Series):
        """
        Monthly expected return and volatility of a stock portfolio
        """
        w = w.reindex(self.mu.index, fill_value = 0)
        return w @ self.mu, np.sqrt(w @ self.cov @ w)

    def fit_risky_share(self):
        """
        Optimal share in the tangency portfolio, y = (E[r] - rf) / (gamma * var), between 0 and 1 (no borrowing)
        """
        mean, vol = self.stats(self.tangency)
        return float(np.clip((mean - self.rf) / (self.gamma * vol ** 2), 0, 1))

    def summary(self):
        """
        Annualized ex-ante statistics of the tangency and the final portfolio
        """
        rows = {}
        for name, w, rf_share in [('tangency', self.tangency, 0.0), ('final', self.weights, 1 - self.risky_share)]:
            mean, vol = self.stats(w)
            mean += rf_share * self.rf
            rows[name] = {
                'expected_return': 12 * mean,
                'volatility': np.sqrt(12) * vol,
                'sharpe_ratio': np.sqrt(12) * (mean - self.rf) / vol if vol > 0 else np.nan,
                'n_stocks': int((w > 0).sum()),
                'max_weight': w.max(),
                'risky_share': 1 - rf_share
            }
        return pd.DataFrame(rows)

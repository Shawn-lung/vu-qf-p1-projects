import numpy as np
import scipy as sc

from lib.callmodel import CallModel


class ExtendedModel(CallModel):
    # same model as CallModel, we only add the things we need for (a), (g) and (i)

    def put_price(self, vS, dT):
        # BS put price, easiest way is through put-call parity
        return self.call_price(vS, dT) - vS + self.iK * np.exp(-self.dR * dT)

    def gamma(self, vS, dT):
        # BS gamma of 1 option (same formula for a call and a put)
        vD1 = (
            np.log(vS / self.iK) + (self.dR + self.dStd**2 / 2) * dT
        ) / (np.sqrt(dT) * self.dStd)
        return sc.stats.norm.pdf(vD1) / (vS * self.dStd * np.sqrt(dT))

    def gamma_position(self, mS, dT, iContracts=-20, iSize=100):
        # gamma of the whole position on every hedging date (rows = paths, columns = dates)
        # we leave out the last column (maturity) because gamma is not defined at T
        iStep = mS.shape[1] - 1
        dDeltaT = dT / iStep
        vRemainingT = dT - np.arange(iStep) * dDeltaT
        return iContracts * iSize * self.gamma(mS[:, :iStep], vRemainingT)

    def gamma_pnl_approximation(self, mS, dT, iContracts=-20, iSize=100):
        # check for (f): per step the hedging error should be roughly
        # 0.5 * gamma * S^2 * (return^2 - sigma^2 * dt), so we add that up along each path
        iStep = mS.shape[1] - 1
        dDeltaT = dT / iStep
        mGamma = self.gamma_position(mS, dT, iContracts, iSize)
        mReturn = mS[:, 1:] / mS[:, :iStep] - 1
        return (
            0.5 * mGamma * mS[:, :iStep] ** 2
            * (mReturn**2 - self.dStd**2 * dDeltaT)
        ).sum(axis=1)

    def put_risk_neutral(self, iS, dT, iN=10000, rng=None):
        # part (i): simulate S(T) under Q, so the drift is r and not mu
        # the payoff only depends on S(T), so we can jump to T in one step
        rng = self.rng if rng is None else rng
        vE = rng.normal(0, 1, iN)
        vST = iS * np.exp(
            (self.dR - self.dStd**2 / 2) * dT
            + self.dStd * np.sqrt(dT) * vE
        )
        # price = average discounted payoff, s.e. = std / sqrt(n)
        vDiscPayoff = np.exp(-self.dR * dT) * np.maximum(self.iK - vST, 0)
        dPrice = vDiscPayoff.mean()
        dStdError = vDiscPayoff.std(ddof=1) / np.sqrt(iN)
        return dPrice, dStdError

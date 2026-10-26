import numpy as np
import scipy as sc


class CallModel:
    def __init__(self, dStd, iK, dAnR, rng):
        self.dStd = dStd
        self.iK = iK
        self.dR = 4 * np.log(1 + dAnR / 4)
        self.rng = rng

    def call_price(self, vS, dT):
        vD1 = (
            np.log(vS / self.iK) + (self.dR + self.dStd**2 / 2) * dT
        ) / (np.sqrt(dT) * self.dStd)
        vD2 = vD1 - np.sqrt(dT) * self.dStd

        vC = (
            vS * sc.stats.norm.cdf(vD1)
            - self.iK * np.exp(-self.dR * dT) * sc.stats.norm.cdf(vD2)
        )
        return vC

    def path_simulation(self, iS, dMu, dT, iStep, iN = 10000):
        time_label = {
            "13": "weekly",
            "3": "monthly",
            "63": "daily",
            "252": "four_times_daily"
        }[str(iStep)]
        
        print(f"Simulating Path for {time_label} with MU = {dMu}...")

        dDeltaT = dT / iStep
        mE = self.rng.normal(0, 1, (iN, iStep))
        mLogS = np.cumsum(
            (dMu - self.dStd**2 / 2) * dDeltaT
            + self.dStd * np.sqrt(dDeltaT) * mE,
            axis=1
        )
        
        return iS * np.exp(np.hstack([np.zeros((iN, 1)), mLogS]))

    def delta(self, vS, dT):
        vD1 = (
            np.log(vS / self.iK) + (self.dR + self.dStd**2 / 2) * dT
        ) / (np.sqrt(dT) * self.dStd)
        return sc.stats.norm.cdf(vD1)

    def hedging(self, mS, dT, iContracts = -20, iSize = 100):
        lDelta = []
        lCash = []
        lStockN = []
        mCall = np.zeros_like(mS, dtype=float)
        mHedge = np.zeros_like(mS, dtype=float)
        dDeltaT = dT / (mS.shape[1] - 1)
        dRemainingT = dT
        vDelta = self.delta(mS[:, 0], dRemainingT)
        vStockN = vDelta * -iContracts * iSize
        lDelta.append(vDelta)
        lCash.append(
            -iContracts * iSize * self.call_price(mS[:, 0], dT)
            - mS[:, 0] * vStockN
        )        
        lStockN.append(vStockN)
        mCall[:, 0] = -iContracts * iSize * self.call_price(mS[:, 0], dT)
        mHedge[:, 0] = lCash[0] + vStockN * mS[:, 0]
        for i in range(1, mS.shape[1] - 1):
            dRemainingT = dT - i * dDeltaT
            vDelta = self.delta(mS[:, i], dRemainingT)
            vStockN = vDelta * -iContracts * iSize
            vI = lCash[i-1] * (np.exp(self.dR*dDeltaT) - 1)
            vChangeS = vStockN - lStockN[i-1]
            vCash = lCash[i-1] + vI - vChangeS * mS[:, i]

            lDelta.append(vDelta)
            lStockN.append(vStockN)
            lCash.append(vCash)
            mCall[:, i] = -iContracts * iSize * self.call_price(mS[:, i], dRemainingT)
            mHedge[:, i] = vCash + vStockN * mS[:, i]
        
        self.mDelta = np.stack(lDelta)
        mCall[:, -1] = -iContracts * iSize * np.maximum(mS[:, -1] - self.iK, 0)
        mHedge[:, -1] = (
            lCash[-1] * np.exp(self.dR*dDeltaT)
            + lStockN[-1] * mS[:, -1]
        )
        
        vPnL = (
            lCash[-1] * np.exp(self.dR*dDeltaT) 
            + lStockN[-1] * mS[:, -1]
            + iContracts * iSize * np.maximum(mS[:, -1] - self.iK, 0)
        )
        self.mCall = mCall
        self.mHedge = mHedge
        return vPnL

    def put_replication(self, mS, dT):
        lCash = []
        lStockN = []
        mPutDelta = self.mDelta.T - 1
        dDeltaT = dT / (mS.shape[1] - 1)
        vDelta = mPutDelta[:, 0]
        vStockN = vDelta
        lStockN.append(vStockN)
        lCash.append(-mS[:,0] * vStockN)
        for i in range(1, mS.shape[1] - 1):
            vDelta = mPutDelta[:, i]
            vStockN = vDelta
            vI = lCash[i-1] * (np.exp(self.dR*dDeltaT) - 1)
            vChangeS = vStockN - lStockN[i-1]
            vCash = lCash[i-1] + vI - vChangeS * mS[:, i]

            lStockN.append(vStockN)
            lCash.append(vCash)

        vPayoff = np.maximum(self.iK - mS[:, -1], 0)
        vShortfall = (
            vPayoff - lStockN[-1] * mS[:, -1]
            - lCash[-1] * np.exp(self.dR*dDeltaT)
        ) 
        dPutPrice = np.exp(-self.dR * dT) * vShortfall.mean()
        return(dPutPrice)

    def put_call_parity(self, iS, dT, dPutPrice, tolerance = 1.0):
        dCallPrice = self.call_price(iS, dT)
        dError = (
            dCallPrice - dPutPrice
            - (iS - self.iK * np.exp(-self.dR * dT))
        )
        return abs(dError) <= tolerance, dError
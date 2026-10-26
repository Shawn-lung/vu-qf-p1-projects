import numpy as np
import pandas as pd
import scipy as sc

from lib.extensions import ExtendedModel


def main():
    df = pd.read_csv("data/SP500.csv")
    df['Return'] = df['SP500'] / df['SP500'].shift(1) - 1
    df = df.dropna()
    df['LOG_Return'] = np.log(1 + df['Return'])

    rng = np.random.default_rng(42)

    dStd = df['LOG_Return'].std() * np.sqrt(12)
    iS = 7685
    iK = 7750
    dT = 3 / 12
    dAnR = 0.03
    iUnits = 20 * 100

    # (label, mu, number of steps), same order as in main.py so we get the same paths
    lScenario = [
        ("weekly", 0.05, 13),
        ("monthly", 0.05, 3),
        ("daily", 0.05, 63),
        ("four-times-daily", 0.05, 252),
        ("daily", 0.10, 63),
        ("four-times-daily", 0.10, 252),
        ("daily", 0.20, 63),
        ("four-times-daily", 0.20, 252),
    ]

    # run every scenario and collect the P&L statistics in one table
    lRow = []
    dPutPrice = None
    for sLabel, dMu, iStep in lScenario:
        model = ExtendedModel(dStd, iK, dAnR, rng)
        mS = model.path_simulation(iS, dMu, dT, iStep)
        vPnL = model.hedging(mS, dT)
        if sLabel == "four-times-daily" and dMu == 0.05:
            # put price from (h), main.py does this with the same paths
            dPutPrice = model.put_replication(mS, dT)
            bPCP, dError = model.put_call_parity(iS, dT, dPutPrice)
        lRow.append({
            "frequency": sLabel,
            "mu": dMu,
            "steps": iStep,
            "mean": vPnL.mean(),
            "se_mean": vPnL.std(ddof=1) / np.sqrt(len(vPnL)),
            "std": vPnL.std(ddof=1),
            "skew": sc.stats.skew(vPnL),
            "q05": np.percentile(vPnL, 5),
            "q95": np.percentile(vPnL, 95),
            "min": vPnL.min(),
            "max": vPnL.max(),
            "prob_loss": (vPnL < 0).mean(),
        })

    dfOut = pd.DataFrame(lRow)
    dPremium = iUnits * model.call_price(iS, dT)
    dfOut["std_pct_premium"] = 100 * dfOut["std"] / dPremium

    # sanity check: rule of thumb for the hedging error when you hedge n times,
    # std is about sqrt(pi/4) * vega * sigma / sqrt(n)
    dD1 = (np.log(iS / iK) + (model.dR + dStd**2 / 2) * dT) / (dStd * np.sqrt(dT))
    dVega = iUnits * iS * sc.stats.norm.pdf(dD1) * np.sqrt(dT)
    dfOut["std_rule_of_thumb"] = np.sqrt(np.pi / 4) * dVega * dStd / np.sqrt(dfOut["steps"])

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"\nPremium received = {dPremium:,.2f} USD, vega of the position = {dVega:,.0f}")
    print(dfOut.to_string(index=False))

    print(f"\n(h) put price by replication = {dPutPrice:.4f}")
    print(f"    call price = {model.call_price(iS, dT):.4f}, "
          f"S - K exp(-rT) = {iS - iK * np.exp(-model.dR * dT):.4f}")
    print(f"    put-call parity holds: {bPCP}, error = {dError:.6f}")

    # extra check for (e): with 10,000 paths the mean P&L is too noisy to see if mu
    # matters, so here we use 200,000 paths per scenario (own seed, 4 chunks of 50,000
    # to keep the memory use low)
    print("\nLarge-sample check (200,000 paths per scenario):")
    rng_large = np.random.default_rng(2026)
    for iStep in (63, 252):
        for dMu in (0.05, 0.10, 0.20):
            model = ExtendedModel(dStd, iK, dAnR, rng_large)
            lPnL = []
            for _ in range(4):
                mS = model.path_simulation(iS, dMu, dT, iStep, iN=50000)
                lPnL.append(model.hedging(mS, dT))
            vPnL = np.concatenate(lPnL)
            print(f"    steps = {iStep:3d}, mu = {dMu:.0%}: mean = {vPnL.mean():9.2f} "
                  f"(s.e. {vPnL.std(ddof=1) / np.sqrt(len(vPnL)):6.2f}), std = {vPnL.std(ddof=1):10.2f}")


if __name__ == "__main__":
    main()

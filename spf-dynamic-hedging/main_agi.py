import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

from pathlib import Path
from lib.extensions import ExtendedModel


def plot_gamma(mGamma, vAvgGamma, vPnL, dMu):
    img_dir = Path(__file__).resolve().parent / 'img'
    img_dir.mkdir(exist_ok=True)

    # plot 1: histogram of the average gamma position of each path
    plt.figure(figsize=(6, 4))
    plt.hist(vAvgGamma, bins=50, edgecolor="black")
    plt.axvline(mGamma[0, 0], color="red", linestyle="--", label="Gamma position at trade date")
    plt.xlabel("Average gamma position over the path")
    plt.ylabel("Frequency")
    plt.title(f"Gamma position: short 20 calls, daily hedging\nmu = {dMu:.0%}")
    plt.legend()
    plt.savefig(str(img_dir / f"daily_mu_{dMu:.0%}_gamma_histogram.pdf"), bbox_inches="tight")
    plt.close()

    # plot 2: how the gamma position spreads out over time (median + percentile bands)
    vTime = np.arange(mGamma.shape[1])
    vP05, vP25, vP50, vP75, vP95 = np.percentile(mGamma, [5, 25, 50, 75, 95], axis=0)
    plt.figure(figsize=(6, 4))
    plt.fill_between(vTime, vP05, vP95, alpha=0.2, color="C0", label="5%-95% of paths")
    plt.fill_between(vTime, vP25, vP75, alpha=0.4, color="C0", label="25%-75% of paths")
    plt.plot(vTime, vP50, color="C0", label="Median")
    plt.xlabel("Trading day")
    plt.ylabel("Gamma position")
    plt.title(f"Gamma position through time: short 20 calls\nmu = {dMu:.0%}")
    plt.legend(loc="lower left")
    plt.savefig(str(img_dir / f"daily_mu_{dMu:.0%}_gamma_over_time.pdf"), bbox_inches="tight")
    plt.close()

    # plot 3: |P&L| against the average |gamma| of the path
    # the black line is the mean |P&L| in each gamma decile, it shows the trend better than the dots
    vAbsGamma = np.abs(vAvgGamma)
    vAbsPnL = np.abs(vPnL)
    vBin = pd.qcut(vAbsGamma, 10, labels=False)
    dfBin = pd.DataFrame({"gamma": vAbsGamma, "pnl": vAbsPnL, "bin": vBin}).groupby("bin").mean()
    plt.figure(figsize=(6, 4))
    plt.scatter(vAbsGamma, vAbsPnL, s=3, alpha=0.15, color="C0", label="Simulated path")
    plt.plot(dfBin["gamma"], dfBin["pnl"], color="black", marker="o", label="Mean per gamma decile")
    plt.xlabel("Average absolute gamma position over the path")
    plt.ylabel("Absolute terminal P&L (USD)")
    plt.title(f"Absolute P&L vs. average gamma: daily hedging\nmu = {dMu:.0%}")
    plt.legend()
    plt.savefig(str(img_dir / f"daily_mu_{dMu:.0%}_gamma_vs_abs_pnl.pdf"), bbox_inches="tight")
    plt.close()


def main():
    # same data prep as main.py: annualised vol from the monthly log returns
    df = pd.read_csv("data/SP500.csv")
    df['Return'] = df['SP500'] / df['SP500'].shift(1) - 1
    df = df.dropna()
    df['LOG_Return'] = np.log(1 + df['Return'])

    dStd = df['LOG_Return'].std() * np.sqrt(12)
    iS = 7685
    iK = 7750
    dT = 3 / 12
    dAnR = 0.03
    dMu = 0.05
    iContracts = -20    # negative because the trader is short
    iSize = 100
    iUnits = -iContracts * iSize
    iStep_weekly = 13
    iStep_monthly = 3
    iStep_daily = 63
    iStep_four_times_daily = 252

    # we use the same seed and simulate in the same order as main.py (weekly, monthly,
    # daily, four times daily), so the daily and four-times-daily paths here are
    # exactly the ones from (d) and (h). the weekly and monthly paths are not used,
    # we only simulate them to keep the random numbers in sync
    rng = np.random.default_rng(42)
    model_weekly = ExtendedModel(dStd, iK, dAnR, rng)
    model_weekly.path_simulation(iS, dMu, dT, iStep_weekly)
    model_monthly = ExtendedModel(dStd, iK, dAnR, rng)
    model_monthly.path_simulation(iS, dMu, dT, iStep_monthly)

    model_daily = ExtendedModel(dStd, iK, dAnR, rng)
    mS_daily = model_daily.path_simulation(iS, dMu, dT, iStep_daily)
    vPnL_daily = model_daily.hedging(mS_daily, dT)

    model_four_times_daily = ExtendedModel(dStd, iK, dAnR, rng)
    mS_four_times_daily = model_four_times_daily.path_simulation(iS, dMu, dT, iStep_four_times_daily)
    vPnL_four_times_daily = model_four_times_daily.hedging(mS_four_times_daily, dT)

    # ---------------- (a) premium ----------------
    dCall = model_daily.call_price(iS, dT)
    print("\n(a) Premium received by the trader")
    print(f"    sigma (annualised)          = {dStd:.4%}")
    print(f"    r (continuously compounded) = {model_daily.dR:.4%}")
    print(f"    Black-Scholes call price    = {dCall:.4f} per unit of index")
    print(f"    delta at trade date         = {model_daily.delta(iS, dT):.4f}")
    print(f"    premium for 20 contracts    = {iUnits * dCall:,.2f} USD")

    # ---------------- (g) gamma ----------------
    # gamma position on every day of every path, then the average per path
    mGamma = model_daily.gamma_position(mS_daily, dT, iContracts, iSize)
    vAvgGamma = mGamma.mean(axis=1)
    vAbsGamma = np.abs(vAvgGamma)
    vAbsPnL = np.abs(vPnL_daily)
    dPearson = sc.stats.pearsonr(vAbsGamma, vAbsPnL)[0]
    dSpearman = sc.stats.spearmanr(vAbsGamma, vAbsPnL)[0]
    vApprox = model_daily.gamma_pnl_approximation(mS_daily, dT, iContracts, iSize)

    print("\n(g) Gamma position with daily hedging")
    print(f"    gamma position at trade date = {mGamma[0, 0]:.4f}")
    print(f"    average gamma per path: mean = {vAvgGamma.mean():.4f}, std = {vAvgGamma.std(ddof=1):.4f}")
    print(f"    5% / 50% / 95% percentile    = "
          + " / ".join(f"{x:.4f}" for x in np.percentile(vAvgGamma, [5, 50, 95])))
    print(f"    most negative gamma on any path and day = {mGamma.min():.4f}")
    print(f"    correlation |avg gamma| and |P&L|: Pearson = {dPearson:.3f}, Spearman = {dSpearman:.3f}")
    print(f"    correlation gamma approximation and P&L = {sc.stats.pearsonr(vApprox, vPnL_daily)[0]:.3f}")

    # split the paths in 5 equal groups from low to high gamma and compare the P&L
    dfQ = pd.DataFrame({
        "avg_abs_gamma": vAbsGamma,
        "abs_pnl": vAbsPnL,
        "pnl": vPnL_daily,
        "quintile": pd.qcut(vAbsGamma, 5, labels=False) + 1,
    }).groupby("quintile").agg(
        avg_abs_gamma=("avg_abs_gamma", "mean"),
        mean_abs_pnl=("abs_pnl", "mean"),
        std_pnl=("pnl", "std"),
        mean_pnl=("pnl", "mean"),
    )
    print("    P&L per quintile of average absolute gamma:")
    print(dfQ.round(2).to_string())
    plot_gamma(mGamma, vAvgGamma, vPnL_daily, dMu)

    # ---------------- (i) put with risk-neutral valuation ----------------
    dPutBS = model_daily.put_price(iS, dT)
    dPutRepl = model_four_times_daily.put_replication(mS_four_times_daily, dT)
    # s.e. of the replication price from (h): the hedging error of 1 put is the same
    # as the one of 1 call on every path (put delta = call delta - 1), so we can take
    # the std of the call P&L per unit. the price is a discounted average, so we
    # discount the s.e. as well
    dSERepl = (
        np.exp(-model_four_times_daily.dR * dT)
        * vPnL_four_times_daily.std(ddof=1) / iUnits
        / np.sqrt(len(vPnL_four_times_daily))
    )

    # separate seed for the risk-neutral draws so they don't depend on the paths above
    rng_q = np.random.default_rng(2026)
    print("\n(i) Risk-neutral valuation of the put")
    print(f"    Black-Scholes put price (benchmark)   = {dPutBS:.4f}")
    print(f"    replication, (h), 10,000 x 252 steps  = {dPutRepl:.4f}  (s.e. {dSERepl:.4f})")
    # 10,000 draws like the other parts, and 2,520,000 = 10,000 x 252, which is the
    # number of random numbers the replication in (h) needed
    for iN in (10000, 2520000):
        dPutRN, dSE = model_daily.put_risk_neutral(iS, dT, iN, rng_q)
        print(f"    risk-neutral MC, {iN:>9,} draws      = {dPutRN:.4f}  (s.e. {dSE:.4f}, "
              f"95% CI [{dPutRN - 1.96 * dSE:.4f}, {dPutRN + 1.96 * dSE:.4f}])")
        # put-call parity check: C - P should be S - K exp(-rT)
        dError = dCall - dPutRN - (iS - iK * np.exp(-model_daily.dR * dT))
        print(f"        put-call parity error             = {dError:.4f}")


if __name__ == "__main__":
    main()

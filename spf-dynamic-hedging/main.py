import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from lib.callmodel import CallModel


def plot_results(rng, vPnL, mHedge, mCall, interval = "week", dMu = 0.05):
    img_dir = Path(__file__).resolve().parent / 'img'
    img_dir.mkdir(exist_ok=True)
    frequency, time_label, steps_per_unit = {
        "week": ("weekly", "Week", 1),
        "month": ("monthly", "Month", 1),
        "day": ("daily", "Trading day", 1),
        "four_times_daily": ("four-times-daily", "Trading day", 4),
    }[interval]

    plt.figure(figsize=(6, 4))
    plt.hist(vPnL, bins=50, edgecolor="black")
    plt.xlabel("Terminal P&L (USD)")
    plt.ylabel("Frequency")
    plt.title(f"3-month P&L: short 20 calls with {frequency} delta hedging\nmu = {dMu:.0%}")
    plt.axvline(0, color="red", linestyle="--")

    iPath = rng.integers(mHedge.shape[0])
    vTime = np.arange(mHedge.shape[1]) / steps_per_unit
    plt.savefig(str(img_dir / f"{frequency}_mu_{dMu:.0%}_pnl_histogram.pdf"), bbox_inches="tight")

    plt.figure(figsize=(6, 4))
    plt.plot(vTime, mHedge[iPath, :], label="Hedge portfolio")
    plt.plot(vTime, mCall[iPath, :], linestyle="--", label="Long calls (20 contracts)")
    plt.xlabel(time_label)
    plt.ylabel("Value (USD)")
    plt.title(f"{frequency.capitalize()} hedge vs. call value: random path {iPath + 1}\nmu = {dMu:.0%}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(img_dir / f"{frequency}_mu_{dMu:.0%}_hedge_path.pdf"), bbox_inches="tight")

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
    dMu = 0.05
    dMu_10 = 0.10
    dMu_20 = 0.20
    iStep_weekly = 13
    iStep_monthly = 3
    iStep_daily = 63
    iStep_four_times_daily = 252

    model_weekly = CallModel(dStd, iK, dAnR, rng)
    mS_weekly = model_weekly.path_simulation(iS, dMu, dT, iStep_weekly)
    vPnL_weekly = model_weekly.hedging(mS_weekly, dT)

    model_monthly = CallModel(dStd, iK, dAnR, rng)
    mS_monthly = model_monthly.path_simulation(iS, dMu, dT, iStep_monthly)
    vPnL_monthly = model_monthly.hedging(mS_monthly, dT)

    model_daily = CallModel(dStd, iK, dAnR, rng)
    mS_daily = model_daily.path_simulation(iS, dMu, dT, iStep_daily)
    vPnL_daily = model_daily.hedging(mS_daily, dT)

    model_four_times_daily = CallModel(dStd, iK, dAnR, rng)
    mS_four_times_daily = model_four_times_daily.path_simulation(iS, dMu, dT, iStep_four_times_daily)
    vPnL_four_times_daily = model_four_times_daily.hedging(mS_four_times_daily, dT)

    model_daily_10 = CallModel(dStd, iK, dAnR, rng)
    mS_daily_10 = model_daily_10.path_simulation(iS, dMu_10, dT, iStep_daily)
    vPnL_daily_10 = model_daily_10.hedging(mS_daily_10, dT)

    model_four_times_daily_10 = CallModel(dStd, iK, dAnR, rng)
    mS_four_times_daily_10 = model_four_times_daily_10.path_simulation(iS, dMu_10, dT, iStep_four_times_daily)
    vPnL_four_times_daily_10 = model_four_times_daily_10.hedging(mS_four_times_daily_10, dT)
    dPutPrice = model_four_times_daily.put_replication(mS_four_times_daily, dT)
    bPCP, dError = model_four_times_daily.put_call_parity(iS, dT, dPutPrice)


    model_daily_20 = CallModel(dStd, iK, dAnR, rng)
    mS_daily_20 = model_daily_20.path_simulation(iS, dMu_20, dT, iStep_daily)
    vPnL_daily_20 = model_daily_20.hedging(mS_daily_20, dT)

    model_four_times_daily_20 = CallModel(dStd, iK, dAnR, rng)
    mS_four_times_daily_20 = model_four_times_daily_20.path_simulation(iS, dMu_20, dT, iStep_four_times_daily)
    vPnL_four_times_daily_20 = model_four_times_daily_20.hedging(mS_four_times_daily_20, dT)
    
    
    print(f"Put price = {dPutPrice}, put-call-parity is: {bPCP} with Error = {dError:.6f}")

    plot_results(rng, vPnL_monthly, model_monthly.mHedge, model_monthly.mCall, interval= "month", dMu=dMu)
    plot_results(rng, vPnL_weekly, model_weekly.mHedge, model_weekly.mCall, interval= "week", dMu=dMu)
    plot_results(rng, vPnL_daily, model_daily.mHedge, model_daily.mCall, interval= "day", dMu=dMu)
    plot_results(rng, vPnL_four_times_daily, model_four_times_daily.mHedge, model_four_times_daily.mCall, interval= "four_times_daily", dMu=dMu)
    plot_results(rng, vPnL_daily_10, model_daily_10.mHedge, model_daily_10.mCall, interval= "day", dMu=dMu_10)
    plot_results(rng, vPnL_four_times_daily_10, model_four_times_daily_10.mHedge, model_four_times_daily_10.mCall, interval= "four_times_daily", dMu=dMu_10)
    plot_results(rng, vPnL_daily_20, model_daily_20.mHedge, model_daily_20.mCall, interval= "day", dMu=dMu_20)
    plot_results(rng, vPnL_four_times_daily_20, model_four_times_daily_20.mHedge, model_four_times_daily_20.mCall, interval= "four_times_daily", dMu=dMu_20)

if __name__ == "__main__":
    main()

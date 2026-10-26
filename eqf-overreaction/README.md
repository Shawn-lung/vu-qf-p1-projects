# EQF Assignment 1 - Testing the overreaction hypothesis

**Team project** (two students). **My part:** the whole pipeline behind the main results: data preparation and checks, corrections for splits, consolidations and price errors, the market data download and the regression models.

> The code and detailed results are not public. The course coordinator asked that assignment solutions stay off public repositories, so that future students cannot copy them; this page only describes the project.

## Question

Do ETFs show the overreaction effect that De Bondt and Thaler (1985) and Clare and Thomas (1995) found in stocks: do past losers outperform past winners?

## Approach

- Daily prices of ETFs listed on Borsa Italiana, 2021 to 2025, with returns measured against the market index.
- Data cleaning: extreme returns were investigated one by one and corrected for splits, consolidations and price errors.
- ETFs were sorted into groups on their past performance in a training period, and the groups were kept fixed in a later evaluation period.
- Tests: the loser-minus-winner return spread, and panel regressions with controls for market risk and liquidity, using robust standard errors.

## Main conclusion

No evidence of overreaction: past losers did not outperform past winners in any specification. A likely reason is that ETFs are diversified and many track bonds, commodities or leveraged futures, so the explanations behind overreaction in individual stocks, such as firm size or a change in perceived risk after losses, do not carry over.

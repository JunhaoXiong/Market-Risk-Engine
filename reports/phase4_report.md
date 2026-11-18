# Phase 4 Report: Stress testing and risk attribution

Portfolio: SPY 40%, QQQ 30%, TLT 20%, GLD 10%; $1,000,000. Positive P&L = gain.

## 1. Historical scenarios
Primary: buy-and-hold (positions fixed at scenario start). Benchmark: daily rebalanced.
Asset columns are each asset's cumulative return over the scenario.

| scenario           | period                   |   days | buy&hold   | P&L       | rebalanced   | max DD (b&h)   | worst day   | SPY    | QQQ    | TLT    | GLD    | largest loss from   |
|:-------------------|:-------------------------|-------:|:-----------|:----------|:-------------|:---------------|:------------|:-------|:-------|:-------|:-------|:--------------------|
| GFC peak-to-trough | 2007-10-09 to 2009-03-09 |    355 | -30.1%     | $-301,358 | -35.2%       | -32.1%         | -5.0%       | -55.2% | -51.6% | +25.1% | +23.9% | SPY                 |
| Post-Lehman        | 2008-09-12 to 2008-11-20 |     49 | -26.4%     | $-263,813 | -27.9%       | -26.7%         | -5.7%       | -39.8% | -41.1% | +10.8% | -2.8%  | SPY                 |
| COVID crash        | 2020-02-19 to 2020-03-23 |     23 | -19.4%     | $-193,767 | -20.3%       | -20.1%         | -6.3%       | -33.7% | -27.9% | +14.2% | -3.6%  | SPY                 |
| 2022 rate shock    | 2021-12-31 to 2022-10-12 |    196 | -26.8%     | $-268,330 | -26.5%       | -26.8%         | -3.8%       | -24.1% | -33.7% | -31.2% | -8.8%  | QQQ                 |
| 2018 Q4 selloff    | 2018-09-20 to 2018-12-24 |     65 | -12.9%     | $-129,158 | -13.4%       | -13.0%         | -2.6%       | -19.3% | -22.0% | +4.5%  | +5.0%  | SPY                 |
| 2013 taper tantrum | 2013-05-21 to 2013-06-24 |     23 | -6.1%      | $-61,198  | -6.1%        | -6.1%          | -2.6%       | -5.6%  | -5.8%  | -7.4%  | -6.7%  | SPY                 |

## 2. Hypothetical instantaneous shocks
P&L = sum_i w_i * shock_i * V (exact for simple returns).

| scenario     | portfolio   | P&L       | SPY shock   | QQQ shock   | TLT shock   | GLD shock   |
|:-------------|:------------|:----------|:------------|:------------|:------------|:------------|
| Equity crash | -12.7%      | $-127,000 | -20.0%      | -20.0%      | +5.0%       | +3.0%       |
| Stagflation  | -11.7%      | $-117,000 | -15.0%      | -15.0%      | -10.0%      | +8.0%       |
| Rate shock   | -8.6%       | $-86,000  | -8.0%       | -8.0%       | -15.0%      | +0.0%       |

## 3. Conditional stress: SPY -10%
Other assets move by their conditional expectation E[r | r_SPY] = Cov(r, r_SPY) / Var(r_SPY) * shock
(zero-mean joint normal), using covariance estimated in each period.

| covariance                      |   corr(SPY,TLT) | SPY    | QQQ    | TLT   | GLD   | portfolio   | P&L      |
|:--------------------------------|----------------:|:-------|:-------|:------|:------|:------------|:---------|
| Current EWMA (as of 2025-12-31) |            0.16 | -10.0% | -13.4% | -1.1% | -4.3% | -8.7%       | $-86,880 |
| Calm 2017                       |           -0.33 | -10.0% | -12.1% | +4.8% | +3.6% | -6.3%       | $-63,155 |
| GFC 2008-09..2009-03            |           -0.44 | -10.0% | -8.7%  | +1.8% | -0.3% | -6.3%       | $-62,768 |
| 2022 rate shock                 |            0.08 | -10.0% | -12.8% | -0.7% | -1.2% | -8.1%       | $-81,121 |

## 4. Risk attribution as of 2025-12-31 (window 2024-01-04 to 2025-12-31)
Gaussian VaR, zero mean, Euler decomposition (components sum to total VaR).

| covariance                         | alpha   | VaR     | SPY   | QQQ   | TLT   | GLD   | diversification   |
|:-----------------------------------|:--------|:--------|:------|:------|:------|:------|:------------------|
| EWMA (MLE lambda=0.9641) [primary] | 95%     | $10,446 | 42.7% | 44.8% | 4.3%  | 8.2%  | $3,060            |
| EWMA (MLE lambda=0.9641) [primary] | 99%     | $14,774 | 42.7% | 44.8% | 4.3%  | 8.2%  | $4,328            |
| Sample (500d)                      | 95%     | $13,926 | 47.1% | 44.6% | 5.1%  | 3.3%  | $3,919            |
| Sample (500d)                      | 99%     | $19,696 | 47.1% | 44.6% | 5.1%  | 3.3%  | $5,543            |
| Ledoit-Wolf const-corr (500d)      | 95%     | $13,793 | 46.8% | 44.3% | 5.5%  | 3.5%  | $4,034            |
| Ledoit-Wolf const-corr (500d)      | 99%     | $19,508 | 46.8% | 44.3% | 5.5%  | 3.5%  | $5,706            |

### Primary (EWMA) detail at 99%
|     | weight   | standalone VaR   | marginal VaR (per 100% wt)   | component VaR   | % of VaR   | incremental VaR   | component ES   |
|:----|:---------|:-----------------|:-----------------------------|:----------------|:-----------|:------------------|:---------------|
| SPY | 40%      | $6,551           | $15,775                      | $6,310          | 42.7%      | $6,129            | $7,229         |
| QQQ | 30%      | $6,964           | $22,041                      | $6,612          | 44.8%      | $6,325            | $7,576         |
| TLT | 20%      | $2,370           | $3,204                       | $641            | 4.3%       | $458              | $734           |
| GLD | 10%      | $3,217           | $12,106                      | $1,211          | 8.2%       | $887              | $1,387         |

Marginal VaR is dVaR/dw_i scaled by the notional, i.e. the VaR change per +100% weight;
incremental VaR is the VaR reduction from closing that position into cash.

### Historical component ES at 99% (window, no distributional assumption)
Tail days: 5; total ES $32,106.
Caveat: with a 500-day window the 99% tail holds only 5 days, so this split is
noisy; it is a cross-check on the parametric split, not a replacement.

|     | component ES   | % of ES   |
|:----|:---------------|:----------|
| SPY | $16,329        | 50.9%     |
| QQQ | $13,807        | 43.0%     |
| TLT | $1,416         | 4.4%      |
| GLD | $554           | 1.7%      |

## 5. Risk contribution over time (EWMA, fixed lambda = 0.94, yearly average share of portfolio variance)
A fixed RiskMetrics lambda is used for this descriptive path so that no parameter is
estimated with future data. Daily series: `reports/phase4_rolling_risk_contribution.csv`.

|   year | SPY   | QQQ   | TLT   | GLD   |
|-------:|:------|:------|:------|:------|
|   2006 | 40%   | 47%   | 4%    | 9%    |
|   2007 | 50%   | 44%   | -1%   | 8%    |
|   2008 | 58%   | 49%   | -6%   | -1%   |
|   2009 | 57%   | 41%   | -3%   | 4%    |
|   2010 | 55%   | 44%   | -7%   | 7%    |
|   2011 | 58%   | 48%   | -9%   | 3%    |
|   2012 | 54%   | 49%   | -11%  | 9%    |
|   2013 | 48%   | 40%   | 2%    | 10%   |
|   2014 | 51%   | 49%   | -2%   | 2%    |
|   2015 | 52%   | 46%   | -0%   | 2%    |
|   2016 | 51%   | 48%   | 1%    | -0%   |
|   2017 | 40%   | 49%   | 7%    | 4%    |
|   2018 | 47%   | 50%   | 1%    | 2%    |
|   2019 | 51%   | 50%   | -2%   | 1%    |
|   2020 | 51%   | 48%   | -3%   | 3%    |
|   2021 | 43%   | 46%   | 7%    | 5%    |
|   2022 | 46%   | 47%   | 5%    | 2%    |
|   2023 | 41%   | 42%   | 14%   | 3%    |
|   2024 | 41%   | 43%   | 10%   | 6%    |
|   2025 | 47%   | 44%   | 5%    | 3%    |

# Phase 5 Report: risk parity with volatility targeting

Out-of-sample: 2009-12-23 to 2025-12-31 (4030 days), same dates as Phase 3.
Risk model: EWMA covariance forecast for each day from the trailing 500 returns, lambda
re-estimated by multivariate MLE every 21 days (median lambda 0.9715).
Target for day t = risk-parity weights (cap 60%) x min(1, vol target / ex-ante vol); the
remainder is cash earning the 13-week T-bill rate (^IRX, quote at t-1). Long-only, no leverage.
Costs: 5 bp per unit of one-way turnover. Sharpe uses returns in excess of the T-bill rate.
Min-variance / min-CVaR / benchmark are fully invested and rebalanced every 21 days.

|                                            | CAGR   | vol   |   Sharpe (excess) | max DD   |   Calmar | VaR99   | ES99   | max 63d vol   | 2020   | 2022   | avg cash   | turnover/yr   |   trades/yr |
|:-------------------------------------------|:-------|:------|------------------:|:---------|---------:|:--------|:-------|:--------------|:-------|:-------|:-----------|:--------------|------------:|
| RP + VT 10%, threshold 5.0% [primary]      | 9.9%   | 8.2%  |              1.02 | -19.9%   |     0.5  | 1.42%   | 1.85%  | 15.5%         | 15.7%  | -15.8% | 4.0%       | 136.5%        |       10.07 |
| RP + VT 10%, monthly                       | 10.1%  | 8.5%  |              1.02 | -20.7%   |     0.49 | 1.44%   | 1.94%  | 19.5%         | 13.7%  | -16.9% | 3.1%       | 127.7%        |       12.01 |
| RP + VT 10%, threshold 2.5%                | 9.9%   | 8.2%  |              1.03 | -20.4%   |     0.49 | 1.40%   | 1.83%  | 15.2%         | 14.8%  | -16.3% | 3.7%       | 220.4%        |       29.01 |
| RP + VT 10%, threshold 10.0%               | 9.9%   | 8.3%  |              1.02 | -22.0%   |     0.45 | 1.45%   | 1.92%  | 16.0%         | 14.7%  | -17.9% | 4.5%       | 78.2%         |        3.06 |
| RP + VT 8%, threshold 5.0%                 | 9.3%   | 7.5%  |              1.03 | -17.5%   |     0.53 | 1.29%   | 1.69%  | 13.4%         | 15.1%  | -13.9% | 9.4%       | 147.8%        |       11.13 |
| RP + VT 12%, threshold 5.0%                | 9.9%   | 8.6%  |              0.99 | -21.9%   |     0.45 | 1.48%   | 1.97%  | 17.1%         | 15.2%  | -17.4% | 2.0%       | 139.7%        |       10.19 |
| RP (no VT, fully invested), threshold 5.0% | 10.6%  | 9.0%  |              1.01 | -23.5%   |     0.45 | 1.55%   | 2.02%  | 22.1%         | 22.0%  | -17.5% | 0.0%       | 126.2%        |        9.25 |
| Min variance (EWMA cov), monthly           | 10.2%  | 8.5%  |              1.03 | -21.5%   |     0.47 | 1.44%   | 1.91%  | 22.3%         | 11.7%  | -14.1% | 0.0%       | 239.9%        |       12.01 |
| Min CVaR 95% (500d scenarios), monthly     | 10.3%  | 9.0%  |              0.98 | -25.7%   |     0.4  | 1.45%   | 2.02%  | 21.5%         | 20.1%  | -19.2% | 0.0%       | 83.9%         |       12.01 |
| Benchmark 40/30/20/10, monthly             | 13.1%  | 12.5% |              0.94 | -27.7%   |     0.47 | 2.20%   | 3.06%  | 37.8%         | 28.6%  | -23.5% | 0.0%       | 32.5%         |       12.01 |

`max 63d vol` = highest realised volatility over any 63-day window (how well the target held).

## Primary strategy: average holdings by year
|   year | SPY   | QQQ   | TLT   | GLD   | cash   |
|-------:|:------|:------|:------|:------|:-------|
|   2009 | 22%   | 22%   | 43%   | 13%   | 0%     |
|   2010 | 21%   | 19%   | 41%   | 19%   | 0%     |
|   2011 | 22%   | 19%   | 35%   | 21%   | 3%     |
|   2012 | 25%   | 19%   | 39%   | 16%   | 0%     |
|   2013 | 27%   | 24%   | 34%   | 16%   | 0%     |
|   2014 | 26%   | 20%   | 34%   | 20%   | 0%     |
|   2015 | 24%   | 20%   | 31%   | 24%   | 0%     |
|   2016 | 25%   | 20%   | 31%   | 24%   | 0%     |
|   2017 | 33%   | 19%   | 24%   | 24%   | 0%     |
|   2018 | 20%   | 14%   | 37%   | 29%   | 0%     |
|   2019 | 22%   | 16%   | 36%   | 26%   | 0%     |
|   2020 | 19%   | 13%   | 33%   | 21%   | 14%    |
|   2021 | 26%   | 17%   | 30%   | 24%   | 3%     |
|   2022 | 16%   | 12%   | 22%   | 29%   | 21%    |
|   2023 | 23%   | 17%   | 20%   | 28%   | 11%    |
|   2024 | 24%   | 19%   | 27%   | 26%   | 3%     |
|   2025 | 19%   | 15%   | 35%   | 23%   | 8%     |

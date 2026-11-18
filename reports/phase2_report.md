# Phase 2 Report: EWMA, GARCH and covariance shrinkage

Estimation window: last 1000 daily returns, 2022-01-06 to 2025-12-31.
Forecasts below are one-day-ahead for the next trading day after 2025-12-31;
they are **not** evaluated here (no data after the window is used). Portfolio $1,000,000.

## EWMA
- Univariate (portfolio returns) lambda by Gaussian MLE: **0.9190** (at bound: False); RiskMetrics benchmark 0.94
- Next-day vol (ann.): MLE 9.43%, lambda=0.94 9.72%
- Covariance (4 assets) common lambda by multivariate Gaussian MLE: **0.9676** (at bound: False)

## GARCH(1,1)
|                     | GARCH-t (primary)   | GARCH-Normal (benchmark)   |
|:--------------------|:--------------------|:---------------------------|
| mu (daily)          | 8.49e-04            | 7.96e-04                   |
| omega               | 1.775e-06           | 1.933e-06                  |
| alpha               | 0.0869              | 0.1021                     |
| beta                | 0.8930              | 0.8783                     |
| alpha+beta          | 0.9799              | 0.9803                     |
| half-life (days)    | 34.1                | 34.9                       |
| long-run vol (ann.) | 14.90%              | 15.74%                     |
| nu                  | 8.53                | -                          |
| log-lik             | 3351.9              | 3337.3                     |
| AIC                 | -6693.8             | -6666.7                    |
| BIC                 | -6669.2             | -6647.0                    |
| next-day vol (ann.) | 10.62%              | 10.58%                     |
| converged           | True                | True                       |

AIC difference (Normal - t): 27.1 (positive favours Student-t).
Note: a likelihood-ratio test of t vs Normal is non-standard (nu = infinity lies on the
boundary), so AIC/BIC are reported instead of a chi-square p-value.

Diagnostics (Ljung-Box, 10 lags, on squared series):
- raw squared returns: Q = 100.7, p = 3.88e-17
- squared standardised GARCH-t residuals: Q = 17.0, p = 0.0738
- excess kurtosis of standardised residuals: 1.43

## Ledoit-Wolf shrinkage (asset covariance)
| target               |   delta | portfolio vol (ann.)   |
|:---------------------|--------:|:-----------------------|
| identity             |  0.0226 | 15.04%                 |
| constant correlation |  0.0205 | 15.10%                 |

Sample correlation in the window:
|     |   SPY |   QQQ |   TLT |   GLD |
|:----|------:|------:|------:|------:|
| SPY |  1    |  0.96 |  0.09 |  0.12 |
| QQQ |  0.96 |  1    |  0.08 |  0.1  |
| TLT |  0.09 |  0.08 |  1    |  0.25 |
| GLD |  0.12 |  0.1  |  0.25 |  1    |

## One-day-ahead VaR / ES forecasts (positive = loss)
| model                                   | alpha   | VaR $   | ES $    |
|:----------------------------------------|:--------|:--------|:--------|
| Historical (window)                     | 95%     | $15,011 | $21,494 |
| Normal, sample var (window)             | 95%     | $15,278 | $19,264 |
| Normal, sample cov                      | 95%     | $15,271 | $19,254 |
| Normal, LW identity cov                 | 95%     | $15,175 | $19,135 |
| Normal, LW const-corr cov               | 95%     | $15,238 | $19,214 |
| EWMA (MLE lambda=0.9190), zero mean     | 95%     | $9,768  | $12,249 |
| EWMA (RiskMetrics 0.94), zero mean      | 95%     | $10,075 | $12,635 |
| EWMA cov (MLE lambda=0.9676), zero mean | 95%     | $10,517 | $13,189 |
| GARCH(1,1)-Normal [benchmark]           | 95%     | $10,168 | $12,953 |
| GARCH(1,1)-t [primary]                  | 95%     | $9,948  | $13,668 |
| Historical (window)                     | 99%     | $25,545 | $32,916 |
| Normal, sample var (window)             | 99%     | $21,779 | $25,011 |
| Normal, sample cov                      | 99%     | $21,768 | $24,998 |
| Normal, LW identity cov                 | 99%     | $21,633 | $24,844 |
| Normal, LW const-corr cov               | 99%     | $21,722 | $24,946 |
| EWMA (MLE lambda=0.9190), zero mean     | 99%     | $13,815 | $15,827 |
| EWMA (RiskMetrics 0.94), zero mean      | 99%     | $14,250 | $16,326 |
| EWMA cov (MLE lambda=0.9676), zero mean | 99%     | $14,875 | $17,042 |
| GARCH(1,1)-Normal [benchmark]           | 99%     | $14,710 | $16,969 |
| GARCH(1,1)-t [primary]                  | 99%     | $15,857 | $19,740 |

## Caveats
- Single estimation window; parameter stability and forecast accuracy are assessed in Phase 3 (rolling backtest).
- EWMA forecasts use a zero mean (RiskMetrics convention); GARCH and window-Normal include the sample mean.
- Covariance-based rows use the same window mean as the Normal row so only the covariance differs.

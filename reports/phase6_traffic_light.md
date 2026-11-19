# Phase 6: Basel traffic light (99% VaR, rolling 250 days)

Source: Phase 3 primary set-up (W=500, refit 21d) out-of-sample forecasts.
Zones per 250-day window: green 0-4 exceptions, yellow 5-9, red 10+.
Shares are over all 3781 overlapping windows; last window ends 2025-12-31.

| model             |   windows | green   | yellow   | red   |   max exceptions |   last 250d exceptions | last zone   |
|:------------------|----------:|:--------|:---------|:------|-----------------:|-----------------------:|:------------|
| Historical        |      3781 | 59.9%   | 39.1%    | 1.0%  |               10 |                      5 | yellow      |
| Normal            |      3781 | 48.2%   | 28.3%    | 23.5% |               23 |                      6 | yellow      |
| EWMA (MLE lambda) |      3781 | 24.1%   | 69.6%    | 6.3%  |               11 |                      5 | yellow      |
| GARCH-Normal      |      3781 | 29.8%   | 63.8%    | 6.4%  |               11 |                      6 | yellow      |
| GARCH-t           |      3781 | 48.7%   | 50.8%    | 0.4%  |               10 |                      5 | yellow      |

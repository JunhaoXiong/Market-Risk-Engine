# Phase 6: Basel traffic light (99% VaR, rolling 250 days)

Source: out-of-sample forecasts 2009-12-23 to 2025-12-31, refit every 21 days (official models from config var_model, other Phase 3 models at W=500).
Zones per 250-day window: green 0-4 exceptions, yellow 5-9, red 10+.
Shares are over all 3781 overlapping windows; last window ends 2025-12-31.

| model                                |   windows | green   | yellow   | red   |   max exceptions |   last 250d exceptions | last zone   |
|:-------------------------------------|----------:|:--------|:---------|:------|-----------------:|-----------------------:|:------------|
| FHS (GJR-t filter), W=1000 [primary] |      3781 | 80.6%   | 19.4%    | 0.0%  |                8 |                      4 | green       |
| GJR-skew-t, W=500 [alternative]      |      3781 | 72.0%   | 28.0%    | 0.0%  |                8 |                      4 | green       |
| GARCH-t, W=500 [previous]            |      3781 | 48.7%   | 50.8%    | 0.4%  |               10 |                      5 | yellow      |
| Historical, W=500                    |      3781 | 59.9%   | 39.1%    | 1.0%  |               10 |                      5 | yellow      |
| Normal, W=500                        |      3781 | 48.2%   | 28.3%    | 23.5% |               23 |                      6 | yellow      |
| EWMA (MLE lambda), W=500             |      3781 | 24.1%   | 69.6%    | 6.3%  |               11 |                      5 | yellow      |
| GARCH-Normal, W=500                  |      3781 | 29.8%   | 63.8%    | 6.4%  |               11 |                      6 | yellow      |

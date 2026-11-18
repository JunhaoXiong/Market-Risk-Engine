# Market Risk Analytics & Model Validation Engine

A reproducible Python engine for measuring and validating market risk on a
multi-asset ETF portfolio (SPY 40%, QQQ 30%, TLT 20%, GLD 10%, $1,000,000).

> Work in progress. Phases 1-5 are complete.

## Roadmap
| Phase | Scope | Status |
|---|---|---|
| 1 | Data pipeline, data quality checks, portfolio returns, Historical / Normal VaR & ES | done |
| 2 | EWMA (MLE lambda), GARCH(1,1)-t vs Normal, Ledoit-Wolf shrinkage | done |
| 3 | Rolling out-of-sample backtest (2009-12 to 2025-12), Kupiec, Christoffersen | done |
| 4 | Historical / hypothetical / conditional stress tests; Euler VaR & ES attribution | done |
| 5 | Risk parity + volatility targeting with cash and threshold rebalancing; min-variance / min-CVaR benchmarks | done |
| 6 | Streamlit dashboard, model validation report | planned |

## Quick start
```bash
pip install -e ".[dev]"
pytest                          # unit tests (offline, synthetic data)
python scripts/run_phase1.py    # uses data/raw/prices.csv snapshot
python scripts/run_phase1.py --refresh   # re-download from Yahoo Finance
python scripts/run_phase2.py    # volatility models on the last 1,000-day window
python scripts/run_phase3.py    # rolling backtest (~1 min on 4 cores)
python scripts/run_phase4.py    # stress tests and risk attribution
python scripts/run_phase5.py    # walk-forward allocation backtest
```

## Key assumptions (all in `config.yaml`)
- **Data**: Yahoo Finance adjusted close (total return), 2006-01-01 to 2025-12-31.
  The download is snapshotted with a SHA-256 hash because Yahoo revises adjusted
  prices retroactively.
- **Returns**: simple returns for the portfolio (they aggregate linearly across assets).
- **Weights**: constant (daily rebalanced to target).
- **Sign convention**: VaR and ES are reported as positive losses.
  `VaR_a = -q_{1-a}(P&L)`, `ES_a = -E[P&L | P&L <= -VaR_a]`.
- **Horizon**: 1 day.
- **Volatility models**: rolling 1,000-day estimation window; GARCH(1,1) with Student-t
  innovations is primary, Normal GARCH is the benchmark; EWMA lambda is chosen by Gaussian
  MLE inside the window only; Ledoit-Wolf shrinkage with identity and constant-correlation targets.
- **Backtesting**: primary set-up is a 500-day rolling window, GARCH/EWMA-lambda refit every
  21 trading days with daily variance updates; sensitivity runs use daily refits and 250 / 1,000-day
  windows. All set-ups forecast the same 4,030 days. Tests at 5% (primary) and 1% significance.
- **Stress testing**: historical scenarios replayed buy-and-hold (primary) and daily rebalanced
  (benchmark); attribution uses the EWMA covariance (MLE lambda) as of the last date.
- **Allocation**: daily EWMA risk forecast (lambda by MLE every 21 days on a 500-day window);
  risk-parity weights scaled to a 10% annual vol target, remainder in T-bills (^IRX); trade only when
  any weight drifts more than 5pp from target; information at the close of t-1 drives day t.
- **Data quality**: issues are reported, never silently fixed; no forward-filling;
  extreme returns are flagged but kept.

## Layout
```
config.yaml            all modelling parameters
src/mre/data/          download + cache (loader.py), quality checks (quality.py)
src/mre/portfolio.py   asset and portfolio returns, P&L
src/mre/risk/var.py    Historical / Normal VaR and ES
src/mre/models/        ewma.py, garch.py, shrinkage.py
src/mre/stress/        scenarios.py (historical, hypothetical, conditional)
src/mre/attribution/   euler.py (component / marginal / incremental VaR, component ES)
src/mre/optimization/  optimizers.py, dynamic.py (vol target, cash, threshold), backtest.py
src/mre/backtest/      rolling.py (out-of-sample forecasts), coverage_tests.py (Kupiec, Christoffersen)
scripts/run_phase1.py  end-to-end Phase 1 pipeline
reports/               generated reports
tests/                 pytest suite
```

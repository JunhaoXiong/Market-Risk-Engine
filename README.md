# Market Risk Analytics & Model Validation Engine

A reproducible Python engine for measuring and validating market risk on a
multi-asset ETF portfolio (SPY 40%, QQQ 30%, TLT 20%, GLD 10%, $1,000,000).

> Work in progress. Phase 1 (data pipeline + static VaR/ES) is complete.

## Roadmap
| Phase | Scope | Status |
|---|---|---|
| 1 | Data pipeline, data quality checks, portfolio returns, Historical / Normal VaR & ES | done |
| 2 | EWMA, GARCH, covariance shrinkage | planned |
| 3 | Rolling backtesting, Kupiec, Christoffersen tests | planned |
| 4 | Stress testing, risk attribution | planned |
| 5 | Portfolio optimisation | planned |
| 6 | Streamlit dashboard, model validation report | planned |

## Quick start
```bash
pip install -e ".[dev]"
pytest                          # unit tests (offline, synthetic data)
python scripts/run_phase1.py    # uses data/raw/prices.csv snapshot
python scripts/run_phase1.py --refresh   # re-download from Yahoo Finance
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
- **Data quality**: issues are reported, never silently fixed; no forward-filling;
  extreme returns are flagged but kept.

## Layout
```
config.yaml            all modelling parameters
src/mre/data/          download + cache (loader.py), quality checks (quality.py)
src/mre/portfolio.py   asset and portfolio returns, P&L
src/mre/risk/var.py    Historical / Normal VaR and ES
scripts/run_phase1.py  end-to-end Phase 1 pipeline
reports/               generated reports
tests/                 pytest suite
```

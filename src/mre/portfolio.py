"""Asset and portfolio returns.

Conventions
-----------
* Simple returns R_t = P_t / P_{t-1} - 1 are used for the portfolio because they
  aggregate linearly across assets: R_p = sum_i w_i R_i holds exactly. Log returns
  do not aggregate across assets (they aggregate across *time* instead).
* Constant weights: today's weights are applied to every historical day, which
  corresponds to daily rebalancing back to target. This is the standard
  "current portfolio, historical scenarios" set-up for VaR.
* The return on date t uses only prices up to and including t (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns; the first row (no prior price) is dropped."""
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def validate_weights(weights: dict[str, float] | pd.Series, columns) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    missing = set(w.index) - set(columns)
    if missing:
        raise KeyError(f"Weights given for tickers not in returns: {sorted(missing)}")
    if not np.isclose(w.sum(), 1.0, atol=1e-9):
        raise ValueError(f"Weights must sum to 1, got {w.sum():.6f}")
    return w


def portfolio_returns(asset_returns: pd.DataFrame, weights: dict[str, float] | pd.Series) -> pd.Series:
    """Constant-weight portfolio simple returns.

    Rows with any missing asset return raise an error instead of being silently
    re-weighted; align prices first.
    """
    w = validate_weights(weights, asset_returns.columns)
    r = asset_returns[w.index]
    if r.isna().any().any():
        bad = r.index[r.isna().any(axis=1)]
        raise ValueError(f"{len(bad)} rows contain missing returns (first: {bad[0].date()}); align prices first")
    out = r.to_numpy() @ w.to_numpy()
    return pd.Series(out, index=r.index, name="portfolio")


def pnl(portfolio_rets: pd.Series, portfolio_value: float) -> pd.Series:
    """Daily P&L in currency for a fixed notional (positive = gain)."""
    return (portfolio_rets * portfolio_value).rename("pnl")

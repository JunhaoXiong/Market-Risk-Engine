"""Daily-updated allocation with volatility targeting, cash and threshold rebalancing.

Timing (no look-ahead)
----------------------
* ``cov_forecasts[i]`` is the covariance forecast for day t = oos_start + i, built
  from returns r[t - window : t] only (EWMA with lambda re-estimated by multivariate
  MLE every ``refit_every`` days on the same trailing window).
* The target computed from that forecast is traded at the close of day t-1 and earns
  day t's return. Equivalently: information up to the close of t-1 drives day t.

Portfolio accounting
--------------------
Weights w (risky assets) plus cash c = 1 - sum(w) >= 0. Day t gross return is
    R_t = w' r_t + c * rf_t
After the return, positions drift:  w <- w (1 + r_t) / (1 + R_t),  c <- c (1 + rf_t) / (1 + R_t).
On a rebalance the cost  cost * sum_i |w_target_i - w_i|  is charged (cash trades are free).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mre.models.ewma import ewma_covariance, fit_ewma_cov_lambda
from mre.optimization.optimizers import risk_parity


def ewma_cov_forecasts(
    R: pd.DataFrame, window: int, oos_start: int, refit_every: int = 21, burn_in: int = 30,
) -> tuple[np.ndarray, pd.Series]:
    """(T_oos x N x N covariance forecasts, lambda path) using trailing windows only."""
    X = R.to_numpy(dtype=float)
    n, N = X.shape
    if oos_start < window or oos_start >= n:
        raise ValueError("need window <= oos_start < n")
    out = np.empty((n - oos_start, N, N))
    lams = np.empty(n - oos_start)
    lam = None
    for i, t in enumerate(range(oos_start, n)):
        win = X[t - window:t]
        if lam is None or i % refit_every == 0:
            fit, nxt = fit_ewma_cov_lambda(win, burn_in)
            lam = fit.lam
        else:
            _, nxt = ewma_covariance(win, lam, burn_in)
        out[i] = nxt
        lams[i] = lam
    return out, pd.Series(lams, index=R.index[oos_start:], name="lambda")


def vol_target_scale(w: np.ndarray, cov: np.ndarray, target_annual: float | None,
                     periods: int = 252) -> float:
    """k = min(1, target / ex-ante vol); no leverage. ``None`` means fully invested."""
    if target_annual is None:
        return 1.0
    vol = float(np.sqrt(w @ cov @ w * periods))
    return min(1.0, target_annual / vol) if vol > 0 else 1.0


def risk_parity_vol_target(cov: np.ndarray, target_annual: float | None, ub: float) -> np.ndarray:
    b = risk_parity(cov, ub=ub).weights
    return vol_target_scale(b, cov, target_annual) * b


@dataclass
class DynamicResult:
    returns: pd.Series          # net daily returns
    weights: pd.DataFrame       # risky weights held at the start of each day (after trading)
    cash: pd.Series
    turnover: pd.Series         # one-way turnover per day (0 when no trade)
    targets: pd.DataFrame

    @property
    def n_rebalances(self) -> int:
        return int((self.turnover > 0).sum())


def simulate(
    asset_returns: pd.DataFrame, targets: np.ndarray, cash_rate_daily: pd.Series | np.ndarray,
    oos_start: int, rule: str = "threshold", threshold: float = 0.05, every: int = 21,
    cost_bps: float = 5.0,
) -> DynamicResult:
    """Run the book given precomputed day-by-day targets (row i applies to day oos_start + i)."""
    X = asset_returns.to_numpy(dtype=float)[oos_start:]
    T, N = X.shape
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (T, N):
        raise ValueError(f"targets must be {(T, N)}, got {targets.shape}")
    if (targets < -1e-10).any() or (targets.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("targets must be long-only with sum <= 1")
    rf = np.asarray(cash_rate_daily, dtype=float)
    rf = rf[oos_start:] if rf.shape[0] == asset_returns.shape[0] else rf
    if rf.shape != (T,):
        raise ValueError("cash rate must align with asset returns")
    if rule not in ("threshold", "calendar"):
        raise ValueError("rule must be 'threshold' or 'calendar'")
    cost = cost_bps / 1e4
    w = np.zeros(N)
    c = 1.0
    net = np.empty(T)
    held = np.empty((T, N))
    cash = np.empty(T)
    turn = np.zeros(T)
    for i in range(T):
        tgt = targets[i]
        if i == 0:
            trade = True
        elif rule == "threshold":
            trade = np.max(np.abs(tgt - w)) > threshold
        else:
            trade = i % every == 0
        tc = 0.0
        if trade:
            turn[i] = np.abs(tgt - w).sum()
            tc = cost * turn[i]
            w = tgt.copy()
            c = 1.0 - w.sum()
        held[i] = w
        cash[i] = c
        g = w @ X[i] + c * rf[i]
        net[i] = g - tc
        w = w * (1 + X[i]) / (1 + g)
        c = c * (1 + rf[i]) / (1 + g)
    idx = asset_returns.index[oos_start:]
    cols = asset_returns.columns
    return DynamicResult(
        returns=pd.Series(net, index=idx, name="net"),
        weights=pd.DataFrame(held, index=idx, columns=cols),
        cash=pd.Series(cash, index=idx, name="cash"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        targets=pd.DataFrame(targets, index=idx, columns=cols),
    )


def targets_from(cov_path: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.vstack([fn(S) for S in cov_path])

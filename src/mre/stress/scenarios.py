"""Stress testing: historical replay, hypothetical shocks, conditional (covariance-implied) shocks.

Conventions
-----------
* Losses are reported as negative P&L (consistent with portfolio.pnl); the summary
  also gives the loss as a positive number for readability.
* Historical scenarios use the close of the first trading day on/after ``start`` as
  the base and the close of the last trading day on/before ``end`` as the end point.
* ``buy_and_hold``: positions fixed at the start (weights drift); scenario return
  is sum_i w_i R_i where R_i is the asset's cumulative return, so asset
  contributions add up exactly.
* ``rebalanced``: weights reset to target daily; scenario return is the compounded
  daily portfolio return. Contributions are not additive and are not reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScenarioResult:
    name: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    method: str
    portfolio_return: float
    pnl: float
    asset_returns: dict = field(default_factory=dict)
    contributions: dict = field(default_factory=dict)  # P&L by asset (buy-and-hold / shocks)
    worst_day: float | None = None
    max_drawdown: float | None = None
    n_days: int | None = None

    def as_row(self) -> dict:
        return {
            "scenario": self.name, "start": None if self.start is None else self.start.date(),
            "end": None if self.end is None else self.end.date(), "method": self.method,
            "days": self.n_days, "portfolio_return": self.portfolio_return, "pnl": self.pnl,
            "worst_day": self.worst_day, "max_drawdown": self.max_drawdown,
        }


def _weights(weights, columns) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    if set(w.index) - set(columns):
        raise KeyError(f"unknown tickers {sorted(set(w.index) - set(columns))}")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return w


def _max_drawdown(path: pd.Series) -> float:
    """Most negative peak-to-trough return of a value path (includes the start value)."""
    running_max = path.cummax()
    return float((path / running_max - 1).min())


def historical_scenario(
    prices: pd.DataFrame, weights, start: str, end: str, portfolio_value: float,
    name: str = "", method: str = "buy_and_hold",
) -> ScenarioResult:
    w = _weights(weights, prices.columns)
    px = prices.loc[start:end, w.index]
    if len(px) < 2:
        raise ValueError(f"scenario {name or (start, end)} has fewer than 2 trading days")
    if px.isna().any().any():
        raise ValueError("prices contain NaN inside the scenario window; align first")
    asset_cum = px.iloc[-1] / px.iloc[0] - 1
    daily = px.pct_change().iloc[1:]

    if method == "buy_and_hold":
        value_path = (px / px.iloc[0]) @ w            # value of $1 at start
        port_ret = float(value_path.iloc[-1] - 1)
        contrib = (w * asset_cum * portfolio_value).to_dict()
        daily_port = value_path.pct_change().iloc[1:]
    elif method == "rebalanced":
        daily_port = daily @ w
        value_path = pd.concat([pd.Series([1.0], index=[px.index[0]]), (1 + daily_port).cumprod()])
        port_ret = float(value_path.iloc[-1] - 1)
        contrib = {}
    else:
        raise ValueError("method must be 'buy_and_hold' or 'rebalanced'")

    return ScenarioResult(
        name=name, start=px.index[0], end=px.index[-1], method=method,
        portfolio_return=port_ret, pnl=port_ret * portfolio_value,
        asset_returns=asset_cum.to_dict(), contributions=contrib,
        worst_day=float(daily_port.min()), max_drawdown=_max_drawdown(value_path),
        n_days=len(px) - 1,
    )


def hypothetical_shock(weights, shocks: dict, portfolio_value: float, name: str = "") -> ScenarioResult:
    """Instantaneous shocks (simple returns) applied to current positions; unshocked assets = 0."""
    w = pd.Series(weights, dtype=float)
    unknown = set(shocks) - set(w.index)
    if unknown:
        raise KeyError(f"shocks for unknown tickers {sorted(unknown)}")
    s = pd.Series(shocks, dtype=float).reindex(w.index).fillna(0.0)
    if (s <= -1).any():
        raise ValueError("a simple-return shock cannot be <= -100%")
    contrib = w * s * portfolio_value
    ret = float((w * s).sum())
    return ScenarioResult(name=name, start=None, end=None, method="instantaneous",
                          portfolio_return=ret, pnl=ret * portfolio_value,
                          asset_returns=s.to_dict(), contributions=contrib.to_dict())


def conditional_expected_returns(cov: pd.DataFrame, shocks: dict) -> pd.Series:
    """E[r_u | r_s = shocks] = Sigma_us Sigma_ss^{-1} s under a zero-mean joint normal.

    Shocked assets keep their specified values; the rest are filled in.
    """
    shocked = list(shocks)
    if not shocked or set(shocked) - set(cov.index):
        raise KeyError("shocks must name at least one asset in the covariance matrix")
    others = [c for c in cov.index if c not in shocked]
    s = np.array([shocks[k] for k in shocked], dtype=float)
    S_ss = cov.loc[shocked, shocked].to_numpy()
    S_us = cov.loc[others, shocked].to_numpy()
    implied = S_us @ np.linalg.solve(S_ss, s)
    out = pd.Series(0.0, index=cov.index)
    out[shocked] = s
    out[others] = implied
    return out


def conditional_shock(cov: pd.DataFrame, weights, shocks: dict, portfolio_value: float,
                      name: str = "") -> ScenarioResult:
    full = conditional_expected_returns(cov, shocks)
    res = hypothetical_shock(weights, full.to_dict(), portfolio_value, name)
    res.method = "conditional"
    return res

"""Walk-forward backtest of allocation strategies.

At each rebalance date t (every ``rebalance_every`` days from ``oos_start``) the
strategy receives only asset returns r[t - window : t] and returns target weights.
Between rebalances positions drift with realised returns (buy-and-hold), which is
what a monthly-rebalanced fund actually holds. At a rebalance the cost
``cost_bps`` x one-way turnover sum_i |w_target_i - w_drifted_i| is deducted from
that day's return. Day t's return is earned on the weights set at the close of t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

Strategy = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class WalkForwardResult:
    returns: pd.Series          # net daily portfolio returns
    gross_returns: pd.Series
    weights: pd.DataFrame       # target weights at each rebalance date
    turnover: pd.Series         # one-way turnover at each rebalance date


def walk_forward(
    asset_returns: pd.DataFrame, strategy: Strategy, window: int, oos_start: int,
    rebalance_every: int = 21, cost_bps: float = 5.0,
) -> WalkForwardResult:
    R = asset_returns
    n, N = R.shape
    if oos_start < window or oos_start >= n:
        raise ValueError("need window <= oos_start < number of observations")
    if R.isna().any().any():
        raise ValueError("asset returns contain NaN")
    X = R.to_numpy()
    cost = cost_bps / 1e4
    net = np.empty(n - oos_start)
    gross = np.empty(n - oos_start)
    w_held = None  # weights at the start of the day (after last close)
    w_rows, t_rows, dates = [], [], []
    for i, t in enumerate(range(oos_start, n)):
        tc = 0.0
        if (t - oos_start) % rebalance_every == 0:
            target = np.asarray(strategy(R.iloc[t - window:t]), dtype=float)
            if target.shape != (N,) or not np.isclose(target.sum(), 1) or (target < -1e-10).any():
                raise ValueError(f"strategy returned invalid weights on {R.index[t].date()}: {target}")
            # first rebalance starts from cash, so the initial purchase counts as turnover
            to = np.abs(target - (w_held if w_held is not None else 0.0)).sum()
            if w_held is None:
                to = 1.0
            tc = cost * to
            w_held = target
            w_rows.append(target)
            t_rows.append(to)
            dates.append(R.index[t])
        g = float(w_held @ X[t])
        gross[i] = g
        net[i] = g - tc
        # drift: value of each position after today's return, renormalised
        grown = w_held * (1 + X[t])
        w_held = grown / grown.sum()
    idx = R.index[oos_start:]
    return WalkForwardResult(
        returns=pd.Series(net, index=idx, name="net"),
        gross_returns=pd.Series(gross, index=idx, name="gross"),
        weights=pd.DataFrame(w_rows, index=pd.DatetimeIndex(dates), columns=R.columns),
        turnover=pd.Series(t_rows, index=pd.DatetimeIndex(dates), name="turnover"),
    )


def performance(r: pd.Series, periods: int = 252) -> dict:
    """Summary statistics of a daily return series (rf = 0)."""
    from mre.risk.var import historical_es, historical_var

    wealth = (1 + r).cumprod()
    years = len(r) / periods
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(periods))
    dd = float((wealth / wealth.cummax() - 1).min())
    return {
        "CAGR": cagr, "vol": vol, "Sharpe (rf=0)": float(r.mean() * periods / vol),
        "max drawdown": dd, "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "VaR99 (daily)": historical_var(r.to_numpy(), 0.99),
        "ES99 (daily)": historical_es(r.to_numpy(), 0.99),
        "worst day": float(r.min()),
    }

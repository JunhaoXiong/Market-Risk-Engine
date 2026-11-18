"""Value-at-Risk and Expected Shortfall.

Sign convention
---------------
Inputs are returns or P&L where *positive = gain*. Outputs are reported as
*positive losses*:

    VaR_a = -q_{1-a}(X)                     (q = quantile of X)
    ES_a  = -E[X | X <= -VaR_a]

so a 99% VaR of 25,000 means "on 1% of days we expect to lose at least 25,000".
All functions are horizon-agnostic: pass 1-day returns to get 1-day risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def _as_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Empty input")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or inf")
    return arr


def _check_level(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Confidence level must be in (0, 1), got {alpha}")


def historical_var(x, alpha: float = 0.99, method: str = "linear") -> float:
    """Historical-simulation VaR: negative of the empirical (1 - alpha) quantile."""
    _check_level(alpha)
    arr = _as_array(x)
    return float(-np.quantile(arr, 1.0 - alpha, method=method))


def historical_es(x, alpha: float = 0.99, method: str = "linear") -> float:
    """Historical ES: mean loss over observations at or beyond the VaR threshold."""
    arr = _as_array(x)
    var = historical_var(arr, alpha, method)
    tail = arr[arr <= -var]
    if tail.size == 0:  # can only happen with exotic interpolation choices
        tail = np.array([arr.min()])
    return float(-tail.mean())


def _moments(arr: np.ndarray, include_mean: bool) -> tuple[float, float]:
    if arr.size < 2:
        raise ValueError("Need at least two observations to estimate volatility")
    mu = float(arr.mean()) if include_mean else 0.0
    sigma = float(arr.std(ddof=1))
    return mu, sigma


def normal_var(x, alpha: float = 0.99, include_mean: bool = True) -> float:
    """Parametric (Gaussian) VaR = -(mu + sigma * z_{1-alpha})."""
    _check_level(alpha)
    mu, sigma = _moments(_as_array(x), include_mean)
    return float(-(mu + sigma * norm.ppf(1.0 - alpha)))


def normal_es(x, alpha: float = 0.99, include_mean: bool = True) -> float:
    """Parametric (Gaussian) ES = -mu + sigma * phi(z_alpha) / (1 - alpha)."""
    _check_level(alpha)
    mu, sigma = _moments(_as_array(x), include_mean)
    return float(-mu + sigma * norm.pdf(norm.ppf(alpha)) / (1.0 - alpha))


@dataclass(frozen=True)
class RiskRow:
    method: str
    alpha: float
    var: float
    es: float


def risk_table(
    pnl: pd.Series,
    levels=(0.95, 0.99),
    quantile_method: str = "linear",
    normal_include_mean: bool = True,
) -> pd.DataFrame:
    """VaR/ES for every (method, level) pair, in the units of ``pnl``."""
    rows = []
    for a in levels:
        rows.append(RiskRow("Historical", a, historical_var(pnl, a, quantile_method),
                            historical_es(pnl, a, quantile_method)))
        rows.append(RiskRow("Normal", a, normal_var(pnl, a, normal_include_mean),
                            normal_es(pnl, a, normal_include_mean)))
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["es_to_var"] = df["es"] / df["var"]
    return df

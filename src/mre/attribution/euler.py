"""Risk attribution.

Parametric (Gaussian) Euler decomposition
-----------------------------------------
With portfolio volatility sigma_p = sqrt(w' S w) and VaR = -w'mu + z_a sigma_p
(z_a = Phi^{-1}(a)), VaR is homogeneous of degree one in w, so Euler's theorem gives

    VaR = sum_i w_i * dVaR/dw_i,     dVaR/dw_i = -mu_i + z_a (S w)_i / sigma_p

* marginal VaR_i   = dVaR/dw_i           (VaR change per unit of weight)
* component VaR_i  = w_i * marginal_i     (sums exactly to total VaR)
* percent contrib  = component_i / VaR
The same split applies to Gaussian ES with z_a replaced by phi(z_a)/(1-a).
Incremental VaR_i = VaR(w) - VaR(w with w_i = 0), i.e. closing the position into cash.

Historical component ES
-----------------------
CES_i = -E[w_i r_i | r_p <= -VaR_p]; these sum to the historical ES of the
portfolio because r_p = sum_i w_i r_i on every tail day. No distributional assumption.
All amounts are returned in currency units when ``portfolio_value`` is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from mre.risk.var import historical_var


def _wvec(weights, index) -> np.ndarray:
    w = pd.Series(weights, dtype=float).reindex(index)
    if w.isna().any():
        raise KeyError("weights missing for some assets in the covariance matrix")
    return w.to_numpy()


def _risk(cov: np.ndarray, w: np.ndarray, mu: np.ndarray, k: float) -> float:
    s = float(np.sqrt(w @ cov @ w))
    return float(-(w @ mu) + k * s)


def parametric_decomposition(
    cov: pd.DataFrame, weights, alpha: float = 0.99, portfolio_value: float = 1.0,
    mean: pd.Series | None = None, measure: str = "var",
) -> pd.DataFrame:
    if measure == "var":
        k = norm.ppf(alpha)
    elif measure == "es":
        k = norm.pdf(norm.ppf(alpha)) / (1 - alpha)
    else:
        raise ValueError("measure must be 'var' or 'es'")
    idx = cov.index
    S = cov.loc[idx, idx].to_numpy()
    w = _wvec(weights, idx)
    mu = np.zeros(len(idx)) if mean is None else mean.reindex(idx).to_numpy(dtype=float)
    sigma_p = float(np.sqrt(w @ S @ w))
    total = _risk(S, w, mu, k)
    marginal = -mu + k * (S @ w) / sigma_p
    component = w * marginal
    incremental = []
    for i in range(len(idx)):
        w0 = w.copy()
        w0[i] = 0.0
        incremental.append(total - (_risk(S, w0, mu, k) if w0.any() else 0.0))
    df = pd.DataFrame({
        "weight": w,
        "marginal": marginal * portfolio_value,
        "component": component * portfolio_value,
        "pct_contribution": component / total,
        "incremental": np.array(incremental) * portfolio_value,
        "standalone": np.array([-(w[i] * mu[i]) + k * abs(w[i]) * np.sqrt(S[i, i])
                                for i in range(len(idx))]) * portfolio_value,
    }, index=idx)
    df.attrs["total"] = total * portfolio_value
    df.attrs["measure"] = measure
    df.attrs["alpha"] = alpha
    df.attrs["diversification_benefit"] = (df["standalone"].sum() - df.attrs["total"])
    return df


def historical_component_es(
    asset_returns: pd.DataFrame, weights, alpha: float = 0.99, portfolio_value: float = 1.0,
    quantile_method: str = "linear",
) -> pd.DataFrame:
    w = pd.Series(weights, dtype=float)
    R = asset_returns[w.index]
    if R.isna().any().any():
        raise ValueError("asset returns contain NaN")
    contrib = R * w  # w_i r_i per day
    port = contrib.sum(axis=1)
    var = historical_var(port.to_numpy(), alpha, quantile_method)
    tail = port <= -var
    ces = -contrib[tail].mean()
    total = float(ces.sum())
    df = pd.DataFrame({
        "weight": w,
        "component_es": ces * portfolio_value,
        "pct_contribution": ces / total,
    })
    df.attrs["total_es"] = total * portfolio_value
    df.attrs["var"] = var * portfolio_value
    df.attrs["tail_days"] = int(tail.sum())
    return df


def rolling_pct_contribution(cov_path: np.ndarray, weights: np.ndarray, index) -> pd.DataFrame:
    """Percent volatility (= Gaussian VaR, zero mean) contribution per day from a T x N x N path."""
    w = np.asarray(weights, dtype=float)
    Sw = np.einsum("tij,j->ti", cov_path, w)
    var_p = Sw @ w
    pct = (w * Sw) / var_p[:, None]
    return pd.DataFrame(pct, index=index)

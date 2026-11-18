"""Exponentially weighted moving average (RiskMetrics) variance and covariance.

Timing convention (no look-ahead)
---------------------------------
``sigma2[t]`` is the variance *forecast for day t* made at the close of day t-1:

    sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t-1]**2

so it uses returns strictly before t. The final element of the returned
``next_forecast`` is the forecast for the first day *after* the sample.

Initialisation: the first ``burn_in`` returns seed the recursion with their
sample variance and are excluded from the likelihood, so the initial value has
no influence on lambda through the very first few terms.

Lambda by maximum likelihood
----------------------------
lambda is chosen to maximise the Gaussian (quasi-)likelihood
    sum_t  -0.5 * [log(2 pi sigma2_t) + r_t^2 / sigma2_t]
over the estimation window only. Gaussian QMLE is consistent for the variance
dynamics even when returns are fat-tailed; tails are handled by the GARCH-t model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

LAMBDA_BOUNDS = (0.80, 0.9995)


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("returns must be non-empty and finite")
    return a


def ewma_variance(r, lam: float, burn_in: int = 30) -> tuple[np.ndarray, float]:
    """Return (sigma2 forecasts aligned with r, next-day forecast).

    ``sigma2[t]`` for t < burn_in is NaN (seeding period).
    """
    r = _arr(r).ravel()
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    if r.size <= burn_in:
        raise ValueError("need more observations than burn_in")
    sigma2 = np.full(r.size, np.nan)
    s0 = float(np.mean(r[:burn_in] ** 2))
    # y[n] = lam * y[n-1] + (1 - lam) * r[burn_in + n]^2, with y[-1] = s0  (vectorised recursion)
    y, _ = lfilter([1.0 - lam], [1.0, -lam], r[burn_in:] ** 2, zi=[lam * s0])
    sigma2[burn_in] = s0
    sigma2[burn_in + 1:] = y[:-1]
    return sigma2, float(y[-1])


def ewma_neg_loglik(lam: float, r, burn_in: int = 30) -> float:
    r = _arr(r).ravel()
    sigma2, _ = ewma_variance(r, lam, burn_in)
    s2 = sigma2[burn_in:]
    x = r[burn_in:]
    return float(0.5 * np.sum(np.log(2 * np.pi * s2) + x**2 / s2))


@dataclass(frozen=True)
class EWMAFit:
    lam: float
    loglik: float
    next_variance: float
    n_obs: int
    at_bound: bool


def fit_ewma_lambda(r, burn_in: int = 30, bounds=LAMBDA_BOUNDS) -> EWMAFit:
    """Choose lambda by Gaussian MLE on the supplied (training) returns only."""
    r = _arr(r).ravel()
    res = minimize_scalar(
        ewma_neg_loglik, bounds=bounds, args=(r, burn_in), method="bounded",
        options={"xatol": 1e-6},
    )
    lam = float(res.x)
    _, nxt = ewma_variance(r, lam, burn_in)
    at_bound = min(abs(lam - bounds[0]), abs(lam - bounds[1])) < 1e-4
    return EWMAFit(lam, -float(res.fun), nxt, r.size - burn_in, at_bound)


# ---------------------------------------------------------------- multivariate

def ewma_covariance(R, lam: float, burn_in: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """EWMA covariance recursion for a T x N return matrix.

    Returns (Sigma forecasts, shape T x N x N with NaN during burn-in;
    next-day forecast N x N). Zero-mean convention, as in RiskMetrics.
    """
    R = _arr(R)
    if R.ndim != 2:
        raise ValueError("R must be T x N")
    T, N = R.shape
    if T <= burn_in:
        raise ValueError("need more observations than burn_in")
    out = np.full((T, N, N), np.nan)
    S0 = R[:burn_in].T @ R[:burn_in] / burn_in
    # the recursion is linear and element-wise in the outer products, so filter all
    # N x N series at once: Y[n] = lam * Y[n-1] + (1 - lam) * x_n x_n',  Y[-1] = S0
    outer = np.einsum("ti,tj->tij", R[burn_in:], R[burn_in:])
    Y, _ = lfilter([1.0 - lam], [1.0, -lam], outer, axis=0, zi=(lam * S0)[None, :, :])
    out[burn_in] = S0
    out[burn_in + 1:] = Y[:-1]
    return out, Y[-1]


def ewma_cov_neg_loglik(lam: float, R, burn_in: int = 30) -> float:
    R = _arr(R)
    sig, _ = ewma_covariance(R, lam, burn_in)
    N = R.shape[1]
    S = sig[burn_in:]
    X = R[burn_in:]
    sign, logdet = np.linalg.slogdet(S)
    if (sign <= 0).any():
        return np.inf
    quad = np.einsum("ti,ti->t", X, np.linalg.solve(S, X[:, :, None])[:, :, 0])
    return float(0.5 * np.sum(N * np.log(2 * np.pi) + logdet + quad))


def fit_ewma_cov_lambda(R, burn_in: int = 30, bounds=LAMBDA_BOUNDS) -> tuple[EWMAFit, np.ndarray]:
    """Common lambda for the covariance matrix by multivariate Gaussian MLE.

    Returns (fit, next-day covariance forecast). ``fit.next_variance`` is NaN here.
    """
    R = _arr(R)
    res = minimize_scalar(
        ewma_cov_neg_loglik, bounds=bounds, args=(R, burn_in), method="bounded",
        options={"xatol": 1e-5},
    )
    lam = float(res.x)
    _, nxt = ewma_covariance(R, lam, burn_in)
    at_bound = min(abs(lam - bounds[0]), abs(lam - bounds[1])) < 1e-4
    return EWMAFit(lam, -float(res.fun), float(np.nan), R.shape[0] - burn_in, at_bound), nxt

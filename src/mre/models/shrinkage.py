"""Ledoit-Wolf covariance shrinkage.

Sigma_shrunk = delta * F + (1 - delta) * S, where S is the sample covariance
(divided by T, as in the papers) and F a structured target. delta is the
analytically optimal intensity estimated from the same data.

Two targets:
* ``identity``: F = mean(diag S) * I         (Ledoit & Wolf, 2004, JMVA)
* ``constant_correlation``: F_ij = rbar * s_i * s_j, F_ii = S_ii
                                               (Ledoit & Wolf, 2004, "Honey, I shrunk...")
The constant-correlation target keeps each asset's own variance, which suits
assets with very different volatilities (e.g. QQQ vs TLT).

With few assets and many observations (here N = 4, T = 1000) the optimal
delta is expected to be small; that is a property of the data, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShrinkageResult:
    covariance: np.ndarray
    sample: np.ndarray
    target: np.ndarray
    delta: float
    target_name: str


def _demean(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite T x N matrix with T >= 2")
    return X - X.mean(axis=0)


def ledoit_wolf_identity(X) -> ShrinkageResult:
    Xc = _demean(X)
    T, N = Xc.shape
    S = Xc.T @ Xc / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    d2 = np.sum((S - F) ** 2)
    # b_bar^2 = (1/T^2) sum_t || x_t x_t' - S ||_F^2
    b2_bar = sum(np.sum((np.outer(x, x) - S) ** 2) for x in Xc) / T**2
    b2 = min(b2_bar, d2)
    delta = 0.0 if d2 == 0 else float(b2 / d2)
    return ShrinkageResult(delta * F + (1 - delta) * S, S, F, delta, "identity")


def ledoit_wolf_constant_correlation(X) -> ShrinkageResult:
    Xc = _demean(X)
    T, N = Xc.shape
    S = Xc.T @ Xc / T
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))

    Y = Xc**2
    # pi_ij = (1/T) sum_t (x_ti x_tj - s_ij)^2
    pi_mat = (Y.T @ Y) / T - S**2
    pi_hat = pi_mat.sum()
    # theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - s_ij)
    theta = ((Xc**3).T @ Xc) / T - np.diag(S)[:, None] * S
    rho_diag = np.trace(pi_mat)
    ratio = sd[None, :] / sd[:, None]  # ratio[i, j] = s_j / s_i
    off = rbar * (ratio * theta + ratio.T * theta.T)
    np.fill_diagonal(off, 0.0)
    rho_hat = rho_diag + off.sum() / 2.0
    gamma_hat = np.sum((F - S) ** 2)
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = float(max(0.0, min(1.0, kappa / T)))
    return ShrinkageResult(delta * F + (1 - delta) * S, S, F, delta, "constant_correlation")


def portfolio_vol(cov: np.ndarray, weights) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ cov @ w))

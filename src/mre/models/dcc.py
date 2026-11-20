"""DCC-GARCH(1,1) (Engle 2002) for the portfolio covariance.

Step 1 (marginals): each asset i has its own univariate GARCH-family model
    r_{i,t} = mu_i + sigma_{i,t} z_{i,t}
fitted by MLE (``mre.models.garch.fit_garch``; GJR-t by default).

Step 2 (correlation): with the standardised residuals z_t = (z_{1,t}, ..., z_{N,t})
    Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1},    Q_0 = Qbar
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
Qbar is the sample second-moment matrix of z on the training window ("correlation
targeting"), so only (a, b) are estimated, by maximising the Gaussian correlation
quasi-likelihood  -1/2 sum_t ( ln|R_t| + z_t' R_t^{-1} z_t )  subject to a, b >= 0,
a + b <= 0.999. Two-step QML is consistent for (a, b) even when z is not Normal
(the Normal is only used as a quasi-likelihood; the VaR quantile is taken from the
filtered portfolio residuals instead, see ``mre.backtest.rolling_dcc``).

Portfolio (constant weights w): H_t = D_t R_t D_t, D_t = diag(sigma_{i,t}),
    sigma_{p,t}^2 = w' H_t w,     mu_p = w' mu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

AB_MAX = 0.999


@dataclass(frozen=True)
class DCCFit:
    a: float
    b: float
    qbar: np.ndarray
    loglik: float          # correlation-stage quasi log-likelihood
    converged: bool
    q_next: np.ndarray = field(repr=False)   # Q for the day after the window
    r_path: np.ndarray = field(repr=False)   # R_t for every day in the window, shape (T, N, N)

    def next_q(self, q: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Q_{t+1} from Q_t and z_t with fixed parameters."""
        return (1 - self.a - self.b) * self.qbar + self.a * np.outer(z, z) + self.b * q


def q_to_r(q: np.ndarray) -> np.ndarray:
    d = 1 / np.sqrt(np.diag(q))
    return q * np.outer(d, d)


def _filter(a: float, b: float, z: np.ndarray, qbar: np.ndarray):
    """Run the Q recursion; returns (R path, Q after the last observation, quasi loglik)."""
    T, N = z.shape
    qs = np.empty((T + 1, N, N))
    qs[0] = qbar
    zz = z[:, :, None] * z[:, None, :]
    c = (1 - a - b) * qbar
    for t in range(T):
        qs[t + 1] = c + a * zz[t] + b * qs[t]
    d = 1 / np.sqrt(np.einsum("tii->ti", qs[:T]))
    rs = qs[:T] * d[:, :, None] * d[:, None, :]
    sign, logdet = np.linalg.slogdet(rs)
    if np.any(sign <= 0):
        return rs, qs[T], -np.inf
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(rs, z[:, :, None])[..., 0])
    return rs, qs[T], float(-0.5 * (logdet + quad).sum())


def fit_dcc(z: np.ndarray) -> DCCFit:
    """Estimate DCC(1,1) (a, b) on standardised residuals z (T x N), correlation targeting."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] < 2 or z.shape[0] < 100 or not np.all(np.isfinite(z)):
        raise ValueError("need a finite T x N array with N >= 2 and T >= 100")
    qbar = z.T @ z / z.shape[0]

    def nll(theta):
        a, b = theta
        if a < 0 or b < 0 or a + b > AB_MAX:
            return 1e10
        return -_filter(a, b, z, qbar)[2]

    best = None
    for start in ((0.02, 0.95), (0.05, 0.90), (0.01, 0.97)):
        res = minimize(nll, np.array(start), method="SLSQP",
                       bounds=[(0.0, AB_MAX), (0.0, AB_MAX)],
                       constraints=[{"type": "ineq", "fun": lambda th: AB_MAX - th[0] - th[1]}],
                       options={"maxiter": 200, "ftol": 1e-9})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        a, b, ok = 0.0, 0.0, False
    else:
        a, b = (float(np.clip(v, 0.0, AB_MAX)) for v in best.x)
        if a + b > AB_MAX:
            b = AB_MAX - a
        ok = bool(best.success)
    rs, q, ll = _filter(a, b, z, qbar)
    return DCCFit(a=a, b=b, qbar=qbar, loglik=float(ll), converged=ok and np.isfinite(ll),
                  q_next=q, r_path=rs)


def fit_ccc(z: np.ndarray) -> DCCFit:
    """Constant conditional correlation (Bollerslev 1990): a = b = 0, R_t = corr(Qbar).

    Used as an ablation: same marginals as DCC, no correlation dynamics.
    """
    z = np.asarray(z, dtype=float)
    qbar = z.T @ z / z.shape[0]
    rs, q, ll = _filter(0.0, 0.0, z, qbar)
    return DCCFit(a=0.0, b=0.0, qbar=qbar, loglik=float(ll), converged=bool(np.isfinite(ll)),
                  q_next=q, r_path=rs)


def portfolio_sigma(w: np.ndarray, sigmas: np.ndarray, r: np.ndarray) -> np.ndarray:
    """sqrt(w' D R D w); ``sigmas`` (..., N) and ``r`` (..., N, N) broadcast over leading dims."""
    ws = np.asarray(w) * sigmas
    return np.sqrt(np.einsum("...i,...ij,...j->...", ws, r, ws))

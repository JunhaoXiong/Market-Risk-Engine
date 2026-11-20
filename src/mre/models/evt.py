"""Extreme value theory: peaks-over-threshold GPD tail for VaR / ES.

Used on the standardised residuals of a volatility filter (McNeil & Frey 2000,
"FHS-EVT" / conditional EVT). Losses are x = -z (left tail of returns = right tail
of losses). For a threshold u (an empirical quantile of x) the excesses y = x - u | x > u
are modelled as Generalised Pareto:

    P(Y <= y) = 1 - (1 + xi y / beta)^(-1/xi),   beta > 0   (xi = 0: exponential)

With n observations and N_u exceedances, for a tail probability p = 1 - alpha < N_u / n:

    VaR_alpha = u + beta / xi * ( (n p / N_u)^(-xi) - 1 )
    ES_alpha  = (VaR_alpha + beta - xi u) / (1 - xi)          (needs xi < 1)

(xi -> 0 limits: VaR = u - beta ln(n p / N_u), ES = VaR + beta.)
xi > 0 is a heavy (power-law) tail, xi = 0 thin (Normal-like), xi < 0 bounded.

Estimation is maximum likelihood with xi restricted to [-0.5, 0.9]: xi > -0.5 is the
range where the MLE is regular (Smith 1985), and xi < 1 keeps ES finite. The bounds
are reported when they bind rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

XI_BOUNDS = (-0.5, 0.9)
_XI0 = 1e-8


@dataclass(frozen=True)
class GPDFit:
    xi: float
    beta: float
    u: float          # threshold, in loss units of the input
    n: int            # sample size the threshold quantile was taken from
    n_exceed: int     # N_u
    loglik: float
    at_bound: bool    # xi ended on one of XI_BOUNDS

    def var(self, alpha: float) -> float:
        p = 1 - alpha
        if not 0 < p <= self.n_exceed / self.n + 1e-9:  # p = N_u/n gives VaR = u
            raise ValueError(f"tail probability {p} must not exceed the threshold exceedance rate "
                             f"{self.n_exceed / self.n:.3f}")
        k = self.n * p / self.n_exceed
        if abs(self.xi) < _XI0:
            return float(self.u - self.beta * np.log(k))
        return float(self.u + self.beta / self.xi * (k ** (-self.xi) - 1))

    def es(self, alpha: float) -> float:
        v = self.var(alpha)
        return float((v + self.beta - self.xi * self.u) / (1 - self.xi))


def _gpd_nll(theta, y):
    xi, lb = theta
    beta = np.exp(lb)
    if abs(xi) < _XI0:
        return y.size * lb + y.sum() / beta
    t = 1 + xi * y / beta
    if np.any(t <= 0):
        return 1e10
    return y.size * lb + (1 + 1 / xi) * np.log(t).sum()


def fit_gpd(excess: np.ndarray) -> tuple[float, float, float, bool]:
    """MLE of (xi, beta) for positive excesses. Returns (xi, beta, loglik, at_bound)."""
    y = np.asarray(excess, dtype=float)
    if y.size < 10 or np.any(y < 0) or not np.all(np.isfinite(y)):
        raise ValueError("need at least 10 finite non-negative excesses")
    m, v = y.mean(), y.var(ddof=1)
    # method-of-moments start (valid for xi < 1/2), clipped into the bounds
    xi0 = float(np.clip(0.5 * (1 - m * m / v), XI_BOUNDS[0] + 0.05, XI_BOUNDS[1] - 0.05))
    b0 = max(m * (1 - xi0), 1e-12)
    best = None
    for start in ((xi0, np.log(b0)), (0.1, np.log(m)), (-0.1, np.log(m))):
        res = minimize(_gpd_nll, np.array(start), args=(y,), method="L-BFGS-B",
                       bounds=[XI_BOUNDS, (np.log(m) - 10, np.log(m) + 10)])
        if best is None or res.fun < best.fun:
            best = res
    xi, beta = float(best.x[0]), float(np.exp(best.x[1]))
    at_bound = bool(min(abs(xi - XI_BOUNDS[0]), abs(xi - XI_BOUNDS[1])) < 1e-6)
    return xi, beta, float(-best.fun), at_bound


def fit_pot(losses: np.ndarray, threshold_q: float = 0.90) -> GPDFit:
    """Peaks over threshold: u = empirical ``threshold_q`` quantile of ``losses``."""
    x = np.asarray(losses, dtype=float)
    if not 0.5 <= threshold_q < 1:
        raise ValueError("threshold_q must be in [0.5, 1)")
    u = float(np.quantile(x, threshold_q))
    y = x[x > u] - u
    xi, beta, ll, ab = fit_gpd(y)
    return GPDFit(xi=xi, beta=beta, u=u, n=x.size, n_exceed=int(y.size), loglik=ll, at_bound=ab)


def evt_var_es(z: np.ndarray, alpha: float, threshold_q: float = 0.90,
               fit: GPDFit | None = None) -> tuple[float, float]:
    """VaR / ES (positive = loss) of standardised residuals ``z`` with a GPD left tail."""
    f = fit_pot(-np.asarray(z, dtype=float), threshold_q) if fit is None else fit
    return f.var(alpha), f.es(alpha)

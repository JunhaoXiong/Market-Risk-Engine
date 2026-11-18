"""GARCH(1,1) volatility models with Student-t (primary) or Normal (benchmark) innovations.

Model:  r_t = mu + eps_t,  eps_t = sigma_t z_t,
        sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
z_t is i.i.d. standardised (unit-variance) Normal or Student-t(nu).

Estimation uses the ``arch`` package (MLE). Returns are scaled by 100 before
fitting for numerical stability and parameters/forecasts are scaled back.
Only the returns passed in are used, so calling ``fit_garch`` on a training
window and forecasting the next day has no look-ahead.

VaR/ES for a standardised Student-t with nu > 2 degrees of freedom: if
T ~ t_nu then z = T * sqrt((nu - 2) / nu) has unit variance, so
    q_z(p)   = t_nu^{-1}(p) * c,        c = sqrt((nu - 2) / nu)
    ES_z(a)  = c * f_nu(t_a) / (1 - a) * (nu + t_a^2) / (nu - 1),   t_a = t_nu^{-1}(a)
(ES expressed as a positive number for the left tail by symmetry).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as student_t

SCALE = 100.0


@dataclass(frozen=True)
class GarchFit:
    dist: str
    mu: float          # daily mean, in return units
    omega: float       # in return^2 units
    alpha: float
    beta: float
    nu: float | None   # Student-t degrees of freedom (None for Normal)
    loglik: float      # log-likelihood of the *unscaled* returns
    aic: float
    bic: float
    n_obs: int
    next_sigma: float  # one-step-ahead volatility forecast
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def half_life_days(self) -> float:
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("inf")

    @property
    def long_run_vol(self) -> float:
        p = self.persistence
        return float(np.sqrt(self.omega / (1 - p))) if p < 1 else float("inf")


def fit_garch(r, dist: str = "t") -> GarchFit:
    from arch import arch_model

    if dist not in ("t", "normal"):
        raise ValueError("dist must be 't' or 'normal'")
    r = np.asarray(r, dtype=float).ravel()
    if r.size < 250 or not np.all(np.isfinite(r)):
        raise ValueError("need at least 250 finite observations to fit GARCH")
    am = arch_model(r * SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
    res = am.fit(disp="off", options={"maxiter": 1000})
    p = res.params
    fc = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    # log-likelihood of unscaled returns: each density picks up a Jacobian log(SCALE)
    ll = float(res.loglikelihood + r.size * np.log(SCALE))
    k = len(p)
    return GarchFit(
        dist=dist,
        mu=float(p["mu"]) / SCALE,
        omega=float(p["omega"]) / SCALE**2,
        alpha=float(p["alpha[1]"]),
        beta=float(p["beta[1]"]),
        nu=float(p["nu"]) if dist == "t" else None,
        loglik=ll,
        aic=2 * k - 2 * ll,
        bic=k * np.log(r.size) - 2 * ll,
        n_obs=r.size,
        next_sigma=float(np.sqrt(fc)) / SCALE,
        converged=res.convergence_flag == 0,
    )


def standardized_quantile(p: float, dist: str, nu: float | None = None) -> float:
    if dist == "normal":
        return float(norm.ppf(p))
    if nu is None or nu <= 2:
        raise ValueError("Student-t needs nu > 2 for unit variance")
    return float(student_t.ppf(p, nu) * np.sqrt((nu - 2) / nu))


def standardized_es(alpha: float, dist: str, nu: float | None = None) -> float:
    """Expected shortfall of the standardised innovation, as a positive number."""
    if dist == "normal":
        return float(norm.pdf(norm.ppf(alpha)) / (1 - alpha))
    if nu is None or nu <= 2:
        raise ValueError("Student-t needs nu > 2 for unit variance")
    ta = student_t.ppf(alpha, nu)
    c = np.sqrt((nu - 2) / nu)
    return float(c * student_t.pdf(ta, nu) / (1 - alpha) * (nu + ta**2) / (nu - 1))


def garch_var_es(fit: GarchFit, alpha: float, include_mean: bool = True) -> tuple[float, float]:
    """One-day-ahead VaR and ES in return units (positive = loss)."""
    mu = fit.mu if include_mean else 0.0
    q = standardized_quantile(1 - alpha, fit.dist, fit.nu)
    var = -(mu + fit.next_sigma * q)
    es = -mu + fit.next_sigma * standardized_es(alpha, fit.dist, fit.nu)
    return float(var), float(es)

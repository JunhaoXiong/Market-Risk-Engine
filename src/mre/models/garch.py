"""GARCH-family volatility models for one-day VaR / ES.

Mean:   r_t = mu + eps_t,  eps_t = sigma_t z_t,  z_t i.i.d. with mean 0, variance 1.
Variance recursions (``vol``):
  garch   sigma2_t = omega + alpha eps_{t-1}^2 + beta sigma2_{t-1}
  gjr     sigma2_t = omega + (alpha + gamma 1[eps_{t-1} < 0]) eps_{t-1}^2 + beta sigma2_{t-1}
          (Glosten-Jagannathan-Runkle 1993; gamma > 0 = bad news raises vol more)
  egarch  ln sigma2_t = omega + alpha (|z_{t-1}| - sqrt(2/pi)) + gamma z_{t-1} + beta ln sigma2_{t-1}
          (Nelson 1991; gamma < 0 = leverage effect; sqrt(2/pi) = E|z| under Normal,
          which is the centring the ``arch`` package uses for every innovation law)
Innovations (``dist``):
  normal  standard Normal
  t       Student-t(nu) rescaled to unit variance
  skewt   Hansen (1994) skewed t(eta, lambda), unit variance; lambda < 0 = fatter left tail

Estimation is MLE with the ``arch`` package. Returns are scaled by 100 before
fitting for numerical stability and parameters/forecasts are scaled back.
Only the returns passed in are used, so fitting on a training window and
forecasting the next day has no look-ahead.

VaR/ES for a standardised Student-t with nu > 2 degrees of freedom: if
T ~ t_nu then z = T * sqrt((nu - 2) / nu) has unit variance, so
    q_z(p)   = t_nu^{-1}(p) * c,        c = sqrt((nu - 2) / nu)
    ES_z(a)  = c * f_nu(t_a) / (1 - a) * (nu + t_a^2) / (nu - 1),   t_a = t_nu^{-1}(a)
(ES expressed as a positive number for the left tail by symmetry).
For the skewed t the quantile is Hansen's closed form (``arch`` SkewStudent.ppf)
and ES_z(a) = -(1/(1-a)) * integral_0^{1-a} q_z(u) du by numerical quadrature
(the left tail is not the mirror of the right tail, so no symmetry shortcut).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.stats import norm, t as student_t

SCALE = 100.0
VOLS = ("garch", "gjr", "egarch")
DISTS = ("normal", "t", "skewt")
_E_ABS_Z = float(np.sqrt(2 / np.pi))


@dataclass(frozen=True)
class GarchFit:
    dist: str
    mu: float          # daily mean, in return units
    omega: float       # return^2 units for garch/gjr; log-variance units (of unscaled returns) for egarch
    alpha: float
    beta: float
    nu: float | None   # Student-t / skew-t degrees of freedom (None for Normal)
    loglik: float      # log-likelihood of the *unscaled* returns
    aic: float
    bic: float
    n_obs: int
    next_sigma: float  # one-step-ahead volatility forecast
    converged: bool
    vol: str = "garch"
    gamma: float = 0.0         # asymmetry (gjr / egarch); 0 for symmetric GARCH
    lam: float | None = None   # skew-t skewness lambda in (-1, 1)
    std_resid: np.ndarray | None = field(default=None, repr=False, compare=False)
    # in-sample standardised residuals (r_t - mu) / sigma_t, oldest first (used by FHS)
    last_sigma2: float = float("nan")  # in-sample variance of the last training day

    @property
    def persistence(self) -> float:
        if self.vol == "egarch":
            return self.beta
        # E[1(z<0) z^2] = 1/2 only for symmetric z; good enough as a summary statistic
        return self.alpha + self.beta + 0.5 * self.gamma

    @property
    def half_life_days(self) -> float:
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("inf")

    @property
    def long_run_vol(self) -> float:
        if self.vol == "egarch":
            return float("nan")  # no simple closed form
        p = self.persistence
        return float(np.sqrt(self.omega / (1 - p))) if p < 1 else float("inf")

    def next_variance(self, eps: float, sigma2: float) -> float:
        """sigma2_{t+1} from the shock eps_t = r_t - mu and sigma2_t, with fixed parameters."""
        if self.vol == "garch":
            return self.omega + self.alpha * eps**2 + self.beta * sigma2
        if self.vol == "gjr":
            return self.omega + (self.alpha + self.gamma * (eps < 0)) * eps**2 + self.beta * sigma2
        z = eps / np.sqrt(sigma2)
        ln_s2 = (self.omega + self.alpha * (abs(z) - _E_ABS_Z) + self.gamma * z
                 + self.beta * np.log(sigma2))
        return float(np.exp(min(ln_s2, 0.0)))  # numerical guard: daily vol capped at 100%


def _bounded_egarch():
    """EGARCH(1,1,1) with alpha >= 0 and beta <= 0.999.

    The unrestricted ``arch`` EGARCH optimum on 500-day windows sometimes has alpha < 0
    and beta ~ 1, so a single large |z| (or a tiny last sigma) makes the log-variance
    recursion explode out of sample. alpha >= 0 (a larger shock never lowers volatility)
    and a stationarity margin on beta are the usual economic restrictions.
    """
    from arch.univariate import EGARCH

    class BoundedEGARCH(EGARCH):
        def bounds(self, resids):
            b = super().bounds(resids)
            return [b[0], (0.0, 2.0), (-2.0, 2.0), (0.0, 0.999)]

    return BoundedEGARCH(p=1, o=1, q=1)


def _make_model(rs: np.ndarray, vol: str, dist: str):
    from arch.univariate import (
        GARCH, ConstantMean, Normal, SkewStudent, StudentsT,
    )

    o = 0 if vol == "garch" else 1
    proc = _bounded_egarch() if vol == "egarch" else GARCH(p=1, o=o, q=1)
    law = {"normal": Normal, "t": StudentsT, "skewt": SkewStudent}[dist]()
    return ConstantMean(rs, volatility=proc, distribution=law, rescale=False)


def _plausible(res, vol: str, dist: str, r_scaled: np.ndarray) -> bool:
    """Reject numerically degenerate optima (common for EGARCH on short windows)."""
    p = res.params
    if res.convergence_flag != 0 or not np.isfinite(res.loglikelihood):
        return False
    fc = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    cv = np.asarray(res.conditional_volatility)
    if not np.isfinite(fc) or fc <= 0 or np.sqrt(fc) > 10 * r_scaled.std() or not np.isfinite(cv).all():
        return False
    if dist in ("t", "skewt") and float(p["nu" if dist == "t" else "eta"]) <= 2.1:
        return False
    return True


def _egarch_starts(r_scaled: np.ndarray, dist: str) -> list[np.ndarray]:
    lv = np.log(r_scaled.var())
    tail = {"normal": [], "t": [8.0], "skewt": [8.0, -0.1]}[dist]
    return [np.array([r_scaled.mean(), (1 - b) * lv, a, g, b, *tail])
            for b, a, g in ((0.95, 0.10, -0.10), (0.98, 0.10, -0.10), (0.90, 0.15, -0.05))]



def fit_garch(r, dist: str = "t", vol: str = "garch") -> GarchFit:
    """MLE fit. ``converged`` is False when no numerically plausible optimum was found."""
    if dist not in DISTS:
        raise ValueError(f"dist must be one of {DISTS}")
    if vol not in VOLS:
        raise ValueError(f"vol must be one of {VOLS}")
    r = np.asarray(r, dtype=float).ravel()
    if r.size < 250 or not np.all(np.isfinite(r)):
        raise ValueError("need at least 250 finite observations to fit GARCH")
    o = 0 if vol == "garch" else 1
    rs = r * SCALE
    am = _make_model(rs, vol, dist)
    res = am.fit(disp="off", options={"maxiter": 1000})
    ok = _plausible(res, vol, dist, rs)
    if vol == "egarch":
        # EGARCH likelihoods are badly behaved on 250-500 obs; retry from standard starts
        # and keep the plausible optimum with the highest likelihood.
        cands = [res]
        for sv in _egarch_starts(rs, dist):
            try:
                cands.append(am.fit(disp="off", starting_values=sv, options={"maxiter": 1000}))
            except (ValueError, np.linalg.LinAlgError):
                continue
        good = [c for c in cands if _plausible(c, vol, dist, rs)]
        if good:
            res, ok = max(good, key=lambda c: c.loglikelihood), True
    p = res.params
    fc = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    # log-likelihood of unscaled returns: each density picks up a Jacobian log(SCALE)
    ll = float(res.loglikelihood + r.size * np.log(SCALE))
    k = len(p)
    if vol == "egarch":
        # ln s2 (scaled) = omega_s + ... + beta ln s2 (scaled); unscaled: s2 = s2_scaled / SCALE^2
        omega = float(p["omega"]) - (1 - float(p["beta[1]"])) * np.log(SCALE**2)
    else:
        omega = float(p["omega"]) / SCALE**2
    cv = np.asarray(res.conditional_volatility, dtype=float)
    return GarchFit(
        dist=dist,
        mu=float(p["mu"]) / SCALE,
        omega=omega,
        alpha=float(p["alpha[1]"]),
        beta=float(p["beta[1]"]),
        nu=float(p["nu"]) if dist == "t" else (float(p["eta"]) if dist == "skewt" else None),
        loglik=ll,
        aic=2 * k - 2 * ll,
        bic=k * np.log(r.size) - 2 * ll,
        n_obs=r.size,
        next_sigma=float(np.sqrt(fc)) / SCALE,
        converged=bool(ok),
        vol=vol,
        gamma=float(p["gamma[1]"]) if o else 0.0,
        lam=float(p["lambda"]) if dist == "skewt" else None,
        std_resid=(rs - float(p["mu"])) / cv,
        last_sigma2=float(cv[-1] / SCALE) ** 2,
    )


def _check_skewt(nu, lam):
    if nu is None or nu <= 2 or lam is None or not -1 < lam < 1:
        raise ValueError("skew-t needs nu > 2 and -1 < lambda < 1")


def _skewt_ppf(u, nu: float, lam: float):
    from arch.univariate import SkewStudent

    return SkewStudent().ppf(np.atleast_1d(np.asarray(u, dtype=float)), np.array([nu, lam]))


def standardized_quantile(p: float, dist: str, nu: float | None = None, lam: float | None = None) -> float:
    if dist == "normal":
        return float(norm.ppf(p))
    if dist == "skewt":
        _check_skewt(nu, lam)
        return float(_skewt_ppf(p, nu, lam)[0])
    if nu is None or nu <= 2:
        raise ValueError("Student-t needs nu > 2 for unit variance")
    return float(student_t.ppf(p, nu) * np.sqrt((nu - 2) / nu))


@lru_cache(maxsize=4096)
def _skewt_es(alpha: float, nu: float, lam: float) -> float:
    tail = 1 - alpha
    val, _ = integrate.quad(lambda u: float(_skewt_ppf(u, nu, lam)[0]), 0.0, tail,
                            limit=200, epsabs=1e-12, epsrel=1e-10)
    return float(-val / tail)


def standardized_es(alpha: float, dist: str, nu: float | None = None, lam: float | None = None) -> float:
    """Expected shortfall of the standardised innovation (left tail), as a positive number."""
    if dist == "normal":
        return float(norm.pdf(norm.ppf(alpha)) / (1 - alpha))
    if dist == "skewt":
        _check_skewt(nu, lam)
        return _skewt_es(float(alpha), float(nu), float(lam))
    if nu is None or nu <= 2:
        raise ValueError("Student-t needs nu > 2 for unit variance")
    ta = student_t.ppf(alpha, nu)
    c = np.sqrt((nu - 2) / nu)
    return float(c * student_t.pdf(ta, nu) / (1 - alpha) * (nu + ta**2) / (nu - 1))


def garch_var_es(fit: GarchFit, alpha: float, include_mean: bool = True) -> tuple[float, float]:
    """One-day-ahead parametric VaR and ES in return units (positive = loss)."""
    mu = fit.mu if include_mean else 0.0
    q = standardized_quantile(1 - alpha, fit.dist, fit.nu, fit.lam)
    var = -(mu + fit.next_sigma * q)
    es = -mu + fit.next_sigma * standardized_es(alpha, fit.dist, fit.nu, fit.lam)
    return float(var), float(es)


def fhs_var_es(fit: GarchFit, z: np.ndarray, alpha: float, sigma: float | None = None,
               include_mean: bool = True) -> tuple[float, float]:
    """Filtered historical simulation (Barone-Adesi et al. 1999).

    VaR = -(mu + sigma_{t+1} * Q_z(1 - alpha)),  ES = -mu + sigma_{t+1} * ES_z(alpha),
    with Q_z / ES_z the empirical quantile / tail mean of the standardised residuals ``z``.
    """
    from mre.risk.var import historical_es, historical_var

    s = fit.next_sigma if sigma is None else sigma
    mu = fit.mu if include_mean else 0.0
    return (float(-mu + s * historical_var(z, alpha)), float(-mu + s * historical_es(z, alpha)))

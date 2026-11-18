"""Long-only portfolio optimisers (fully invested, box constraints).

* min_variance:  min w' S w
* risk_parity:   equal (or budgeted) risk contributions  w_i (S w)_i / (w' S w) = b_i
                 solved with the convex Spinu (2013) formulation
                     min_y 0.5 y' S y - sum_i b_i log y_i ,  y > 0,   w = y / sum(y)
                 which has a unique solution; if it breaches the upper bounds we fall back
                 to SLSQP on the squared deviation of risk shares, subject to the bounds.
* min_cvar:      Rockafellar-Uryasev (2000) linear programme on historical scenarios
                     min_{w, z, u} z + 1/((1-a) T) sum_t u_t
                     s.t. u_t >= -r_t' w - z,  u_t >= 0,  sum w = 1,  lb <= w <= ub
                 the optimum equals the portfolio's historical CVaR (ES) at level a.
* max_sharpe:    max (w' mu - rf) / sqrt(w' S w); depends on noisy mean estimates and is
                 included as a cautionary comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class OptResult:
    weights: np.ndarray
    method: str
    success: bool
    message: str = ""


def _bounds(n: int, lb: float, ub: float) -> list[tuple[float, float]]:
    if lb < 0 or ub > 1 or lb > ub or n * ub < 1 - 1e-12 or n * lb > 1 + 1e-12:
        raise ValueError(f"infeasible bounds lb={lb}, ub={ub} for {n} assets")
    return [(lb, ub)] * n


def _check_cov(cov) -> np.ndarray:
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.allclose(S, S.T):
        raise ValueError("cov must be a symmetric square matrix")
    if np.linalg.eigvalsh(S).min() <= 0:
        raise ValueError("cov must be positive definite")
    return S


def _clean(w: np.ndarray, lb: float, ub: float) -> np.ndarray:
    w = np.clip(w, lb, ub)
    return w / w.sum()


def min_variance(cov, lb: float = 0.0, ub: float = 1.0) -> OptResult:
    S = _check_cov(cov)
    n = S.shape[0]
    res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), jac=lambda w: 2 * S @ w,
                   bounds=_bounds(n, lb, ub), constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return OptResult(_clean(res.x, lb, ub), "min_variance", bool(res.success), res.message)


def risk_contributions(cov, w) -> np.ndarray:
    S = np.asarray(cov, dtype=float)
    w = np.asarray(w, dtype=float)
    return w * (S @ w) / (w @ S @ w)


def risk_parity(cov, budgets=None, lb: float = 0.0, ub: float = 1.0) -> OptResult:
    S = _check_cov(cov)
    n = S.shape[0]
    b = np.full(n, 1 / n) if budgets is None else np.asarray(budgets, dtype=float)
    if b.shape != (n,) or (b <= 0).any() or not np.isclose(b.sum(), 1):
        raise ValueError("budgets must be positive and sum to 1")
    _bounds(n, lb, ub)
    y0 = 1 / np.sqrt(np.diag(S))
    res = minimize(lambda y: 0.5 * y @ S @ y - b @ np.log(y), y0, jac=lambda y: S @ y - b / y,
                   bounds=[(1e-12, None)] * n, method="L-BFGS-B",
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000})
    w = res.x / res.x.sum()
    if (w <= ub + 1e-9).all() and (w >= lb - 1e-9).all():
        return OptResult(w, "risk_parity", bool(res.success), "spinu")

    def obj(w):
        return float(np.sum((risk_contributions(S, w) - b) ** 2))

    res2 = minimize(obj, _clean(w, lb, ub), bounds=_bounds(n, lb, ub),
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                    method="SLSQP", options={"ftol": 1e-16, "maxiter": 1000})
    return OptResult(_clean(res2.x, lb, ub), "risk_parity", bool(res2.success), "bounded-slsqp")


def min_cvar(scenarios, alpha: float = 0.95, lb: float = 0.0, ub: float = 1.0) -> OptResult:
    R = np.asarray(scenarios, dtype=float)
    if R.ndim != 2 or not np.all(np.isfinite(R)):
        raise ValueError("scenarios must be a finite T x N matrix")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    T, n = R.shape
    _bounds(n, lb, ub)
    # variables: [w (n), z (1), u (T)]
    c = np.r_[np.zeros(n), 1.0, np.full(T, 1 / ((1 - alpha) * T))]
    # -R w - z - u <= 0
    A_ub = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
    b_ub = np.zeros(T)
    A_eq = np.r_[np.ones(n), 0.0, np.zeros(T)][None, :]
    bounds = [(lb, ub)] * n + [(None, None)] + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if res.status != 0:
        return OptResult(np.full(n, 1 / n), "min_cvar", False, res.message)
    return OptResult(_clean(res.x[:n], lb, ub), "min_cvar", True, f"cvar={res.fun:.15g}")


def max_sharpe(mu, cov, rf: float = 0.0, lb: float = 0.0, ub: float = 1.0) -> OptResult:
    S = _check_cov(cov)
    mu = np.asarray(mu, dtype=float)
    n = S.shape[0]
    if (mu - rf).max() <= 0:
        # no asset beats rf: Sharpe maximisation is ill-posed for a long-only portfolio
        r = min_variance(S, lb, ub)
        return OptResult(r.weights, "max_sharpe", False, "all excess means <= 0; fell back to min variance")

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ S @ w)

    best = None
    for x0 in [np.full(n, 1 / n)] + [np.eye(n)[i] * 0.5 + 0.5 / n for i in range(n)]:
        res = minimize(neg_sharpe, _clean(x0, lb, ub), bounds=_bounds(n, lb, ub),
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    return OptResult(_clean(best.x, lb, ub), "max_sharpe", bool(best.success), best.message)

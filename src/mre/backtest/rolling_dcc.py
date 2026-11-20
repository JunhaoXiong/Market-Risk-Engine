"""Rolling out-of-sample portfolio VaR / ES from a multivariate DCC-GARCH model.

Same protocol as ``mre.backtest.rolling``: for a forecast date t the models see only
asset returns before t. Every ``refit_every`` days (from ``oos_start``) the four
univariate marginals and the DCC (a, b) are re-estimated on the last ``window``
days; in between, each sigma_{i,t} and Q_t are updated daily with fixed parameters.

The portfolio is the constant-weight portfolio (r_p = w' r), so its conditional mean
and volatility are mu_p = w' mu and sigma_{p,t} = sqrt(w' D_t R_t D_t w).

Models (portfolio-level tail on top of the DCC sigma_{p,t}):
dcc_normal  VaR = -(mu_p + sigma_p Phi^{-1}(1 - alpha)) (benchmark: correlation dynamics only)
dcc_fhs     filtered historical simulation: empirical quantile / tail mean of the last
            ``window`` portfolio standardised residuals z_{p,s} = (r_{p,s} - mu_p) / sigma_{p,s},
            all filtered with the latest refit's parameters and rolled forward daily
dcc_evt     GPD (peaks over threshold, ``evt_threshold`` quantile) on the same pool

``correlation="ccc"`` gives the constant-correlation ablation (a = b = 0, R = corr of the
window's residuals), which separates the effect of correlation dynamics from that of
modelling the four assets' volatilities separately.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm

from mre.backtest.rolling import _fit_garch_quiet, _refit_points
from mre.models.dcc import DCCFit, fit_ccc, fit_dcc, portfolio_sigma, q_to_r
from mre.models.evt import fit_pot
from mre.portfolio import validate_weights
from mre.risk.var import historical_es, historical_var

DCC_MODELS = ("dcc_normal", "dcc_fhs", "dcc_evt")


def _fit_dcc_quiet(z) -> DCCFit:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_dcc(z)


def _fit_ccc_quiet(z) -> DCCFit:
    return fit_ccc(z)


def _pmap(fn, jobs, n_jobs):
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
            return list(ex.map(fn, jobs, chunksize=8))
    return [fn(j) for j in jobs]


def rolling_dcc_forecasts(
    asset_returns: pd.DataFrame,
    weights,
    window: int,
    oos_start: int,
    levels=(0.95, 0.99),
    refit_every: int = 21,
    models=DCC_MODELS,
    vol: str = "gjr",
    dist: str = "t",
    evt_threshold: float = 0.90,
    n_jobs: int | None = None,
    corr_pairs=(("SPY", "TLT"),),
    correlation: str = "dcc",
) -> dict[str, pd.DataFrame]:
    """Return {model: DataFrame indexed by forecast date with var_/es_ columns and 'realized'}.

    ``correlation="ccc"`` fixes a = b = 0 (constant correlation ablation, same marginals).
    """
    if correlation not in ("dcc", "ccc"):
        raise ValueError("correlation must be 'dcc' or 'ccc'")
    w_s = validate_weights(weights, asset_returns.columns)
    X = asset_returns[w_s.index].to_numpy(dtype=float)
    w = w_s.to_numpy()
    names = list(w_s.index)
    n, N = X.shape
    if oos_start < window:
        raise ValueError("oos_start must be >= window so the first forecast has a full window")
    if oos_start >= n:
        raise ValueError("oos_start beyond sample")
    if not np.all(np.isfinite(X)):
        raise ValueError("asset returns contain missing values; align prices first")
    unknown = set(models) - set(DCC_MODELS)
    if unknown:
        raise ValueError(f"unknown models {unknown}")
    n_jobs = n_jobs or min(4, os.cpu_count() or 1)
    rp = X @ w
    dates = asset_returns.index[oos_start:]
    T = n - oos_start
    refits = _refit_points(oos_start, n, refit_every)

    # ---- estimation (all fits use only X[t - window : t])
    marg = _pmap(_fit_garch_quiet, [(X[t - window:t, i], dist, vol) for t in refits for i in range(N)],
                 n_jobs)
    marg = {t: marg[j * N:(j + 1) * N] for j, t in enumerate(refits)}
    zs = [np.column_stack([f.std_resid for f in marg[t]]) for t in refits]
    dccs = dict(zip(refits, _pmap(_fit_dcc_quiet if correlation == "dcc" else _fit_ccc_quiet, zs, n_jobs)))

    pair_idx = [(names.index(a), names.index(b)) for a, b in corr_pairs]
    cols = {f"{k}_{int(round(a * 100))}": np.empty(T) for a in levels for k in ("var", "es")}
    out = {m: {c: v.copy() for c, v in cols.items()} for m in models}
    sig_p = np.empty(T)
    corr = np.empty((T, len(pair_idx)))
    a_path, b_path = np.empty(T), np.empty(T)
    xi, xi_bound = np.full(T, np.nan), np.zeros(T)
    refit_ok = np.zeros(T, dtype=bool)

    fs = dcc = None
    for i, t in enumerate(range(oos_start, n)):
        accepted = False
        if t in dccs:
            ok = all(f.converged for f in marg[t]) and dccs[t].converged
            if ok or fs is None:
                fs, dcc, accepted = marg[t], dccs[t], True
                refit_ok[i] = ok
                mu = np.array([f.mu for f in fs])
                mu_p = float(w @ mu)
                s2 = np.array([f.next_sigma ** 2 for f in fs])
                q = dcc.q_next
                # portfolio standardised residuals for the training window
                sig_in = np.column_stack([f.sigma_path for f in fs])
                sp_in = portfolio_sigma(w, sig_in, dcc.r_path)
                pool = list((rp[t - window:t] - mu_p) / sp_in)
        if not accepted:
            # roll day t-1 into the filters with fixed parameters
            eps = X[t - 1] - mu
            pool.append((rp[t - 1] - mu_p) / sp_prev)
            q = dcc.next_q(q, eps / np.sqrt(s2))
            s2 = np.array([f.next_variance(e, v) for f, e, v in zip(fs, eps, s2)])
        R = q_to_r(q)
        sp = float(portfolio_sigma(w, np.sqrt(s2), R))
        sp_prev = sp
        sig_p[i] = sp
        corr[i] = [R[a, b] for a, b in pair_idx]
        a_path[i], b_path[i] = dcc.a, dcc.b
        z = np.asarray(pool[-window:])
        if "dcc_evt" in out:
            g = fit_pot(-z, evt_threshold)
            xi[i], xi_bound[i] = g.xi, float(g.at_bound)
        for al in levels:
            k = int(round(al * 100))
            if "dcc_normal" in out:
                out["dcc_normal"][f"var_{k}"][i] = -(mu_p + sp * norm.ppf(1 - al))
                out["dcc_normal"][f"es_{k}"][i] = -mu_p + sp * norm.pdf(norm.ppf(al)) / (1 - al)
            if "dcc_fhs" in out:
                out["dcc_fhs"][f"var_{k}"][i] = -mu_p + sp * historical_var(z, al)
                out["dcc_fhs"][f"es_{k}"][i] = -mu_p + sp * historical_es(z, al)
            if "dcc_evt" in out:
                out["dcc_evt"][f"var_{k}"][i] = -mu_p + sp * g.var(al)
                out["dcc_evt"][f"es_{k}"][i] = -mu_p + sp * g.es(al)

    frames = {}
    for m, d in out.items():
        df = pd.DataFrame(d, index=dates)
        df["sigma"] = sig_p
        for j, (a, b) in enumerate(corr_pairs):
            df[f"corr_{a}_{b}"] = corr[:, j]
        df["dcc_a"], df["dcc_b"] = a_path, b_path
        df["refit_used"] = refit_ok.astype(float)
        if m == "dcc_evt":
            df["xi"], df["xi_at_bound"] = xi, xi_bound
        df["realized"] = rp[oos_start:]
        frames[m] = df
    return frames

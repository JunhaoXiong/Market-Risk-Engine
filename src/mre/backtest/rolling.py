"""Rolling out-of-sample VaR / ES forecasting.

For every forecast date t (index >= ``oos_start``) each model sees only the
returns r[t - window : t] (GARCH additionally uses returns between its last
refit and t-1 for the daily variance update). The realised return r[t] is
attached afterwards for evaluation. No information from t or later enters a
forecast for t; ``tests/test_rolling.py`` checks this by perturbing the future.

Models
------
historical   empirical quantile / tail mean of the window
normal       window mean and standard deviation
ewma         lambda re-estimated by MLE every ``refit_every`` days on the window;
             variance recomputed daily from the window; zero mean
garch_t /    GARCH(1,1) refit every ``refit_every`` days on the window; between
garch_normal refits the conditional variance is updated daily with fixed params:
             s2[t] = omega + alpha * (r[t-1] - mu)^2 + beta * s2[t-1]

Phase 3b tail models (same refit / daily-update scheme, see ``GARCH_SPECS``):
garch_skewt  GARCH(1,1) with Hansen skewed-t innovations
gjr_t        GJR-GARCH(1,1,1) with Student-t innovations (leverage effect)
gjr_skewt    GJR-GARCH(1,1,1) with skewed-t innovations
egarch_t     EGARCH(1,1,1) with Student-t innovations
fhs_garch /  filtered historical simulation: the GARCH-t / GJR-t filter fitted on the
fhs_gjr      window gives sigma[t]; VaR/ES use the empirical quantile / tail mean of the
             standardised residuals z_s = (r_s - mu) / sigma_s for the last ``window``
             days s < t, all filtered with the parameters of the latest refit (residuals
             inside the training window come from the fit; later days are filtered
             daily with the same fixed parameters). Only data before t is used.

Phase 3c:
fhs_evt      conditional EVT (McNeil & Frey 2000): the same GJR-t filter and residual pool
             as fhs_gjr, but the left tail of the pool is a GPD fitted daily by MLE to the
             losses above the ``evt_threshold`` empirical quantile (``mre.models.evt``).
Multivariate DCC-GARCH models live in ``mre.backtest.rolling_dcc``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm

from mre.models.evt import fit_pot
from mre.models.ewma import ewma_variance, fit_ewma_lambda
from mre.models.garch import GarchFit, fit_garch, standardized_es, standardized_quantile
from mre.risk.var import historical_es, historical_var

BASE_MODELS = ("historical", "normal", "ewma", "garch_normal", "garch_t")
TAIL_MODELS = ("garch_skewt", "gjr_t", "gjr_skewt", "egarch_t", "fhs_garch", "fhs_gjr")
EVT_MODELS = ("fhs_evt",)
ALL_MODELS = BASE_MODELS + TAIL_MODELS + EVT_MODELS

# model -> (variance recursion, innovation law used in the likelihood, residual tail)
# residual tail: False = parametric, True = empirical (FHS), "evt" = GPD on the FHS pool
GARCH_SPECS = {
    "garch_normal": ("garch", "normal", False),
    "garch_t": ("garch", "t", False),
    "garch_skewt": ("garch", "skewt", False),
    "gjr_t": ("gjr", "t", False),
    "gjr_skewt": ("gjr", "skewt", False),
    "egarch_t": ("egarch", "t", False),
    "fhs_garch": ("garch", "t", True),
    "fhs_gjr": ("gjr", "t", True),
    "fhs_evt": ("gjr", "t", "evt"),
}


def _fit_garch_quiet(args) -> GarchFit:
    window_returns, dist, vol = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_garch(window_returns, dist, vol)


def _refit_points(oos_start: int, n: int, every: int) -> list[int]:
    return list(range(oos_start, n, every))


def rolling_forecasts(
    returns: pd.Series,
    window: int,
    oos_start: int,
    levels=(0.95, 0.99),
    refit_every: int = 21,
    models=BASE_MODELS,
    ewma_burn_in: int = 30,
    n_jobs: int | None = None,
    evt_threshold: float = 0.90,
) -> dict[str, pd.DataFrame]:
    """Return {model: DataFrame indexed by forecast date with var_/es_ columns and 'realized'}."""
    r = returns.to_numpy(dtype=float)
    n = r.size
    if oos_start < window:
        raise ValueError("oos_start must be >= window so the first forecast has a full window")
    if oos_start >= n:
        raise ValueError("oos_start beyond sample")
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"unknown models {unknown}")
    dates = returns.index[oos_start:]
    T = n - oos_start
    cols = {f"{k}_{int(round(a * 100))}": np.empty(T) for a in levels for k in ("var", "es")}
    out = {m: {c: v.copy() for c, v in cols.items()} for m in models}
    refits = _refit_points(oos_start, n, refit_every)
    n_jobs = n_jobs or min(4, os.cpu_count() or 1)

    # ---- Historical & Normal: recomputed every day from the window
    for i, t in enumerate(range(oos_start, n)):
        w = r[t - window:t]
        if "historical" in out:
            for a in levels:
                k = int(round(a * 100))
                out["historical"][f"var_{k}"][i] = historical_var(w, a)
                out["historical"][f"es_{k}"][i] = historical_es(w, a)
        if "normal" in out:
            mu, sd = w.mean(), w.std(ddof=1)
            for a in levels:
                k = int(round(a * 100))
                out["normal"][f"var_{k}"][i] = -(mu + sd * norm.ppf(1 - a))
                out["normal"][f"es_{k}"][i] = -mu + sd * norm.pdf(norm.ppf(a)) / (1 - a)

    # ---- EWMA
    if "ewma" in out:
        lam = None
        lam_path = np.empty(T)
        for i, t in enumerate(range(oos_start, n)):
            w = r[t - window:t]
            if lam is None or (t - oos_start) % refit_every == 0:
                lam = fit_ewma_lambda(w, ewma_burn_in).lam
            lam_path[i] = lam
            _, s2 = ewma_variance(w, lam, ewma_burn_in)
            s = np.sqrt(s2)
            for a in levels:
                k = int(round(a * 100))
                out["ewma"][f"var_{k}"][i] = -s * norm.ppf(1 - a)
                out["ewma"][f"es_{k}"][i] = s * norm.pdf(norm.ppf(a)) / (1 - a)
        out["ewma"]["lambda"] = lam_path

    # ---- GARCH family (parametric and FHS). Fits are shared between models with the
    # same (vol, dist), e.g. fhs_garch reuses the garch_t fits.
    fit_cache: dict[tuple[str, str], dict[int, GarchFit]] = {}

    def fits_for(vol: str, dist: str) -> dict[int, GarchFit]:
        key = (vol, dist)
        if key not in fit_cache:
            jobs = [(r[t - window:t], dist, vol) for t in refits]
            if n_jobs > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
                    fits = list(ex.map(_fit_garch_quiet, jobs, chunksize=8))
            else:
                fits = [_fit_garch_quiet(j) for j in jobs]
            fit_cache[key] = dict(zip(refits, fits))
        return fit_cache[key]

    for m in [m for m in models if m in GARCH_SPECS]:
        vol, dist, tail = GARCH_SPECS[m]
        fhs, evt = bool(tail), tail == "evt"
        xi = np.full(T, np.nan)
        xi_bound = np.zeros(T)
        fit_at = fits_for(vol, dist)
        sig = np.empty(T)
        pers = np.empty(T)
        nu = np.full(T, np.nan)
        lam = np.full(T, np.nan)
        gam = np.full(T, np.nan)
        refit_ok = np.zeros(T, dtype=bool)
        f, s2, z_hist, std = None, None, None, None
        for i, t in enumerate(range(oos_start, n)):
            if t in fit_at and f is not None and not fit_at[t].converged:
                # implausible optimum: keep the previous parameters and keep updating daily
                pass
            elif t in fit_at:
                f = fit_at[t]
                refit_ok[i] = f.converged
                # one-step-ahead variance for day t = arch forecast from the training window
                s2 = f.next_sigma**2
                std = {a: (standardized_quantile(1 - a, dist, f.nu, f.lam),
                           standardized_es(a, dist, f.nu, f.lam)) for a in levels}
                if fhs:
                    z_hist = list(f.std_resid)  # residuals for days t-window .. t-1
            if t not in fit_at or f is not fit_at[t]:
                eps = r[t - 1] - f.mu
                if fhs:
                    z_hist.append(eps / np.sqrt(s2))  # s2 is still sigma2[t-1] here
                s2 = f.next_variance(eps, s2)
            s = np.sqrt(s2)
            sig[i] = s
            pers[i] = f.persistence
            gam[i] = f.gamma
            if f.nu is not None:
                nu[i] = f.nu
            if f.lam is not None:
                lam[i] = f.lam
            if fhs:
                z = np.asarray(z_hist[-window:])
            if evt:
                g = fit_pot(-z, evt_threshold)
                xi[i], xi_bound[i] = g.xi, float(g.at_bound)
            for a in levels:
                k = int(round(a * 100))
                if evt:
                    out[m][f"var_{k}"][i] = -f.mu + s * g.var(a)
                    out[m][f"es_{k}"][i] = -f.mu + s * g.es(a)
                elif fhs:
                    out[m][f"var_{k}"][i] = -f.mu + s * historical_var(z, a)
                    out[m][f"es_{k}"][i] = -f.mu + s * historical_es(z, a)
                else:
                    q, es = std[a]
                    out[m][f"var_{k}"][i] = -(f.mu + s * q)
                    out[m][f"es_{k}"][i] = -f.mu + s * es
        out[m]["sigma"] = sig
        out[m]["persistence"] = pers
        # 1 on refit days whose fit was used, 0 elsewhere; failed refits = refit days with 0
        out[m]["refit_used"] = refit_ok.astype(float)
        if dist != "normal":
            out[m]["nu"] = nu
        if dist == "skewt":
            out[m]["lam"] = lam
        if vol != "garch":
            out[m]["gamma"] = gam
        if evt:
            out[m]["xi"] = xi
            out[m]["xi_at_bound"] = xi_bound

    frames = {}
    for m, d in out.items():
        df = pd.DataFrame(d, index=dates)
        df["realized"] = r[oos_start:]
        frames[m] = df
    return frames


def hits(frame: pd.DataFrame, alpha: float) -> pd.Series:
    """1 where the realised loss exceeded VaR (realised return < -VaR)."""
    k = int(round(alpha * 100))
    return (frame["realized"] < -frame[f"var_{k}"]).astype(int)


def es_check(frame: pd.DataFrame, alpha: float) -> dict:
    """Descriptive ES diagnostic: average realised loss vs average forecast ES on violation days.

    Not a formal test (formal ES backtests such as Acerbi-Szekely are a later extension).
    """
    k = int(round(alpha * 100))
    h = hits(frame, alpha).astype(bool)
    if not h.any():
        return {"mean_loss_on_violation": np.nan, "mean_es_on_violation": np.nan, "ratio": np.nan}
    loss = -frame.loc[h, "realized"].mean()
    es = frame.loc[h, f"es_{k}"].mean()
    return {"mean_loss_on_violation": float(loss), "mean_es_on_violation": float(es),
            "ratio": float(loss / es)}

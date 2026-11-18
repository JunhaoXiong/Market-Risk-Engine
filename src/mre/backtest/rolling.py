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
"""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm

from mre.models.ewma import ewma_variance, fit_ewma_lambda
from mre.models.garch import GarchFit, fit_garch, standardized_es, standardized_quantile
from mre.risk.var import historical_es, historical_var

ALL_MODELS = ("historical", "normal", "ewma", "garch_normal", "garch_t")


def _fit_garch_quiet(args) -> GarchFit:
    window_returns, dist = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_garch(window_returns, dist)


def _refit_points(oos_start: int, n: int, every: int) -> list[int]:
    return list(range(oos_start, n, every))


def rolling_forecasts(
    returns: pd.Series,
    window: int,
    oos_start: int,
    levels=(0.95, 0.99),
    refit_every: int = 21,
    models=ALL_MODELS,
    ewma_burn_in: int = 30,
    n_jobs: int | None = None,
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

    # ---- GARCH
    for m, dist in (("garch_normal", "normal"), ("garch_t", "t")):
        if m not in out:
            continue
        jobs = [(r[t - window:t], dist) for t in refits]
        if n_jobs > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
                fits = list(ex.map(_fit_garch_quiet, jobs, chunksize=8))
        else:
            fits = [_fit_garch_quiet(j) for j in jobs]
        fit_at = dict(zip(refits, fits))
        sig = np.empty(T)
        nu = np.full(T, np.nan)
        pers = np.empty(T)
        f, s2 = None, None
        for i, t in enumerate(range(oos_start, n)):
            if t in fit_at:
                f = fit_at[t]
                s2 = f.next_sigma**2
            else:
                s2 = f.omega + f.alpha * (r[t - 1] - f.mu) ** 2 + f.beta * s2
            s = np.sqrt(s2)
            sig[i] = s
            pers[i] = f.persistence
            if f.nu is not None:
                nu[i] = f.nu
            for a in levels:
                k = int(round(a * 100))
                out[m][f"var_{k}"][i] = -(f.mu + s * standardized_quantile(1 - a, dist, f.nu))
                out[m][f"es_{k}"][i] = -f.mu + s * standardized_es(a, dist, f.nu)
        out[m]["sigma"] = sig
        out[m]["persistence"] = pers
        if dist == "t":
            out[m]["nu"] = nu

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

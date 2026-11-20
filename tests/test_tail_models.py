"""Phase 3b tail models: skewed-t innovations, GJR / EGARCH asymmetry and FHS."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from mre.backtest.rolling import TAIL_MODELS, rolling_forecasts
from mre.models.garch import (
    fhs_var_es, fit_garch, garch_var_es, standardized_es, standardized_quantile,
)
from mre.risk.var import historical_es, historical_var


def _simulate_gjr(n, mu, omega, alpha, gamma, beta, nu, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta - gamma / 2)
    for t in range(n):
        eps = np.sqrt(s2) * z[t]
        r[t] = mu + eps
        s2 = omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * s2
    return r


@pytest.fixture(scope="module")
def sim_gjr():
    return _simulate_gjr(6000, 4e-4, 2e-6, 0.02, 0.12, 0.90, 7.0, seed=5)


# ---------------------------------------------------------------- skewed t
@pytest.mark.parametrize("nu", [4.0, 8.0, 30.0])
def test_skewt_with_zero_skew_equals_standardised_t(nu):
    for p in (0.01, 0.05):
        assert standardized_quantile(p, "skewt", nu, 0.0) == pytest.approx(
            standardized_quantile(p, "t", nu), rel=1e-7)
    for a in (0.95, 0.99):
        assert standardized_es(a, "skewt", nu, 0.0) == pytest.approx(
            standardized_es(a, "t", nu), rel=1e-6)


@pytest.mark.parametrize("nu,lam", [(5.0, -0.2), (10.0, 0.3)])
def test_skewt_es_matches_density_integral(nu, lam):
    """Independent check: integrate x f(x) with the density from arch's log-likelihood."""
    from arch.univariate import SkewStudent

    d = SkewStudent()
    par = np.array([nu, lam])

    def pdf(x):
        return float(np.exp(d.loglikelihood(par, np.array([x]), np.array([1.0]), individual=True)[0]))

    mass, _ = integrate.quad(pdf, -np.inf, np.inf)
    mean, _ = integrate.quad(lambda x: x * pdf(x), -np.inf, np.inf)
    var, _ = integrate.quad(lambda x: x * x * pdf(x), -np.inf, np.inf)
    assert mass == pytest.approx(1, abs=1e-7)
    assert mean == pytest.approx(0, abs=1e-6)
    assert var == pytest.approx(1, abs=1e-5)
    for a in (0.95, 0.99):
        q = standardized_quantile(1 - a, "skewt", nu, lam)
        cdf_q, _ = integrate.quad(pdf, -np.inf, q)
        assert cdf_q == pytest.approx(1 - a, abs=1e-7)
        tail, _ = integrate.quad(lambda x: x * pdf(x), -np.inf, q)
        assert standardized_es(a, "skewt", nu, lam) == pytest.approx(-tail / (1 - a), rel=1e-6)


def test_negative_skew_fattens_left_tail():
    assert standardized_quantile(0.01, "skewt", 8, -0.2) < standardized_quantile(0.01, "t", 8)
    assert standardized_es(0.99, "skewt", 8, -0.2) > standardized_es(0.99, "t", 8)


def test_skewt_rejects_bad_parameters():
    with pytest.raises(ValueError):
        standardized_quantile(0.01, "skewt", 8, None)
    with pytest.raises(ValueError):
        standardized_es(0.99, "skewt", 2.0, 0.1)


def test_skewt_fit_reports_lambda(sim_gjr):
    f = fit_garch(sim_gjr, "skewt", "garch")
    assert f.dist == "skewt" and -1 < f.lam < 1 and f.nu > 2
    var, es = garch_var_es(f, 0.99)
    assert es > var > 0


# ---------------------------------------------------------------- GJR / EGARCH
def test_gjr_recovers_leverage(sim_gjr):
    f = fit_garch(sim_gjr, "t", "gjr")
    assert f.vol == "gjr"
    assert f.gamma == pytest.approx(0.12, abs=0.05)
    assert f.beta == pytest.approx(0.90, abs=0.04)
    sym = fit_garch(sim_gjr, "t", "garch")
    assert f.loglik > sym.loglik + 10  # asymmetry clearly preferred on asymmetric data


def test_egarch_detects_leverage(sim_gjr):
    f = fit_garch(sim_gjr, "t", "egarch")
    assert f.gamma < 0  # negative shocks raise log-variance more
    assert 0.8 < f.beta < 1


@pytest.mark.parametrize("vol", ["garch", "gjr", "egarch"])
def test_daily_update_reproduces_arch_filter(sim_gjr, vol):
    """next_variance (used between refits) must equal arch's own recursion and forecast."""
    r = sim_gjr[:1500]
    f = fit_garch(r, "t", vol)
    sigma = (r - f.mu) / f.std_resid
    assert sigma[-1] ** 2 == pytest.approx(f.last_sigma2, rel=1e-10)
    pred = np.array([f.next_variance(r[s] - f.mu, sigma[s] ** 2) for s in range(r.size - 1)])
    np.testing.assert_allclose(pred, sigma[1:] ** 2, rtol=1e-8)
    assert f.next_variance(r[-1] - f.mu, f.last_sigma2) == pytest.approx(f.next_sigma**2, rel=1e-8)


def test_gjr_negative_shock_raises_variance_more(sim_gjr):
    f = fit_garch(sim_gjr, "t", "gjr")
    assert f.next_variance(-0.02, 1e-4) > f.next_variance(0.02, 1e-4)


# ---------------------------------------------------------------- FHS
def test_fhs_formula(sim_gjr):
    f = fit_garch(sim_gjr[:1000], "t", "gjr")
    z = f.std_resid
    var, es = fhs_var_es(f, z, 0.99)
    assert var == pytest.approx(-f.mu + f.next_sigma * historical_var(z, 0.99))
    assert es == pytest.approx(-f.mu + f.next_sigma * historical_es(z, 0.99))
    assert es > var > 0


# ---------------------------------------------------------------- rolling integration
@pytest.fixture(scope="module")
def series(sim_gjr):
    r = sim_gjr[:700]
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=r.size))


@pytest.fixture(scope="module")
def frames(series):
    return rolling_forecasts(series, window=300, oos_start=400, refit_every=50,
                             models=TAIL_MODELS + ("garch_t",), n_jobs=1)


def test_tail_model_shapes(series, frames):
    for m in TAIL_MODELS:
        f = frames[m]
        assert len(f) == 300
        assert (f["var_99"] > f["var_95"]).all(), m
        assert (f["es_99"] >= f["var_99"]).all(), m
        assert (f["es_95"] >= f["var_95"]).all(), m
    assert {"lam", "nu"} <= set(frames["gjr_skewt"].columns)
    assert "gamma" in frames["egarch_t"].columns


def test_fhs_first_day_and_shared_filter(series, frames):
    fit = fit_garch(series.iloc[100:400].to_numpy(), "t", "garch")
    var, es = fhs_var_es(fit, fit.std_resid, 0.99)
    first = frames["fhs_garch"].iloc[0]
    assert first["var_99"] == pytest.approx(var, rel=1e-8)
    assert first["es_99"] == pytest.approx(es, rel=1e-8)
    # FHS uses the GARCH-t filter: identical sigma path, different tail quantile
    np.testing.assert_allclose(frames["fhs_garch"]["sigma"], frames["garch_t"]["sigma"])


def test_fhs_residual_pool_rolls_daily(series, frames):
    """Between refits the residual pool = last `window` residuals under the latest fit's params."""
    i = 30                                          # day 430, last refit at 400
    t = 400 + i
    fit = fit_garch(series.iloc[100:400].to_numpy(), "t", "gjr")
    r = series.to_numpy()
    z = list(fit.std_resid)
    s2 = fit.next_sigma**2
    for u in range(400, t):
        z.append((r[u] - fit.mu) / np.sqrt(s2))
        s2 = fit.next_variance(r[u] - fit.mu, s2)
    z = np.asarray(z[-300:])
    expected = -fit.mu + np.sqrt(s2) * historical_var(z, 0.99)
    assert frames["fhs_gjr"]["var_99"].iloc[i] == pytest.approx(expected, rel=1e-8)


def test_tail_models_no_look_ahead(series, frames):
    k = 520
    shocked = series.copy()
    shocked.iloc[k:] = shocked.iloc[k:] * 5 + 0.01
    after = rolling_forecasts(shocked, window=300, oos_start=400, refit_every=50,
                              models=TAIL_MODELS, n_jobs=1)
    cut = series.index[k]
    later = series.index[k + 1:k + 60]
    for m in TAIL_MODELS:
        a = frames[m].drop(columns="realized").loc[:cut]
        b = after[m].drop(columns="realized").loc[:cut]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)
        assert (frames[m].loc[later, "var_95"] != after[m].loc[later, "var_95"]).any(), m


def test_failed_refit_keeps_previous_parameters(series, monkeypatch):
    """A refit flagged implausible must not be used; the previous fit carries on."""
    import dataclasses

    import mre.backtest.rolling as rolling

    real = rolling._fit_garch_quiet
    bad_window = series.iloc[150:450].to_numpy()  # the refit used for day 450

    def fake(args):
        f = real(args)
        if np.array_equal(args[0], bad_window):
            return dataclasses.replace(f, converged=False, beta=0.5)
        return f

    monkeypatch.setattr(rolling, "_fit_garch_quiet", fake)
    out = rolling.rolling_forecasts(series, window=300, oos_start=400, refit_every=50,
                                    models=("gjr_t",), n_jobs=1)["gjr_t"]
    assert out["refit_used"].iloc[[0, 50, 100]].tolist() == [1.0, 0.0, 1.0]
    assert out["persistence"].iloc[50] == out["persistence"].iloc[49]  # old params kept
    assert out["persistence"].iloc[100] != out["persistence"].iloc[49]
    # variance keeps updating daily through the failed refit day
    assert out["sigma"].iloc[50] != out["sigma"].iloc[49]

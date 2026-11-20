"""Phase 3c: GPD / conditional EVT and DCC-GARCH."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import genpareto

from mre.backtest.rolling import rolling_forecasts
from mre.backtest.rolling_dcc import DCC_MODELS, rolling_dcc_forecasts
from mre.models.dcc import fit_ccc, fit_dcc, portfolio_sigma, q_to_r
from mre.models.evt import GPDFit, evt_var_es, fit_gpd, fit_pot
from mre.models.garch import fit_garch


# ---------------------------------------------------------------- GPD / EVT
@pytest.mark.parametrize("xi,beta", [(0.25, 1.0), (-0.2, 0.5)])
def test_gpd_mle_recovers_parameters(xi, beta):
    y = genpareto.rvs(xi, scale=beta, size=20000, random_state=np.random.default_rng(1))
    xh, bh, ll, at_bound = fit_gpd(y)
    assert xh == pytest.approx(xi, abs=0.03)
    assert bh == pytest.approx(beta, rel=0.04)
    assert not at_bound
    assert ll == pytest.approx(genpareto.logpdf(y, xh, scale=bh).sum(), rel=1e-8)


@pytest.mark.parametrize("xi", [0.3, 0.0, -0.3])
def test_pot_var_es_match_gpd_tail(xi):
    """VaR/ES formulas vs the GPD quantile and a numerical tail integral of it."""
    g = GPDFit(xi=xi, beta=0.6, u=1.5, n=1000, n_exceed=100, loglik=0.0, at_bound=False)
    for a in (0.95, 0.99):
        p = 1 - a
        cond = 1 - p / 0.1                        # P(Y <= y | X > u) at the VaR
        assert g.var(a) == pytest.approx(1.5 + genpareto.ppf(cond, xi, scale=0.6), rel=1e-9)
        tail, _ = integrate.quad(lambda s: 1.5 + genpareto.ppf(s, xi, scale=0.6), cond, 1)
        assert g.es(a) == pytest.approx(tail / (1 - cond), rel=1e-6)
        assert g.es(a) > g.var(a)


def test_xi_zero_limit_is_continuous():
    kw = dict(beta=0.6, u=1.5, n=1000, n_exceed=100, loglik=0.0, at_bound=False)
    near = GPDFit(xi=1e-6, **kw)
    zero = GPDFit(xi=0.0, **kw)
    assert near.var(0.99) == pytest.approx(zero.var(0.99), rel=1e-5)
    assert near.es(0.99) == pytest.approx(zero.es(0.99), rel=1e-5)


def test_pot_rejects_level_inside_threshold():
    g = GPDFit(xi=0.1, beta=1, u=1, n=500, n_exceed=25, loglik=0, at_bound=False)
    with pytest.raises(ValueError):
        g.var(0.90)  # tail prob 10% > exceedance rate 5%


def test_pot_on_student_t_has_heavy_tail():
    x = np.random.default_rng(2).standard_t(3, 50000)
    f = fit_pot(x, 0.95)
    assert f.n_exceed == pytest.approx(2500, abs=2)
    assert f.xi == pytest.approx(1 / 3, abs=0.07)   # t_nu has tail index 1/nu
    var, es = evt_var_es(-x, 0.99, 0.95)              # z = -losses
    assert var == pytest.approx(np.quantile(x, 0.99), rel=0.05)
    assert es > var


# ---------------------------------------------------------------- DCC
QBAR = np.array([[1, .9, -.3, .1], [.9, 1, -.3, .1], [-.3, -.3, 1, .2], [.1, .1, .2, 1.]])


def _simulate_dcc(T, a, b, seed):
    rng = np.random.default_rng(seed)
    q, z = QBAR.copy(), np.empty((T, 4))
    for t in range(T):
        z[t] = np.linalg.cholesky(q_to_r(q)) @ rng.standard_normal(4)
        q = (1 - a - b) * QBAR + a * np.outer(z[t], z[t]) + b * q
    return z


def test_dcc_recovers_parameters():
    z = _simulate_dcc(3000, 0.04, 0.94, seed=3)
    f = fit_dcc(z)
    assert f.converged
    assert f.a == pytest.approx(0.04, abs=0.015)
    assert f.b == pytest.approx(0.94, abs=0.03)
    assert f.loglik > fit_ccc(z).loglik + 20


def test_dcc_filter_and_next_q():
    z = _simulate_dcc(400, 0.05, 0.9, seed=4)
    f = fit_dcc(z)
    q = f.qbar.copy()
    for t in range(z.shape[0]):
        np.testing.assert_allclose(f.r_path[t], q_to_r(q), atol=1e-12)
        q = f.next_q(q, z[t])
    np.testing.assert_allclose(f.q_next, q, atol=1e-12)
    assert np.allclose(np.einsum("tii->ti", f.r_path), 1)


def test_ccc_is_constant_correlation():
    z = _simulate_dcc(400, 0.05, 0.9, seed=5)
    f = fit_ccc(z)
    assert f.a == f.b == 0
    np.testing.assert_allclose(f.r_path, np.broadcast_to(f.r_path[0], f.r_path.shape))
    np.testing.assert_allclose(f.r_path[0], q_to_r(z.T @ z / 400))


def test_portfolio_sigma_matches_quadratic_form():
    w = np.array([0.4, 0.3, 0.2, 0.1])
    s = np.array([0.012, 0.015, 0.009, 0.01])
    R = q_to_r(QBAR)
    H = np.diag(s) @ R @ np.diag(s)
    assert portfolio_sigma(w, s, R) == pytest.approx(np.sqrt(w @ H @ w), rel=1e-12)


# ---------------------------------------------------------------- rolling DCC / FHS-EVT
W8 = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}


@pytest.fixture(scope="module")
def assets():
    rng = np.random.default_rng(7)
    n = 700
    z = _simulate_dcc(n, 0.04, 0.93, seed=8)
    X = np.empty_like(z)
    s2 = np.full(4, 1e-4)
    for t in range(n):
        X[t] = 2e-4 + np.sqrt(s2) * z[t] * np.sqrt(rng.chisquare(8) / 8) ** -1 * np.sqrt(6 / 8)
        s2 = 2e-6 + (0.03 + 0.1 * (X[t] < 2e-4)) * (X[t] - 2e-4) ** 2 + 0.9 * s2
    return pd.DataFrame(X, index=pd.bdate_range("2015-01-01", periods=n), columns=list(W8))


@pytest.fixture(scope="module")
def dcc_frames(assets):
    return rolling_dcc_forecasts(assets, W8, window=300, oos_start=400, refit_every=50, n_jobs=1,
                                 corr_pairs=(("A", "C"),))


def test_dcc_shapes_and_realized(assets, dcc_frames):
    rp = assets.to_numpy() @ np.array(list(W8.values()))
    for m in DCC_MODELS:
        f = dcc_frames[m]
        assert len(f) == 300 and f.index[0] == assets.index[400]
        np.testing.assert_allclose(f["realized"], rp[400:])
        assert (f["var_99"] > f["var_95"]).all(), m
        assert (f["es_99"] >= f["var_99"]).all(), m
    np.testing.assert_allclose(dcc_frames["dcc_fhs"]["sigma"], dcc_frames["dcc_evt"]["sigma"])


def test_dcc_first_day_matches_direct_fit(assets, dcc_frames):
    X = assets.to_numpy()[100:400]
    w = np.array(list(W8.values()))
    fits = [fit_garch(X[:, i], "t", "gjr") for i in range(4)]
    d = fit_dcc(np.column_stack([f.std_resid for f in fits]))
    s = np.array([f.next_sigma for f in fits])
    sp = portfolio_sigma(w, s, q_to_r(d.q_next))
    first = dcc_frames["dcc_fhs"].iloc[0]
    assert first["sigma"] == pytest.approx(sp, rel=1e-6)
    assert first["dcc_a"] == pytest.approx(d.a, rel=1e-6)
    mu_p = w @ np.array([f.mu for f in fits])
    from scipy.stats import norm
    assert dcc_frames["dcc_normal"].iloc[0]["var_99"] == pytest.approx(
        -(mu_p + sp * norm.ppf(0.01)), rel=1e-6)


def test_dcc_daily_update_between_refits(dcc_frames):
    f = dcc_frames["dcc_fhs"]
    assert f["sigma"].iloc[1] != f["sigma"].iloc[0]
    assert f["corr_A_C"].iloc[1] != f["corr_A_C"].iloc[0]       # correlation moves daily
    assert (f["dcc_a"].iloc[:50] == f["dcc_a"].iloc[0]).all()  # parameters fixed until refit


def test_ccc_correlation_fixed_between_refits(assets):
    f = rolling_dcc_forecasts(assets, W8, 300, 400, refit_every=50, models=("dcc_fhs",), n_jobs=1,
                              corr_pairs=(("A", "C"),), correlation="ccc")["dcc_fhs"]
    c = f["corr_A_C"].to_numpy()
    assert np.allclose(c[:50], c[0]) and not np.isclose(c[50], c[0])
    assert (f["dcc_a"] == 0).all()


def test_dcc_no_look_ahead(assets, dcc_frames):
    k = 520
    shocked = assets.copy()
    shocked.iloc[k:] = shocked.iloc[k:] * 5 + 0.01
    after = rolling_dcc_forecasts(shocked, W8, 300, 400, refit_every=50, n_jobs=1,
                                  corr_pairs=(("A", "C"),))
    cut, later = assets.index[k], assets.index[k + 1:k + 60]
    for m in DCC_MODELS:
        a = dcc_frames[m].drop(columns="realized").loc[:cut]
        b = after[m].drop(columns="realized").loc[:cut]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)
        assert (dcc_frames[m].loc[later, "var_95"] != after[m].loc[later, "var_95"]).any(), m


def test_dcc_rejects_bad_input(assets):
    with pytest.raises(ValueError):
        rolling_dcc_forecasts(assets, W8, 300, 200)
    with pytest.raises(ValueError):
        rolling_dcc_forecasts(assets, W8, 300, 400, correlation="xyz")


@pytest.fixture(scope="module")
def port(assets):
    return pd.Series(assets.to_numpy() @ np.array(list(W8.values())), index=assets.index)


def test_fhs_evt_shares_filter_and_uses_gpd(port):
    fr = rolling_forecasts(port, 300, 400, refit_every=50, models=("fhs_gjr", "fhs_evt"), n_jobs=1)
    np.testing.assert_allclose(fr["fhs_evt"]["sigma"], fr["fhs_gjr"]["sigma"])
    fit = fit_garch(port.iloc[100:400].to_numpy(), "t", "gjr")
    g = fit_pot(-fit.std_resid, 0.90)
    first = fr["fhs_evt"].iloc[0]
    assert first["var_99"] == pytest.approx(-fit.mu + fit.next_sigma * g.var(0.99), rel=1e-8)
    assert first["es_99"] == pytest.approx(-fit.mu + fit.next_sigma * g.es(0.99), rel=1e-8)
    assert first["xi"] == pytest.approx(g.xi)
    assert (fr["fhs_evt"]["es_99"] > fr["fhs_evt"]["var_99"]).all()


def test_fhs_evt_no_look_ahead(port):
    k = 520
    a = rolling_forecasts(port, 300, 400, refit_every=50, models=("fhs_evt",), n_jobs=1)["fhs_evt"]
    shocked = port.copy()
    shocked.iloc[k:] = shocked.iloc[k:] * 5 + 0.01
    b = rolling_forecasts(shocked, 300, 400, refit_every=50, models=("fhs_evt",), n_jobs=1)["fhs_evt"]
    cut = port.index[k]
    pd.testing.assert_frame_equal(a.drop(columns="realized").loc[:cut],
                                  b.drop(columns="realized").loc[:cut], check_exact=False, rtol=1e-10)

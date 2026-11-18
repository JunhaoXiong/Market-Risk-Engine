import numpy as np
import pytest

from mre.optimization.optimizers import (
    max_sharpe, min_cvar, min_variance, risk_contributions, risk_parity,
)
from mre.risk.var import historical_es


@pytest.fixture
def cov():
    vols = np.array([0.012, 0.015, 0.009, 0.011])
    C = np.array([[1, .9, -.3, .1], [.9, 1, -.3, .1], [-.3, -.3, 1, .2], [.1, .1, .2, 1]])
    return np.outer(vols, vols) * C


def test_min_variance_two_asset_closed_form():
    s1, s2, rho = 0.2, 0.1, 0.3
    S = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    res = min_variance(S)
    np.testing.assert_allclose(res.weights, [w1, 1 - w1], atol=1e-6)


def test_min_variance_respects_bounds_and_beats_alternatives(cov):
    res = min_variance(cov, ub=0.6)
    w = res.weights
    assert res.success and np.isclose(w.sum(), 1) and (w <= 0.6 + 1e-9).all() and (w >= -1e-12).all()
    rng = np.random.default_rng(0)
    for _ in range(500):
        x = rng.dirichlet(np.ones(4))
        if (x <= 0.6).all():
            assert w @ cov @ w <= x @ cov @ x + 1e-12


def test_risk_parity_equalises_contributions(cov):
    res = risk_parity(cov)
    np.testing.assert_allclose(risk_contributions(cov, res.weights), 0.25, atol=1e-6)
    assert res.message == "spinu"


def test_risk_parity_diagonal_is_inverse_vol():
    vols = np.array([0.1, 0.2, 0.4])
    res = risk_parity(np.diag(vols**2))
    expected = (1 / vols) / (1 / vols).sum()
    np.testing.assert_allclose(res.weights, expected, atol=1e-7)


def test_risk_parity_budgets(cov):
    b = np.array([0.4, 0.3, 0.2, 0.1])
    res = risk_parity(cov, budgets=b)
    np.testing.assert_allclose(risk_contributions(cov, res.weights), b, atol=1e-6)


def test_risk_parity_with_binding_cap():
    S = np.diag([0.01**2, 0.2**2, 0.2**2])  # unconstrained ERC puts ~91% in asset 0
    res = risk_parity(S, ub=0.5)
    assert res.message == "bounded-slsqp"
    assert res.weights[0] == pytest.approx(0.5, abs=1e-6)
    assert np.isclose(res.weights.sum(), 1)


def test_min_cvar_equals_historical_es_and_beats_grid():
    rng = np.random.default_rng(1)
    R = np.column_stack([rng.standard_t(3, 400) * 0.012, rng.standard_normal(400) * 0.008])
    res = min_cvar(R, alpha=0.95)
    w = res.weights
    port = R @ w
    # with (1-a)T = 20 integer, the RU optimum equals the mean of the 20 worst losses
    worst = -np.sort(port)[:20].mean()
    assert float(res.message.split("=")[1]) == pytest.approx(worst, rel=1e-6)
    grid = np.linspace(0, 1, 1001)
    best = min(-np.sort(R @ np.array([g, 1 - g]))[:20].mean() for g in grid)
    assert worst <= best + 1e-9
    assert historical_es(port, 0.95, method="lower") >= worst - 1e-9


def test_min_cvar_bounds():
    rng = np.random.default_rng(2)
    R = rng.standard_normal((300, 3)) * [0.02, 0.01, 0.001]
    res = min_cvar(R, 0.95, ub=0.5)
    assert (res.weights <= 0.5 + 1e-9).all() and np.isclose(res.weights.sum(), 1)


def test_max_sharpe_unconstrained_tangency():
    S = np.array([[0.04, 0.006], [0.006, 0.01]])
    mu = np.array([0.08, 0.05])
    t = np.linalg.solve(S, mu)
    t /= t.sum()  # both positive here, so long-only constraint is not binding
    res = max_sharpe(mu, S)
    np.testing.assert_allclose(res.weights, t, atol=1e-5)


def test_max_sharpe_falls_back_when_no_positive_mean(cov):
    res = max_sharpe(np.full(4, -0.001), cov)
    assert not res.success and "fell back" in res.message


@pytest.mark.parametrize("lb,ub", [(0.3, 1.0), (0.0, 0.2)])
def test_infeasible_bounds(cov, lb, ub):
    with pytest.raises(ValueError):
        min_variance(cov, lb, ub)


def test_bad_cov():
    with pytest.raises(ValueError):
        min_variance(np.array([[1.0, 2.0], [2.0, 1.0]]))

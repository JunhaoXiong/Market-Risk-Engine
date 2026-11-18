import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mre.attribution.euler import (
    historical_component_es, parametric_decomposition, rolling_pct_contribution,
)
from mre.risk.var import historical_es

IDX = ["SPY", "QQQ", "TLT", "GLD"]
W = {"SPY": 0.4, "QQQ": 0.3, "TLT": 0.2, "GLD": 0.1}


@pytest.fixture
def cov():
    vols = np.array([0.012, 0.015, 0.009, 0.011])
    C = np.array([[1, .9, -.3, .1], [.9, 1, -.3, .1], [-.3, -.3, 1, .2], [.1, .1, .2, 1]])
    return pd.DataFrame(np.outer(vols, vols) * C, index=IDX, columns=IDX)


@pytest.mark.parametrize("measure", ["var", "es"])
def test_components_sum_to_total(cov, measure):
    d = parametric_decomposition(cov, W, 0.99, 1_000_000, measure=measure)
    assert d["component"].sum() == pytest.approx(d.attrs["total"])
    assert d["pct_contribution"].sum() == pytest.approx(1.0)


def test_total_matches_closed_form(cov):
    w = np.array(list(W.values()))
    s = np.sqrt(w @ cov.to_numpy() @ w)
    d = parametric_decomposition(cov, W, 0.99, 1.0)
    assert d.attrs["total"] == pytest.approx(norm.ppf(0.99) * s)


def test_marginal_is_numerical_derivative(cov):
    mu = pd.Series([3e-4, 4e-4, 1e-4, 2e-4], index=IDX)
    d = parametric_decomposition(cov, W, 0.99, 1.0, mean=mu)
    h = 1e-6
    for a in IDX:
        up = dict(W)
        up[a] += h
        dn = dict(W)
        dn[a] -= h
        # bypass sum-to-one: the function only needs the weight vector
        f = lambda ww: parametric_decomposition(cov, ww, 0.99, 1.0, mean=mu).attrs["total"]
        assert d.loc[a, "marginal"] == pytest.approx((f(up) - f(dn)) / (2 * h), rel=1e-5)
    assert d["component"].sum() == pytest.approx(d.attrs["total"])


def test_uncorrelated_assets_contribute_by_variance_share():
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["a", "b"], columns=["a", "b"])
    d = parametric_decomposition(cov, {"a": 0.5, "b": 0.5}, 0.95)
    var_share = np.array([0.25 * 0.04, 0.25 * 0.01])
    np.testing.assert_allclose(d["pct_contribution"], var_share / var_share.sum())


def test_hedge_has_negative_contribution(cov):
    d = parametric_decomposition(cov, W, 0.99)
    assert d.loc["QQQ", "component"] > 0
    # TLT is negatively correlated with equities, which dominate the portfolio
    assert d.loc["TLT", "component"] < d.loc["TLT", "standalone"]
    assert d.attrs["diversification_benefit"] > 0


def test_incremental_var(cov):
    d = parametric_decomposition(cov, W, 0.99, 1.0)
    w = pd.Series(W)
    w0 = w.copy()
    w0["SPY"] = 0
    s0 = np.sqrt(w0 @ cov @ w0)
    assert d.loc["SPY", "incremental"] == pytest.approx(d.attrs["total"] - norm.ppf(0.99) * s0)


def test_historical_component_es_sums_to_es():
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.standard_t(4, (3000, 4)) * 0.01, columns=IDX)
    d = historical_component_es(R, W, 0.99, 1.0)
    port = R @ pd.Series(W)
    assert d.attrs["total_es"] == pytest.approx(historical_es(port.to_numpy(), 0.99))
    assert d["pct_contribution"].sum() == pytest.approx(1.0)


def test_rolling_pct_contribution_rows_sum_to_one(cov):
    path = np.stack([cov.to_numpy(), 2 * cov.to_numpy()])
    out = rolling_pct_contribution(path, np.array(list(W.values())), [0, 1])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out.iloc[0], out.iloc[1])  # scale invariant


def test_bad_inputs(cov):
    with pytest.raises(KeyError):
        parametric_decomposition(cov, {"SPY": 1.0}, 0.99)
    with pytest.raises(ValueError):
        parametric_decomposition(cov, W, 0.99, measure="cvar")

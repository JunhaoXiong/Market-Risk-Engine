import numpy as np
import pytest

from mre.models.ewma import (
    ewma_covariance, ewma_variance, fit_ewma_cov_lambda, fit_ewma_lambda,
)


def _simulate_ewma(lam, n, seed, sigma0=0.01):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = sigma0**2
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = lam * s2 + (1 - lam) * r[t] ** 2
    return r


def test_recursion_hand_computed():
    r = np.array([0.1, -0.2, 0.3, 0.0])
    s2, nxt = ewma_variance(r, 0.9, burn_in=2)
    seed = (0.01 + 0.04) / 2
    assert np.isnan(s2[:2]).all()
    assert s2[2] == pytest.approx(seed)
    assert s2[3] == pytest.approx(0.9 * seed + 0.1 * 0.09)
    assert nxt == pytest.approx(0.9 * s2[3] + 0.1 * 0.0)


def test_no_look_ahead():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(200) * 0.01
    base, _ = ewma_variance(r, 0.94)
    shocked = r.copy()
    shocked[120:] *= 10
    after, _ = ewma_variance(shocked, 0.94)
    # forecast for day t uses returns < t, so forecasts up to day 120 are unchanged
    np.testing.assert_array_equal(base[:121], after[:121])
    assert after[121] != base[121]


def test_mle_recovers_true_lambda():
    r = _simulate_ewma(0.94, 6000, seed=1)
    fit = fit_ewma_lambda(r)
    assert fit.lam == pytest.approx(0.94, abs=0.015)
    assert not fit.at_bound


def test_mle_flags_bound_for_iid_data():
    # i.i.d. returns have constant variance -> likelihood prefers lambda -> 1
    rng = np.random.default_rng(2)
    fit = fit_ewma_lambda(rng.standard_normal(3000))
    assert fit.lam > 0.99


def test_covariance_diagonal_matches_univariate_and_is_psd():
    rng = np.random.default_rng(3)
    R = rng.standard_normal((400, 3)) @ np.array([[1, .5, 0], [0, 1, .3], [0, 0, 1]]) * 0.01
    sig, nxt = ewma_covariance(R, 0.94)
    for j in range(3):
        uni, uni_next = ewma_variance(R[:, j], 0.94)
        np.testing.assert_allclose(sig[30:, j, j], uni[30:])
        assert nxt[j, j] == pytest.approx(uni_next)
    np.testing.assert_allclose(nxt, nxt.T)
    assert np.linalg.eigvalsh(nxt).min() > 0


def test_multivariate_mle_recovers_lambda():
    rng = np.random.default_rng(4)
    lam, n, N = 0.95, 3000, 2
    L = np.array([[1.0, 0.0], [0.6, 0.8]]) * 0.01
    S = L @ L.T
    R = np.empty((n, N))
    for t in range(n):
        R[t] = np.linalg.cholesky(S) @ rng.standard_normal(N)
        S = lam * S + (1 - lam) * np.outer(R[t], R[t])
    fit, nxt = fit_ewma_cov_lambda(R)
    assert fit.lam == pytest.approx(lam, abs=0.015)
    assert nxt.shape == (2, 2)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        ewma_variance([0.1, np.nan, 0.2] * 20, 0.94)
    with pytest.raises(ValueError):
        ewma_variance(np.ones(50), 1.0)
    with pytest.raises(ValueError):
        ewma_variance(np.ones(10), 0.94, burn_in=30)

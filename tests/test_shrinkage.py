import numpy as np
import pytest

from mre.models.shrinkage import (
    ledoit_wolf_constant_correlation, ledoit_wolf_identity, portfolio_vol,
)


def _data(T, seed):
    rng = np.random.default_rng(seed)
    vols = np.array([0.012, 0.015, 0.009, 0.011])
    C = np.array([[1, .8, -.3, .1], [.8, 1, -.3, .1], [-.3, -.3, 1, .2], [.1, .1, .2, 1]])
    cov = np.outer(vols, vols) * C
    return rng.multivariate_normal(np.zeros(4), cov, T)


def _brute_force_cc_delta(X):
    """Direct loop implementation of Ledoit-Wolf (2004) constant-correlation intensity."""
    X = X - X.mean(0)
    T, N = X.shape
    S = X.T @ X / T
    s = np.sqrt(np.diag(S))
    rbar = sum(S[i, j] / (s[i] * s[j]) for i in range(N) for j in range(N) if i != j) / (N * (N - 1))
    F = np.array([[S[i, i] if i == j else rbar * s[i] * s[j] for j in range(N)] for i in range(N)])
    pi = sum(np.mean((X[:, i] * X[:, j] - S[i, j]) ** 2) for i in range(N) for j in range(N))
    rho = sum(np.mean((X[:, i] ** 2 - S[i, i]) ** 2) for i in range(N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            th_ii = np.mean((X[:, i] ** 2 - S[i, i]) * (X[:, i] * X[:, j] - S[i, j]))
            th_jj = np.mean((X[:, j] ** 2 - S[j, j]) * (X[:, i] * X[:, j] - S[i, j]))
            rho += rbar / 2 * (s[j] / s[i] * th_ii + s[i] / s[j] * th_jj)
    gamma = np.sum((F - S) ** 2)
    return max(0, min(1, (pi - rho) / gamma / T))


def test_constant_correlation_matches_brute_force():
    X = _data(300, 0)
    res = ledoit_wolf_constant_correlation(X)
    assert res.delta == pytest.approx(_brute_force_cc_delta(X), rel=1e-10)


def test_identity_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    X = _data(300, 1)
    cov, delta = sk.ledoit_wolf(X)
    res = ledoit_wolf_identity(X)
    assert res.delta == pytest.approx(delta, rel=1e-8)
    np.testing.assert_allclose(res.covariance, cov, rtol=1e-8)


@pytest.mark.parametrize("fn", [ledoit_wolf_identity, ledoit_wolf_constant_correlation])
def test_properties(fn):
    X = _data(250, 2)
    res = fn(X)
    assert 0 <= res.delta <= 1
    np.testing.assert_allclose(res.covariance, res.covariance.T)
    assert np.linalg.eigvalsh(res.covariance).min() > 0
    np.testing.assert_allclose(res.sample, np.cov(X.T, bias=True))


def test_constant_correlation_target_structure():
    res = ledoit_wolf_constant_correlation(_data(500, 3))
    F = res.target
    np.testing.assert_allclose(np.diag(F), np.diag(res.sample))
    sd = np.sqrt(np.diag(F))
    corr = F / np.outer(sd, sd)
    off = corr[~np.eye(4, dtype=bool)]
    np.testing.assert_allclose(off, off[0])


def test_intensity_shrinks_as_sample_grows():
    small = ledoit_wolf_constant_correlation(_data(60, 4)).delta
    large = ledoit_wolf_constant_correlation(_data(20000, 4)).delta
    assert large < small
    assert large < 0.05


def test_portfolio_vol():
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    assert portfolio_vol(cov, [0.5, 0.5]) == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09))

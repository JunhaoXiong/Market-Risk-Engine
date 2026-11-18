import numpy as np
import pytest
from scipy.stats import chi2

from mre.backtest.coverage_tests import (
    backtest_summary, christoffersen_cc, christoffersen_independence, kupiec_pof,
    transition_counts,
)


def test_kupiec_hand_computed():
    n, x, p = 250, 8, 0.01
    h = np.r_[np.ones(x), np.zeros(n - x)]
    phat = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    res = kupiec_pof(h, p)
    assert res.statistic == pytest.approx(lr)
    assert res.p_value == pytest.approx(chi2.sf(lr, 1))
    assert res.reject(0.05)


def test_kupiec_exact_coverage_has_zero_stat():
    h = np.r_[np.ones(10), np.zeros(990)]
    assert kupiec_pof(h, 0.01).statistic == pytest.approx(0.0, abs=1e-10)
    assert kupiec_pof(h, 0.01).p_value == pytest.approx(1.0)


def test_kupiec_zero_violations_is_finite():
    res = kupiec_pof(np.zeros(1000), 0.01)
    assert np.isfinite(res.statistic)
    assert res.statistic == pytest.approx(-2 * 1000 * np.log(0.99))
    assert res.reject(0.05)


def test_transition_counts():
    assert transition_counts([0, 1, 1, 0, 0, 1]) == (1, 2, 1, 1)


def test_independence_hand_computed():
    h = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    n00, n01, n10, n11 = transition_counts(h)
    pi01, pi11 = n01 / (n00 + n01), n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    l1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    assert christoffersen_independence(h).statistic == pytest.approx(-2 * (l0 - l1))


def test_clustered_violations_fail_independence_but_pass_kupiec():
    n = 1000
    h = np.zeros(n, dtype=int)
    h[500:510] = 1  # right count (1%), all in one cluster
    assert not kupiec_pof(h, 0.01).reject(0.05)
    assert christoffersen_independence(h).reject(0.01)
    assert christoffersen_cc(h, 0.01).reject(0.01)


def test_cc_equals_pof_plus_ind_up_to_first_observation():
    rng = np.random.default_rng(0)
    h = (rng.random(2001) < 0.02).astype(int)
    h[0] = 0
    lhs = christoffersen_cc(h, 0.01).statistic
    rhs = kupiec_pof(h[1:], 0.01).statistic + christoffersen_independence(h).statistic
    assert lhs == pytest.approx(rhs, rel=1e-8)


@pytest.mark.parametrize("p", [0.05, 0.01])
def test_size_under_null(p):
    """Under H0 (i.i.d. Bernoulli(p)) rejection rates at 5% should be near 5%."""
    rng = np.random.default_rng(42)
    sims = 2000
    rej = {"pof": 0, "cc": 0}
    for _ in range(sims):
        h = (rng.random(1000) < p).astype(int)
        rej["pof"] += kupiec_pof(h, p).reject(0.05)
        rej["cc"] += christoffersen_cc(h, p).reject(0.05)
    # discreteness of the binomial makes the exact size differ slightly from 5%
    assert 0.025 < rej["pof"] / sims < 0.08
    assert 0.025 < rej["cc"] / sims < 0.09


def test_power_against_undercoverage():
    rng = np.random.default_rng(1)
    h = (rng.random(1000) < 0.03).astype(int)  # 3% violations vs 1% target
    assert kupiec_pof(h, 0.01).reject(0.01)


@pytest.mark.parametrize("bad", [[0], [0, 2, 1], np.zeros((3, 3))])
def test_bad_hits(bad):
    with pytest.raises(ValueError):
        kupiec_pof(bad, 0.01)


def test_summary_keys():
    s = backtest_summary(np.r_[np.ones(3), np.zeros(97)], 0.01)
    assert s["violations"] == 3 and s["expected"] == pytest.approx(1.0)

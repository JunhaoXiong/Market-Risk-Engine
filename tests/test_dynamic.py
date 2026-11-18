import numpy as np
import pandas as pd
import pytest

from mre.models.ewma import ewma_covariance
from mre.optimization.dynamic import (
    ewma_cov_forecasts, risk_parity_vol_target, simulate, targets_from, vol_target_scale,
)
from mre.optimization.optimizers import risk_contributions
from mre.portfolio import portfolio_returns


@pytest.fixture
def R():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=420)
    L = np.array([[1, 0, 0], [0.6, 0.8, 0], [-0.3, 0.1, 0.9]]) * 0.01
    return pd.DataFrame(rng.standard_normal((420, 3)) @ L.T, index=idx, columns=list("abc"))


def test_vol_target_scale():
    cov = np.diag([0.01**2, 0.01**2])
    w = np.array([0.5, 0.5])
    vol = np.sqrt(w @ cov @ w * 252)
    assert vol_target_scale(w, cov, 0.5) == 1.0               # below target: no leverage
    assert vol_target_scale(w, cov, vol / 2) == pytest.approx(0.5)
    assert vol_target_scale(w, cov, None) == 1.0


def test_rp_vol_target_keeps_risk_shares_and_caps_vol():
    cov = np.array([[4e-4, 1e-4], [1e-4, 1e-4]])
    w = risk_parity_vol_target(cov, 0.05, ub=1.0)
    assert w.sum() < 1
    assert np.sqrt(w @ cov @ w * 252) == pytest.approx(0.05)
    np.testing.assert_allclose(risk_contributions(cov, w), 0.5, atol=1e-6)


def test_full_investment_daily_rebalance_matches_constant_weights(R):
    tg = np.tile([0.5, 0.3, 0.2], (len(R) - 100, 1))
    res = simulate(R, tg, np.zeros(len(R)), 100, rule="calendar", every=1, cost_bps=0)
    np.testing.assert_allclose(res.returns, portfolio_returns(R.iloc[100:], dict(zip("abc", [0.5, 0.3, 0.2]))))


def test_all_cash_earns_rf(R):
    rf = np.full(len(R), 1e-4)
    res = simulate(R, np.zeros((len(R) - 100, 3)), rf, 100, cost_bps=0)
    np.testing.assert_allclose(res.returns, 1e-4)
    assert (res.cash == 1).all()


def test_cash_and_weights_sum_to_one_and_drift(R):
    tg = np.tile([0.3, 0.2, 0.1], (len(R) - 100, 1))
    res = simulate(R, tg, np.full(len(R), 5e-5), 100, rule="calendar", every=50, cost_bps=0)
    np.testing.assert_allclose(res.weights.sum(axis=1) + res.cash, 1.0)
    assert res.weights.iloc[1, 0] != 0.3                     # drifted
    assert res.n_rebalances == int(np.ceil((len(R) - 100) / 50))


def test_threshold_rule_trades_only_on_large_deviation(R):
    T = len(R) - 100
    tg = np.tile([0.4, 0.3, 0.3], (T, 1))
    tg[150:] = [0.1, 0.3, 0.6]                               # regime change in target
    res = simulate(R, tg, np.zeros(len(R)), 100, rule="threshold", threshold=0.05, cost_bps=0)
    traded = np.flatnonzero(res.turnover.to_numpy() > 0)
    assert traded[0] == 0 and 150 in traded
    # on every non-trade day the held (drifted) weights are within the 5pp band of target
    held = res.weights.to_numpy()
    no_trade = np.setdiff1d(np.arange(T), traded)
    assert (np.abs(tg[no_trade] - held[no_trade]).max(axis=1) <= 0.05 + 1e-12).all()
    assert len(traded) < 40


def test_costs_charged_on_turnover(R):
    tg = np.tile([0.5, 0.5, 0.0], (len(R) - 100, 1))
    a = simulate(R, tg, np.zeros(len(R)), 100, rule="calendar", every=21, cost_bps=0)
    b = simulate(R, tg, np.zeros(len(R)), 100, rule="calendar", every=21, cost_bps=10)
    np.testing.assert_allclose(a.returns - b.returns, 10e-4 * b.turnover)
    assert b.turnover.iloc[0] == pytest.approx(1.0)


def test_invalid_targets(R):
    with pytest.raises(ValueError):
        simulate(R, np.tile([0.6, 0.6, 0.0], (len(R) - 100, 1)), np.zeros(len(R)), 100)
    with pytest.raises(ValueError):
        simulate(R, np.tile([-0.1, 0.6, 0.0], (len(R) - 100, 1)), np.zeros(len(R)), 100)


def test_cov_forecasts_use_trailing_window_only(R):
    covs, lam = ewma_cov_forecasts(R, window=300, oos_start=300, refit_every=40)
    i = 55
    _, direct = ewma_covariance(R.iloc[300 + i - 300:300 + i].to_numpy(), lam.iloc[i])
    np.testing.assert_allclose(covs[i], direct)
    assert lam.iloc[41] == lam.iloc[79] and lam.nunique() <= 3
    shocked = R.copy()
    shocked.iloc[360:] *= 4
    covs2, _ = ewma_cov_forecasts(shocked, 300, 300, 40)
    np.testing.assert_allclose(covs[:61], covs2[:61])         # forecasts up to day 360 unchanged
    assert not np.allclose(covs[61], covs2[61])


def test_targets_from(R):
    covs = np.stack([np.eye(3) * 1e-4] * 2)
    out = targets_from(covs, lambda S: risk_parity_vol_target(S, None, 1.0))
    np.testing.assert_allclose(out, 1 / 3)

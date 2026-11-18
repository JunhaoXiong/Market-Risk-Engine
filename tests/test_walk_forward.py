import numpy as np
import pandas as pd
import pytest

from mre.optimization.backtest import performance, walk_forward
from mre.portfolio import portfolio_returns


@pytest.fixture
def R():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    return pd.DataFrame(rng.normal(3e-4, 0.01, (400, 3)), index=idx, columns=list("abc"))


def fixed(w):
    return lambda window: np.array(w)


def test_daily_rebalance_no_cost_equals_constant_weight(R):
    w = [0.5, 0.3, 0.2]
    res = walk_forward(R, fixed(w), window=100, oos_start=100, rebalance_every=1, cost_bps=0)
    expected = portfolio_returns(R.iloc[100:], dict(zip("abc", w)))
    np.testing.assert_allclose(res.returns, expected)


def test_drift_between_rebalances_matches_buy_and_hold(R):
    w = np.array([0.5, 0.3, 0.2])
    res = walk_forward(R, fixed(w), 100, 100, rebalance_every=50, cost_bps=0)
    block = R.iloc[100:150]
    value = (w * (1 + block).cumprod()).sum(axis=1)
    np.testing.assert_allclose((1 + res.gross_returns.iloc[:50]).cumprod(), value)


def test_turnover_and_costs(R):
    w = np.array([0.5, 0.3, 0.2])
    res = walk_forward(R, fixed(w), 100, 100, rebalance_every=50, cost_bps=10)
    assert res.turnover.iloc[0] == 1.0
    assert res.returns.iloc[0] == pytest.approx(res.gross_returns.iloc[0] - 10e-4)
    # second rebalance: turnover is drift back to target
    block = R.iloc[100:150]
    drifted = w * (1 + block).prod()
    drifted = drifted / drifted.sum()
    assert res.turnover.iloc[1] == pytest.approx(np.abs(w - drifted).sum())


def test_strategy_sees_only_past_data(R):
    seen = []

    def spy(window):
        seen.append(window.index[-1])
        return np.full(3, 1 / 3)

    res = walk_forward(R, spy, 100, 150, rebalance_every=25, cost_bps=0)
    for last, rebal in zip(seen, res.weights.index):
        assert last < rebal
    assert len(seen) == len(res.weights) == 10


def test_no_look_ahead_in_returns(R):
    strat = lambda win: np.abs(win.mean().to_numpy()) / np.abs(win.mean().to_numpy()).sum()
    base = walk_forward(R, strat, 100, 100, 20, 5)
    shocked = R.copy()
    shocked.iloc[250:] *= -3
    after = walk_forward(shocked, strat, 100, 100, 20, 5)
    pd.testing.assert_series_equal(base.returns.iloc[:150], after.returns.iloc[:150])


def test_invalid_weights_rejected(R):
    with pytest.raises(ValueError, match="invalid weights"):
        walk_forward(R, fixed([0.5, 0.5, 0.5]), 100, 100)
    with pytest.raises(ValueError):
        walk_forward(R, fixed([1, 0, 0]), 100, 50)


def test_performance_hand_check():
    r = pd.Series([0.1, -0.5, 0.2, 0.0] * 63)
    p = performance(r)
    # wealth 1.1 -> 0.55 -> 0.66 -> 0.66 -> 0.726 -> 0.363 ... never regains the 1.1 peak
    wealth = (1 + r).cumprod()
    assert p["max drawdown"] == pytest.approx((wealth / wealth.cummax() - 1).min())
    assert p["max drawdown"] < -0.5
    assert p["vol"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))

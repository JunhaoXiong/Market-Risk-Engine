import numpy as np
import pandas as pd
import pytest

from mre.stress.scenarios import (
    conditional_expected_returns, conditional_shock, historical_scenario, hypothetical_shock,
)

W = {"A": 0.6, "B": 0.4}


def _prices():
    idx = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame({"A": [100.0, 90.0, 80.0, 88.0], "B": [50.0, 55.0, 55.0, 50.0]}, index=idx)


def test_buy_and_hold_hand_computed():
    res = historical_scenario(_prices(), W, "2024-01-01", "2024-01-04", 1_000_000)
    # A: -12%, B: 0%  ->  0.6 * -0.12 = -7.2%
    assert res.portfolio_return == pytest.approx(-0.072)
    assert res.pnl == pytest.approx(-72_000)
    assert sum(res.contributions.values()) == pytest.approx(res.pnl)
    # value path: 1, 0.54+0.44=0.98, 0.48+0.44=0.92, 0.528+0.4=0.928
    assert res.max_drawdown == pytest.approx(0.92 - 1)
    assert res.worst_day == pytest.approx(0.92 / 0.98 - 1)
    assert res.n_days == 3


def test_rebalanced_compounds_daily_portfolio_returns():
    px = _prices()
    res = historical_scenario(px, W, "2024-01-01", "2024-01-04", 1.0, method="rebalanced")
    d = px.pct_change().iloc[1:]
    expected = np.prod(1 + (0.6 * d["A"] + 0.4 * d["B"])) - 1
    assert res.portfolio_return == pytest.approx(expected)
    assert res.contributions == {}


def test_scenario_dates_snap_to_trading_days():
    res = historical_scenario(_prices(), W, "2023-12-30", "2024-01-06", 1.0)
    assert res.start == pd.Timestamp("2024-01-01") and res.end == pd.Timestamp("2024-01-04")


def test_scenario_validation():
    with pytest.raises(ValueError):
        historical_scenario(_prices(), W, "2024-01-01", "2024-01-01", 1.0)
    with pytest.raises(ValueError):
        historical_scenario(_prices(), W, "2024-01-01", "2024-01-04", 1.0, method="x")
    with pytest.raises(ValueError):
        historical_scenario(_prices(), {"A": 0.5, "B": 0.6}, "2024-01-01", "2024-01-04", 1.0)


def test_hypothetical_shock_is_linear_and_exact():
    res = hypothetical_shock(W, {"A": -0.20, "B": 0.05}, 1_000_000)
    assert res.pnl == pytest.approx(1_000_000 * (0.6 * -0.20 + 0.4 * 0.05))
    assert res.contributions["B"] == pytest.approx(20_000)
    partial = hypothetical_shock(W, {"A": -0.10}, 1.0)
    assert partial.asset_returns["B"] == 0.0
    with pytest.raises(KeyError):
        hypothetical_shock(W, {"Z": -0.1}, 1.0)
    with pytest.raises(ValueError):
        hypothetical_shock(W, {"A": -1.5}, 1.0)


def test_conditional_single_asset_is_regression_beta():
    cov = pd.DataFrame([[0.04, 0.012, -0.004], [0.012, 0.09, 0.0], [-0.004, 0.0, 0.01]],
                       index=list("SXY"), columns=list("SXY"))
    out = conditional_expected_returns(cov, {"S": -0.10})
    assert out["S"] == -0.10
    assert out["X"] == pytest.approx(0.012 / 0.04 * -0.10)
    assert out["Y"] == pytest.approx(-0.004 / 0.04 * -0.10)  # negatively correlated asset rises


def test_conditional_matches_simulation():
    rng = np.random.default_rng(0)
    L = np.array([[1.0, 0, 0], [0.7, 0.5, 0], [-0.2, 0.1, 0.6]]) * 0.01
    X = rng.standard_normal((400_000, 3)) @ L.T
    cov = pd.DataFrame(np.cov(X.T), index=list("abc"), columns=list("abc"))
    implied = conditional_expected_returns(cov, {"a": -0.02})
    near = np.abs(X[:, 0] + 0.02) < 0.001
    np.testing.assert_allclose(X[near, 1:].mean(0), implied[["b", "c"]], atol=8e-4)


def test_conditional_multi_asset_and_portfolio():
    cov = pd.DataFrame(np.diag([0.04, 0.09, 0.01]), index=list("abc"), columns=list("abc"))
    out = conditional_expected_returns(cov, {"a": -0.1, "b": -0.05})
    assert out["c"] == pytest.approx(0.0)  # uncorrelated -> no move
    res = conditional_shock(cov, {"a": 0.5, "b": 0.3, "c": 0.2}, {"a": -0.1}, 100.0)
    assert res.pnl == pytest.approx(100 * 0.5 * -0.1)
    assert res.method == "conditional"

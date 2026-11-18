import numpy as np
import pandas as pd
import pytest

from mre.portfolio import log_returns, pnl, portfolio_returns, simple_returns


def _tiny():
    idx = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 55.0]}, index=idx)


def test_simple_returns_hand_computed():
    r = simple_returns(_tiny())
    assert len(r) == 2
    np.testing.assert_allclose(r["A"], [0.10, -0.10])
    np.testing.assert_allclose(r["B"], [0.00, 0.10])


def test_log_returns_relation(clean_prices):
    np.testing.assert_allclose(log_returns(clean_prices), np.log1p(simple_returns(clean_prices)))


def test_portfolio_return_is_weighted_sum():
    r = simple_returns(_tiny())
    rp = portfolio_returns(r, {"A": 0.6, "B": 0.4})
    np.testing.assert_allclose(rp, [0.6 * 0.10, 0.6 * -0.10 + 0.4 * 0.10])


def test_weight_order_does_not_matter(clean_prices):
    r = simple_returns(clean_prices)
    a = portfolio_returns(r, {"SPY": 0.4, "QQQ": 0.3, "TLT": 0.2, "GLD": 0.1})
    b = portfolio_returns(r, {"GLD": 0.1, "TLT": 0.2, "QQQ": 0.3, "SPY": 0.4})
    pd.testing.assert_series_equal(a, b)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_returns(simple_returns(_tiny()), {"A": 0.6, "B": 0.6})


def test_unknown_ticker_rejected():
    with pytest.raises(KeyError):
        portfolio_returns(simple_returns(_tiny()), {"A": 0.5, "Z": 0.5})


def test_missing_returns_raise_instead_of_reweighting():
    r = simple_returns(_tiny())
    r.iloc[1, 1] = np.nan
    with pytest.raises(ValueError, match="missing"):
        portfolio_returns(r, {"A": 0.5, "B": 0.5})


def test_no_look_ahead(clean_prices):
    """Changing prices after date t must not change any return dated <= t."""
    t = 150
    base = simple_returns(clean_prices)
    shocked = clean_prices.copy()
    shocked.iloc[t + 1:] *= 3.0
    after = simple_returns(shocked)
    pd.testing.assert_frame_equal(base.iloc[:t], after.iloc[:t])


def test_pnl_scales_by_notional():
    rp = pd.Series([0.01, -0.02])
    np.testing.assert_allclose(pnl(rp, 1_000_000), [10_000, -20_000])

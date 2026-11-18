import numpy as np
import pandas as pd
import pytest

from mre.backtest.rolling import es_check, hits, rolling_forecasts
from mre.models.ewma import ewma_variance
from mre.risk.var import historical_var


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    n = 700
    s2, r = 1e-4, np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_t(6) * np.sqrt(4 / 6)
        s2 = 2e-6 + 0.08 * r[t] ** 2 + 0.9 * s2
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n))


@pytest.fixture(scope="module")
def frames(series):
    return rolling_forecasts(series, window=300, oos_start=400, refit_every=50, n_jobs=1)


def test_shapes_and_alignment(series, frames):
    for m, f in frames.items():
        assert len(f) == 300
        assert f.index[0] == series.index[400]
        np.testing.assert_array_equal(f["realized"], series.iloc[400:].to_numpy())
        assert (f["var_99"] > f["var_95"]).all(), m
        assert (f["es_99"] >= f["var_99"]).all(), m


def test_historical_matches_direct_window(series, frames):
    t = 450
    expected = historical_var(series.iloc[t - 300:t].to_numpy(), 0.99)
    assert frames["historical"].loc[series.index[t], "var_99"] == pytest.approx(expected)


def test_ewma_uses_refit_lambda_on_window(series, frames):
    f = frames["ewma"]
    lam = f["lambda"].iloc[0]
    assert (f["lambda"].iloc[:50] == lam).all()        # held fixed between refits
    _, s2 = ewma_variance(series.iloc[100:400].to_numpy(), lam)
    assert f["var_99"].iloc[0] == pytest.approx(np.sqrt(s2) * 2.3263478740, rel=1e-8)


def test_garch_daily_update_between_refits(series, frames):
    f = frames["garch_t"]
    assert f["sigma"].iloc[1] != f["sigma"].iloc[0]    # variance updated daily
    assert f["nu"].iloc[1] == f["nu"].iloc[0]          # parameters fixed until refit


def test_no_look_ahead(series, frames):
    """Perturbing returns from day k on must leave every forecast for days <= k unchanged."""
    k = 520
    shocked = series.copy()
    shocked.iloc[k:] = shocked.iloc[k:] * 5 + 0.01
    after = rolling_forecasts(shocked, window=300, oos_start=400, refit_every=50, n_jobs=1)
    cut = series.index[k]
    for m in frames:
        a = frames[m].drop(columns="realized").loc[:cut]
        b = after[m].drop(columns="realized").loc[:cut]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)
        # and forecasts after k do react to the perturbation
        later = series.index[k + 1:k + 60]  # includes the next refit (day 550)
        assert (frames[m].loc[later, "var_95"] != after[m].loc[later, "var_95"]).any(), m


def test_parallel_equals_serial(series):
    a = rolling_forecasts(series, 300, 400, refit_every=50, models=("garch_t",), n_jobs=1)
    b = rolling_forecasts(series, 300, 400, refit_every=50, models=("garch_t",), n_jobs=2)
    pd.testing.assert_frame_equal(a["garch_t"], b["garch_t"])


def test_hits_and_es_check():
    f = pd.DataFrame({"realized": [-0.03, 0.01, -0.005], "var_99": [0.02, 0.02, 0.02],
                      "es_99": [0.025, 0.025, 0.025]})
    assert hits(f, 0.99).tolist() == [1, 0, 0]
    assert es_check(f, 0.99)["ratio"] == pytest.approx(0.03 / 0.025)


def test_input_validation(series):
    with pytest.raises(ValueError):
        rolling_forecasts(series, window=300, oos_start=200)
    with pytest.raises(ValueError):
        rolling_forecasts(series, window=300, oos_start=400, models=("foo",))

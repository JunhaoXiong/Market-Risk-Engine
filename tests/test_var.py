import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mre.risk.var import historical_es, historical_var, normal_es, normal_var, risk_table


def test_historical_var_known_quantile():
    # P&L = -1, -2, ..., -100 plus gains; losses of 1..100 with 100 more zero days
    x = np.concatenate([-np.arange(1, 101, dtype=float), np.zeros(100)])
    # 200 obs; with method="lower" the 1% quantile is the 2nd smallest (index floor(0.01*199)=1)
    assert historical_var(x, 0.99, method="lower") == 99.0


def test_historical_es_hand_computed():
    x = np.array([-10.0, -8.0, -6.0, -4.0, -2.0, 0, 1, 2, 3, 4])
    # 80% VaR with method="lower": 20% quantile at index floor(0.2*9)=1 -> -8
    assert historical_var(x, 0.80, method="lower") == 8.0
    assert historical_es(x, 0.80, method="lower") == 9.0  # mean of 10 and 8


def test_normal_var_closed_form():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.02, 500)
    mu, s = x.mean(), x.std(ddof=1)
    assert normal_var(x, 0.99) == pytest.approx(-(mu + s * norm.ppf(0.01)))
    assert normal_var(x, 0.99, include_mean=False) == pytest.approx(-s * norm.ppf(0.01))
    assert normal_es(x, 0.975) == pytest.approx(-mu + s * norm.pdf(norm.ppf(0.975)) / 0.025)


def test_normal_es_975_close_to_var_99():
    # Well-known Basel FRTB fact: for a normal, ES_97.5 ~ VaR_99 (2.338 vs 2.326 sigma)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(10_000)
    assert normal_es(x, 0.975, False) / normal_var(x, 0.99, False) == pytest.approx(1.005, abs=0.002)


def test_historical_converges_to_normal_on_gaussian_data():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200_000)
    assert historical_var(x, 0.99) == pytest.approx(norm.ppf(0.99), abs=0.03)
    assert historical_es(x, 0.99) == pytest.approx(norm.pdf(norm.ppf(0.99)) / 0.01, abs=0.04)


@pytest.mark.parametrize("fn", [historical_var, historical_es, normal_var, normal_es])
def test_positive_homogeneity_and_translation(fn):
    rng = np.random.default_rng(4)
    x = rng.standard_t(4, 2_000) * 0.01
    base = fn(x, 0.99)
    assert fn(3.0 * x, 0.99) == pytest.approx(3.0 * base)
    # Adding a sure gain c reduces risk by c (for normal models only if the mean is used)
    assert fn(x + 0.005, 0.99) == pytest.approx(base - 0.005)


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_es_at_least_var(alpha):
    rng = np.random.default_rng(5)
    x = rng.standard_t(3, 5_000)
    assert historical_es(x, alpha) >= historical_var(x, alpha)
    assert normal_es(x, alpha) >= normal_var(x, alpha)


def test_higher_confidence_means_higher_risk():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(5_000)
    assert historical_var(x, 0.99) > historical_var(x, 0.95)
    assert normal_var(x, 0.99) > normal_var(x, 0.95)


@pytest.mark.parametrize("bad", [[], [1.0, np.nan], [np.inf, 1.0]])
def test_bad_inputs_raise(bad):
    with pytest.raises(ValueError):
        historical_var(bad, 0.99)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_bad_alpha_raises(alpha):
    with pytest.raises(ValueError):
        normal_var([1.0, 2.0, 3.0], alpha)


def test_risk_table_shape():
    rng = np.random.default_rng(7)
    tbl = risk_table(pd.Series(rng.standard_normal(1_000)), levels=(0.95, 0.99))
    assert len(tbl) == 4
    assert set(tbl["method"]) == {"Historical", "Normal"}
    assert (tbl["es_to_var"] >= 1).all()

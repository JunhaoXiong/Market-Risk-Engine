import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, t as student_t

from mre.models.garch import (
    fit_garch, garch_var_es, standardized_es, standardized_quantile,
)


def _simulate_garch_t(n, mu, omega, alpha, beta, nu, seed):
    rng = np.random.default_rng(seed)
    c = np.sqrt((nu - 2) / nu)
    z = rng.standard_t(nu, n) * c
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        eps = np.sqrt(s2) * z[t]
        r[t] = mu + eps
        s2 = omega + alpha * eps**2 + beta * s2
    return r, s2  # s2 is the true next-day variance


@pytest.fixture(scope="module")
def sim():
    return _simulate_garch_t(8000, 4e-4, 2e-6, 0.08, 0.90, 6.0, seed=11)


def test_parameter_recovery_t(sim):
    r, _ = sim
    fit = fit_garch(r, "t")
    assert fit.converged
    assert fit.alpha == pytest.approx(0.08, abs=0.025)
    assert fit.beta == pytest.approx(0.90, abs=0.03)
    assert fit.nu == pytest.approx(6.0, abs=1.5)
    assert fit.persistence < 1


def test_t_beats_normal_on_fat_tailed_data(sim):
    r, _ = sim
    ft, fn = fit_garch(r, "t"), fit_garch(r, "normal")
    assert ft.aic < fn.aic
    assert fn.nu is None


def test_one_step_forecast_matches_recursion(sim):
    r, _ = sim
    fit = fit_garch(r, "t")
    # Reproduce arch's filtered variance with fitted parameters and check the forecast
    eps = r - fit.mu
    from arch import arch_model
    res = arch_model(r * 100, mean="Constant", vol="GARCH", dist="t", rescale=False).fit(disp="off")
    last_s2 = (res.conditional_volatility[-1] / 100) ** 2
    expected = fit.omega + fit.alpha * eps[-1] ** 2 + fit.beta * last_s2
    assert fit.next_sigma**2 == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("nu", [3.5, 5.0, 10.0])
def test_standardized_t_has_unit_variance_and_correct_es(nu):
    c = np.sqrt((nu - 2) / nu)
    pdf = lambda z: student_t.pdf(z / c, nu) / c
    var, _ = integrate.quad(lambda z: z**2 * pdf(z), -np.inf, np.inf)
    assert var == pytest.approx(1.0, rel=1e-6)
    a = 0.99
    q = standardized_quantile(1 - a, "t", nu)
    tail, _ = integrate.quad(lambda z: z * pdf(z), -np.inf, q)
    assert standardized_es(a, "t", nu) == pytest.approx(-tail / (1 - a), rel=1e-6)


def test_t_converges_to_normal_for_large_nu():
    assert standardized_quantile(0.01, "t", 1e6) == pytest.approx(norm.ppf(0.01), abs=1e-4)
    assert standardized_es(0.99, "t", 1e6) == pytest.approx(standardized_es(0.99, "normal"), abs=1e-4)


def test_t_tail_heavier_than_normal_at_99():
    # unit-variance t has a larger 99% ES than the normal
    assert standardized_es(0.99, "t", 5) > standardized_es(0.99, "normal")


def test_var_es_ordering(sim):
    r, _ = sim
    fit = fit_garch(r, "t")
    v95, e95 = garch_var_es(fit, 0.95)
    v99, e99 = garch_var_es(fit, 0.99)
    assert 0 < v95 < v99 and v95 < e95 and v99 < e99


def test_rejects_short_or_bad_input():
    with pytest.raises(ValueError):
        fit_garch(np.zeros(100))
    with pytest.raises(ValueError):
        fit_garch(np.r_[np.zeros(300), np.nan])
    with pytest.raises(ValueError):
        fit_garch(np.zeros(300), dist="skewt")

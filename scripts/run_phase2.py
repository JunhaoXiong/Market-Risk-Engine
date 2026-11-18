"""Phase 2: fit EWMA / GARCH / shrinkage on the most recent estimation window and
produce one-day-ahead VaR / ES forecasts for the first trading day after the window.

Everything is estimated on the last ``estimation_window`` returns up to the data
end date only. The forecasts are NOT evaluated here (that would require data
after the window); rolling out-of-sample evaluation is Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from mre.config import PROJECT_ROOT, load_config
from mre.data.loader import get_prices
from mre.data.quality import align_prices
from mre.models.ewma import ewma_variance, fit_ewma_cov_lambda, fit_ewma_lambda
from mre.models.garch import fit_garch, garch_var_es
from mre.models.shrinkage import (
    ledoit_wolf_constant_correlation, ledoit_wolf_identity, portfolio_vol,
)
from mre.portfolio import portfolio_returns, simple_returns
from mre.risk.var import historical_es, historical_var, normal_es, normal_var


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    x = x - x.mean()
    n = x.size
    ac = np.array([np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(ac**2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def main() -> None:
    cfg = load_config()
    V = cfg.portfolio_value
    prices = align_prices(get_prices(cfg.tickers, cfg.start, cfg.end, cfg.cache_path, cfg.auto_adjust))
    asset_r = simple_returns(prices)
    port_r = portfolio_returns(asset_r, cfg.weights)

    W = cfg.estimation_window
    win_assets = asset_r.iloc[-W:]
    win = port_r.iloc[-W:]
    r = win.to_numpy()
    w = np.array([cfg.weights[t] for t in cfg.tickers])
    z = {a: stats.norm.ppf(1 - a) for a in cfg.confidence_levels}
    phi = {a: stats.norm.pdf(stats.norm.ppf(a)) / (1 - a) for a in cfg.confidence_levels}

    # ---- EWMA
    ew = fit_ewma_lambda(r, cfg.ewma_burn_in)
    _, rm_next = ewma_variance(r, 0.94, cfg.ewma_burn_in)
    ewc, ew_cov_next = fit_ewma_cov_lambda(win_assets.to_numpy(), cfg.ewma_burn_in)

    # ---- GARCH
    g_t = fit_garch(r, cfg.garch_primary_dist)
    g_n = fit_garch(r, cfg.garch_benchmark_dist)

    # ---- Shrinkage
    lw_i = ledoit_wolf_identity(win_assets.to_numpy())
    lw_c = ledoit_wolf_constant_correlation(win_assets.to_numpy())

    rows = []
    for a in cfg.confidence_levels:
        rows.append(("Historical (window)", a, historical_var(r, a), historical_es(r, a)))
        rows.append(("Normal, sample var (window)", a, normal_var(r, a), normal_es(r, a)))
        for name, cov in [("Normal, sample cov", lw_c.sample),
                          ("Normal, LW identity cov", lw_i.covariance),
                          ("Normal, LW const-corr cov", lw_c.covariance)]:
            s = portfolio_vol(cov, w)
            rows.append((name, a, -(r.mean() + s * z[a]), -r.mean() + s * phi[a]))
        s = np.sqrt(ew.next_variance)
        rows.append((f"EWMA (MLE lambda={ew.lam:.4f}), zero mean", a, -s * z[a], s * phi[a]))
        s = np.sqrt(rm_next)
        rows.append(("EWMA (RiskMetrics 0.94), zero mean", a, -s * z[a], s * phi[a]))
        s = portfolio_vol(ew_cov_next, w)
        rows.append((f"EWMA cov (MLE lambda={ewc.lam:.4f}), zero mean", a, -s * z[a], s * phi[a]))
        rows.append(("GARCH(1,1)-Normal [benchmark]", a, *garch_var_es(g_n, a)))
        rows.append(("GARCH(1,1)-t [primary]", a, *garch_var_es(g_t, a)))
    fc = pd.DataFrame(rows, columns=["model", "alpha", "var_ret", "es_ret"])
    fc["VaR $"] = fc["var_ret"] * V
    fc["ES $"] = fc["es_ret"] * V

    # ---- GARCH diagnostics: standardised residuals of the primary model
    from arch import arch_model
    res_t = arch_model(r * 100, mean="Constant", vol="GARCH", dist="t", rescale=False).fit(disp="off")
    zt = np.asarray(res_t.std_resid)
    lb_raw = ljung_box(r**2)
    lb_z = ljung_box(zt**2)

    def gp(g):
        return {
            "mu (daily)": f"{g.mu:.2e}", "omega": f"{g.omega:.3e}", "alpha": f"{g.alpha:.4f}",
            "beta": f"{g.beta:.4f}", "alpha+beta": f"{g.persistence:.4f}",
            "half-life (days)": f"{g.half_life_days:.1f}",
            "long-run vol (ann.)": f"{g.long_run_vol * np.sqrt(252):.2%}",
            "nu": "-" if g.nu is None else f"{g.nu:.2f}",
            "log-lik": f"{g.loglik:.1f}", "AIC": f"{g.aic:.1f}", "BIC": f"{g.bic:.1f}",
            "next-day vol (ann.)": f"{g.next_sigma * np.sqrt(252):.2%}",
            "converged": g.converged,
        }

    garch_tbl = pd.DataFrame({"GARCH-t (primary)": gp(g_t), "GARCH-Normal (benchmark)": gp(g_n)})
    fc_fmt = fc.assign(alpha=fc["alpha"].map("{:.0%}".format),
                       **{"VaR $": fc["VaR $"].map("${:,.0f}".format),
                          "ES $": fc["ES $"].map("${:,.0f}".format)})[["model", "alpha", "VaR $", "ES $"]]

    lw_tbl = pd.DataFrame({
        "target": ["identity", "constant correlation"],
        "delta": [f"{lw_i.delta:.4f}", f"{lw_c.delta:.4f}"],
        "portfolio vol (ann.)": [f"{portfolio_vol(lw_i.covariance, w) * np.sqrt(252):.2%}",
                                 f"{portfolio_vol(lw_c.covariance, w) * np.sqrt(252):.2%}"],
    })
    sample_corr = pd.DataFrame(np.corrcoef(win_assets.T), index=cfg.tickers, columns=cfg.tickers)

    report = f"""# Phase 2 Report: EWMA, GARCH and covariance shrinkage

Estimation window: last {W} daily returns, {win.index[0].date()} to {win.index[-1].date()}.
Forecasts below are one-day-ahead for the next trading day after {win.index[-1].date()};
they are **not** evaluated here (no data after the window is used). Portfolio ${V:,.0f}.

## EWMA
- Univariate (portfolio returns) lambda by Gaussian MLE: **{ew.lam:.4f}** (at bound: {ew.at_bound}); RiskMetrics benchmark 0.94
- Next-day vol (ann.): MLE {np.sqrt(ew.next_variance * 252):.2%}, lambda=0.94 {np.sqrt(rm_next * 252):.2%}
- Covariance (4 assets) common lambda by multivariate Gaussian MLE: **{ewc.lam:.4f}** (at bound: {ewc.at_bound})

## GARCH(1,1)
{garch_tbl.to_markdown()}

AIC difference (Normal - t): {g_n.aic - g_t.aic:.1f} (positive favours Student-t).
Note: a likelihood-ratio test of t vs Normal is non-standard (nu = infinity lies on the
boundary), so AIC/BIC are reported instead of a chi-square p-value.

Diagnostics (Ljung-Box, 10 lags, on squared series):
- raw squared returns: Q = {lb_raw[0]:.1f}, p = {lb_raw[1]:.3g}
- squared standardised GARCH-t residuals: Q = {lb_z[0]:.1f}, p = {lb_z[1]:.3g}
- excess kurtosis of standardised residuals: {stats.kurtosis(zt):.2f}

## Ledoit-Wolf shrinkage (asset covariance)
{lw_tbl.to_markdown(index=False)}

Sample correlation in the window:
{sample_corr.round(2).to_markdown()}

## One-day-ahead VaR / ES forecasts (positive = loss)
{fc_fmt.to_markdown(index=False)}

## Caveats
- Single estimation window; parameter stability and forecast accuracy are assessed in Phase 3 (rolling backtest).
- EWMA forecasts use a zero mean (RiskMetrics convention); GARCH and window-Normal include the sample mean.
- Covariance-based rows use the same window mean as the Normal row so only the covariance differs.
"""
    out = PROJECT_ROOT / "reports"
    (out / "phase2_report.md").write_text(report, encoding="utf-8")
    fc.to_csv(out / "phase2_forecasts.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()

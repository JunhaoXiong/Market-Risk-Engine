"""Phase 4: stress testing and risk attribution.

As-of date for attribution: last date in the data; every estimate uses only the
``estimation_window`` returns up to that date. Historical scenarios replay realised
returns on today's weights, which is the purpose of a stress test, not a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mre.attribution.euler import (
    historical_component_es, parametric_decomposition, rolling_pct_contribution,
)
from mre.config import PROJECT_ROOT, load_config
from mre.data.loader import get_prices
from mre.data.quality import align_prices
from mre.models.ewma import ewma_covariance, fit_ewma_cov_lambda
from mre.models.shrinkage import ledoit_wolf_constant_correlation
from mre.portfolio import simple_returns
from mre.stress.scenarios import (
    conditional_expected_returns, conditional_shock, historical_scenario, hypothetical_shock,
)


def pct(x):
    return f"{x:+.1%}"


def usd(x):
    return f"${x:,.0f}"


def main() -> None:
    cfg = load_config()
    V, tick = cfg.portfolio_value, cfg.tickers
    st, at = cfg.raw["stress"], cfg.raw["attribution"]
    prices = align_prices(get_prices(cfg.tickers, cfg.start, cfg.end, cfg.cache_path, cfg.auto_adjust))
    R = simple_returns(prices)
    w = pd.Series(cfg.weights)

    # ---------------- historical scenarios
    hist_rows = []
    for sc in st["historical"]:
        bh = historical_scenario(prices, w, sc["start"], sc["end"], V, sc["name"], st["method_primary"])
        rb = historical_scenario(prices, w, sc["start"], sc["end"], V, sc["name"], st["method_benchmark"])
        row = {"scenario": sc["name"], "period": f"{bh.start.date()} to {bh.end.date()}",
               "days": bh.n_days, "buy&hold": pct(bh.portfolio_return), "P&L": usd(bh.pnl),
               "rebalanced": pct(rb.portfolio_return), "max DD (b&h)": pct(bh.max_drawdown),
               "worst day": pct(bh.worst_day)}
        row.update({t: pct(bh.asset_returns[t]) for t in tick})
        row["largest loss from"] = pd.Series(bh.contributions).idxmin()
        hist_rows.append(row)
    hist = pd.DataFrame(hist_rows)

    # ---------------- hypothetical
    hyp_rows = []
    for sc in st["hypothetical"]:
        res = hypothetical_shock(w, sc["shocks"], V, sc["name"])
        row = {"scenario": sc["name"], "portfolio": pct(res.portfolio_return), "P&L": usd(res.pnl)}
        row.update({f"{t} shock": pct(res.asset_returns[t]) for t in tick})
        hyp_rows.append(row)
    hyp = pd.DataFrame(hyp_rows)

    # ---------------- conditional
    shocks = st["conditional"]["shocks"]
    cond_rows = []
    W_ = cfg.estimation_window
    lam_fit, ew_next = fit_ewma_cov_lambda(R.iloc[-W_:].to_numpy(), cfg.ewma_burn_in)
    ew_cov = pd.DataFrame(ew_next, index=tick, columns=tick)
    periods = [("Current EWMA (as of %s)" % R.index[-1].date(), ew_cov)]
    for p in st["conditional"]["covariance_periods"]:
        periods.append((p["name"], R.loc[p["start"]:p["end"]].cov()))
    for name, cov in periods:
        implied = conditional_expected_returns(cov, shocks)
        res = conditional_shock(cov, w, shocks, V, name)
        corr = cov.loc["SPY", "TLT"] / np.sqrt(cov.loc["SPY", "SPY"] * cov.loc["TLT", "TLT"])
        row = {"covariance": name, "corr(SPY,TLT)": f"{corr:+.2f}"}
        row.update({t: pct(implied[t]) for t in tick})
        row.update({"portfolio": pct(res.portfolio_return), "P&L": usd(res.pnl)})
        cond_rows.append(row)
    cond = pd.DataFrame(cond_rows)

    # ---------------- attribution as of the last date
    win = R.iloc[-W_:]
    covs = {
        f"EWMA (MLE lambda={lam_fit.lam:.4f}) [primary]": ew_cov,
        f"Sample ({W_}d)": win.cov(),
        f"Ledoit-Wolf const-corr ({W_}d)": pd.DataFrame(
            ledoit_wolf_constant_correlation(win.to_numpy()).covariance, index=tick, columns=tick),
    }
    attr_sections = []
    summary_rows = []
    for name, cov in covs.items():
        for a in cfg.confidence_levels:
            d = parametric_decomposition(cov, w, a, V)
            summary_rows.append({"covariance": name, "alpha": f"{a:.0%}", "VaR": usd(d.attrs["total"]),
                                 **{t: f"{d.loc[t, 'pct_contribution']:.1%}" for t in tick},
                                 "diversification": usd(d.attrs["diversification_benefit"])})
    primary_name = next(iter(covs))
    d99 = parametric_decomposition(covs[primary_name], w, 0.99, V)
    e99 = parametric_decomposition(covs[primary_name], w, 0.99, V, measure="es")
    detail = pd.DataFrame({
        "weight": d99["weight"].map("{:.0%}".format),
        "standalone VaR": d99["standalone"].map(usd),
        "marginal VaR (per 100% wt)": d99["marginal"].map(usd),
        "component VaR": d99["component"].map(usd),
        "% of VaR": d99["pct_contribution"].map("{:.1%}".format),
        "incremental VaR": d99["incremental"].map(usd),
        "component ES": e99["component"].map(usd),
    })
    hces = historical_component_es(win, w, 0.99, V, cfg.quantile_method)
    hces_tbl = pd.DataFrame({"component ES": hces["component_es"].map(usd),
                             "% of ES": hces["pct_contribution"].map("{:.1%}".format)})

    # ---------------- rolling contribution path (fixed lambda, descriptive)
    lam_roll = float(at["rolling_lambda"])
    path, _ = ewma_covariance(R.to_numpy(), lam_roll, cfg.ewma_burn_in)
    valid = slice(cfg.ewma_burn_in, None)
    contrib_path = rolling_pct_contribution(path[valid], w.to_numpy(), R.index[valid])
    contrib_path.columns = tick
    out = PROJECT_ROOT / "reports"
    contrib_path.to_csv(out / "phase4_rolling_risk_contribution.csv", float_format="%.5f")
    yearly = contrib_path.groupby(contrib_path.index.year).mean()
    yearly.index.name = "year"
    yearly_fmt = yearly.map(lambda v: f"{v:.0%}")

    report = f"""# Phase 4 Report: Stress testing and risk attribution

Portfolio: {", ".join(f"{t} {cfg.weights[t]:.0%}" for t in tick)}; ${V:,.0f}. Positive P&L = gain.

## 1. Historical scenarios
Primary: buy-and-hold (positions fixed at scenario start). Benchmark: daily rebalanced.
Asset columns are each asset's cumulative return over the scenario.

{hist.to_markdown(index=False)}

## 2. Hypothetical instantaneous shocks
P&L = sum_i w_i * shock_i * V (exact for simple returns).

{hyp.to_markdown(index=False)}

## 3. Conditional stress: SPY {shocks['SPY']:+.0%}
Other assets move by their conditional expectation E[r | r_SPY] = Cov(r, r_SPY) / Var(r_SPY) * shock
(zero-mean joint normal), using covariance estimated in each period.

{cond.to_markdown(index=False)}

## 4. Risk attribution as of {R.index[-1].date()} (window {win.index[0].date()} to {win.index[-1].date()})
Gaussian VaR, zero mean, Euler decomposition (components sum to total VaR).

{pd.DataFrame(summary_rows).to_markdown(index=False)}

### Primary (EWMA) detail at 99%
{detail.to_markdown()}

Marginal VaR is dVaR/dw_i scaled by the notional, i.e. the VaR change per +100% weight;
incremental VaR is the VaR reduction from closing that position into cash.

### Historical component ES at 99% (window, no distributional assumption)
Tail days: {hces.attrs['tail_days']}; total ES {usd(hces.attrs['total_es'])}.
Caveat: with a {W_}-day window the 99% tail holds only {hces.attrs['tail_days']} days, so this split is
noisy; it is a cross-check on the parametric split, not a replacement.

{hces_tbl.to_markdown()}

## 5. Risk contribution over time (EWMA, fixed lambda = {lam_roll}, yearly average share of portfolio variance)
A fixed RiskMetrics lambda is used for this descriptive path so that no parameter is
estimated with future data. Daily series: `reports/phase4_rolling_risk_contribution.csv`.

{yearly_fmt.to_markdown()}
"""
    (out / "phase4_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

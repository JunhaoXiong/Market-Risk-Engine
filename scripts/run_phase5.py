"""Phase 5: risk-parity + volatility-target allocation with cash and threshold rebalancing.

Every target for day t uses only data up to the close of t-1. Out-of-sample period
matches Phase 3 (from return #oos_start_index). Cash earns the 13-week T-bill rate
(^IRX) quoted at t-1.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mre.config import PROJECT_ROOT, load_config
from mre.data.loader import get_cash_rate, get_prices
from mre.data.quality import align_prices
from mre.optimization.dynamic import ewma_cov_forecasts, risk_parity_vol_target, simulate, targets_from
from mre.optimization.optimizers import min_cvar, min_variance
from mre.portfolio import simple_returns
from mre.risk.var import historical_es, historical_var


def stats(r: pd.Series, rf: pd.Series, res=None, periods=252) -> dict:
    wealth = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = wealth.iloc[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(periods)
    ex = r - rf
    dd = (wealth / wealth.cummax() - 1).min()
    rv = r.rolling(63).std() * np.sqrt(periods)
    out = {
        "CAGR": cagr, "vol": vol, "Sharpe (excess)": ex.mean() * periods / (ex.std(ddof=1) * np.sqrt(periods)),
        "max DD": dd, "Calmar": cagr / abs(dd), "VaR99": historical_var(r.to_numpy(), 0.99),
        "ES99": historical_es(r.to_numpy(), 0.99), "max 63d vol": rv.max(),
        "2020": (1 + r.loc["2020"]).prod() - 1, "2022": (1 + r.loc["2022"]).prod() - 1,
    }
    if res is not None:
        out.update({"avg cash": res.cash.mean(), "turnover/yr": res.turnover.sum() / yrs,
                    "trades/yr": res.n_rebalances / yrs})
    return out


def main() -> None:
    cfg = load_config()
    oc = cfg.raw["optimization"]
    prices = align_prices(get_prices(cfg.tickers, cfg.start, cfg.end, cfg.cache_path, cfg.auto_adjust))
    R = simple_returns(prices)
    start = cfg.oos_start_index
    W, ub, cost = int(oc["window"]), float(oc["upper_bound"]), float(oc["cost_bps"])

    rate = get_cash_rate(cfg.start, cfg.end, PROJECT_ROOT / oc["cash_rate_cache"])
    # rate is a level (not a price), so carrying the last quote over a missing day is acceptable;
    # shift(1): the rate earned on day t is the quote known at the close of t-1
    rf = (rate.reindex(prices.index).ffill().shift(1) / 252.0).reindex(R.index)
    if rf.iloc[start:].isna().any():
        raise ValueError("cash rate missing in the out-of-sample period")
    rf = rf.fillna(0.0)
    rf_oos = rf.iloc[start:]

    t0 = time.time()
    covs, lam = ewma_cov_forecasts(R, W, start, int(oc["lambda_refit_every"]), cfg.ewma_burn_in)
    print(f"covariance path: {time.time() - t0:.0f}s, lambda median {lam.median():.4f}")

    vt, th = float(oc["vol_target"]), float(oc["threshold"])
    every = int(oc["calendar_rebalance_every"])
    rp = {v: targets_from(covs, lambda S, v=v: risk_parity_vol_target(S, v, ub))
          for v in [vt, *oc["vol_target_sensitivity"], None]}

    def monthly_targets(fn):
        T = len(R) - start
        out = np.empty((T, len(cfg.tickers)))
        for i in range(T):
            if i % every == 0:
                cur = fn(start + i, i)
            out[i] = cur
        return out

    X = R.to_numpy()
    runs = {}
    runs[f"RP + VT {vt:.0%}, threshold {th:.1%} [primary]"] = simulate(R, rp[vt], rf, start, "threshold", th, cost_bps=cost)
    runs[f"RP + VT {vt:.0%}, monthly"] = simulate(R, rp[vt], rf, start, "calendar", every=every, cost_bps=cost)
    for t in oc["threshold_sensitivity"]:
        runs[f"RP + VT {vt:.0%}, threshold {t:.1%}"] = simulate(R, rp[vt], rf, start, "threshold", float(t), cost_bps=cost)
    for v in oc["vol_target_sensitivity"]:
        runs[f"RP + VT {v:.0%}, threshold {th:.1%}"] = simulate(R, rp[v], rf, start, "threshold", th, cost_bps=cost)
    runs[f"RP (no VT, fully invested), threshold {th:.1%}"] = simulate(R, rp[None], rf, start, "threshold", th, cost_bps=cost)
    runs["Min variance (EWMA cov), monthly"] = simulate(
        R, monthly_targets(lambda t, i: min_variance(covs[i], ub=ub).weights), rf, start, "calendar", every=every, cost_bps=cost)
    a = float(oc["cvar_alpha"])
    runs[f"Min CVaR {a:.0%} ({W}d scenarios), monthly"] = simulate(
        R, monthly_targets(lambda t, i: min_cvar(X[t - W:t], a, ub=ub).weights), rf, start, "calendar", every=every, cost_bps=cost)
    bench = np.tile([cfg.weights[t] for t in cfg.tickers], (len(R) - start, 1))
    runs["Benchmark 40/30/20/10, monthly"] = simulate(R, bench, rf, start, "calendar", every=every, cost_bps=cost)

    table = pd.DataFrame({k: stats(v.returns, rf_oos, v) for k, v in runs.items()}).T
    out = PROJECT_ROOT / "reports"
    table.to_csv(out / "phase5_results.csv")
    prim = runs[next(iter(runs))]
    pd.concat({"weights": prim.weights, "targets": prim.targets}, axis=1).assign(
        cash=prim.cash, turnover=prim.turnover, net=prim.returns).to_csv(
        out / "phase5_primary_path.csv.gz", float_format="%.6g")

    fmt = table.copy()
    for c in ["CAGR", "vol", "max DD", "VaR99", "ES99", "max 63d vol", "2020", "2022", "avg cash", "turnover/yr"]:
        fmt[c] = table[c].map(lambda x, c=c: f"{x:.2%}" if c in ("VaR99", "ES99") else f"{(0.0 if abs(x) < 5e-4 else x):.1%}")
    for c in ["Sharpe (excess)", "Calmar", "trades/yr"]:
        fmt[c] = table[c].map("{:.2f}".format)

    yearly_w = prim.weights.assign(cash=prim.cash).groupby(prim.weights.index.year).mean().map(
        lambda x: f"{(0.0 if abs(x) < 5e-3 else x):.0%}")
    yearly_w.index.name = "year"
    first, last = R.index[start].date(), R.index[-1].date()
    report = f"""# Phase 5 Report: risk parity with volatility targeting

Out-of-sample: {first} to {last} ({len(R) - start} days), same dates as Phase 3.
Risk model: EWMA covariance forecast for each day from the trailing {W} returns, lambda
re-estimated by multivariate MLE every {oc['lambda_refit_every']} days (median lambda {lam.median():.4f}).
Target for day t = risk-parity weights (cap {ub:.0%}) x min(1, vol target / ex-ante vol); the
remainder is cash earning the 13-week T-bill rate (^IRX, quote at t-1). Long-only, no leverage.
Costs: {cost:.0f} bp per unit of one-way turnover. Sharpe uses returns in excess of the T-bill rate.
Min-variance / min-CVaR / benchmark are fully invested and rebalanced every {every} days.

{fmt.to_markdown()}

`max 63d vol` = highest realised volatility over any 63-day window (how well the target held).

## Primary strategy: average holdings by year
{yearly_w.to_markdown()}
"""
    (out / "phase5_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

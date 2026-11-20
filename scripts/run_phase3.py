"""Phase 3: rolling out-of-sample VaR/ES backtest with Kupiec and Christoffersen tests.

Primary:      window = models.estimation_window (500), refit every 21 days, all models
Sensitivity:  daily GARCH refit at the primary window; windows 250 and 1000 (refit 21)
All set-ups forecast the same dates (from return #oos_start_index) so results are comparable.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mre.backtest.coverage_tests import backtest_summary
from mre.backtest.rolling import BASE_MODELS as ALL_MODELS, es_check, hits, rolling_forecasts
from mre.config import PROJECT_ROOT, load_config
from mre.data.loader import get_prices
from mre.data.quality import align_prices
from mre.portfolio import portfolio_returns, simple_returns

LABEL = {"historical": "Historical", "normal": "Normal", "ewma": "EWMA (MLE lambda)",
         "garch_normal": "GARCH-Normal", "garch_t": "GARCH-t"}


def evaluate(frames, setup, levels, sig_levels):
    rows = []
    for m, f in frames.items():
        for a in levels:
            h = hits(f, a)
            s = backtest_summary(h.to_numpy(), 1 - a)
            e = es_check(f, a)
            row = {"setup": setup, "model": LABEL[m], "alpha": a, **s,
                   "es_ratio": e["ratio"]}
            for sl in sig_levels:
                tag = f"{int(sl * 100)}%"
                row[f"reject_pof@{tag}"] = s["kupiec_p"] < sl
                row[f"reject_cc@{tag}"] = s["cc_p"] < sl
            rows.append(row)
    return rows


def main() -> None:
    cfg = load_config()
    V = cfg.portfolio_value
    prices = align_prices(get_prices(cfg.tickers, cfg.start, cfg.end, cfg.cache_path, cfg.auto_adjust))
    r = portfolio_returns(simple_returns(prices), cfg.weights)
    start = cfg.oos_start_index
    levels = cfg.confidence_levels

    setups = [(f"Primary: W={cfg.estimation_window}, refit {cfg.refit_every}d",
               cfg.estimation_window, cfg.refit_every, ALL_MODELS)]
    setups.append((f"Sens: W={cfg.estimation_window}, refit daily", cfg.estimation_window,
                   cfg.sensitivity_refit_every, ("garch_normal", "garch_t")))
    for w in cfg.sensitivity_windows:
        setups.append((f"Sens: W={w}, refit {cfg.refit_every}d", w, cfg.refit_every, ALL_MODELS))

    rows, primary = [], None
    for name, w, every, models in setups:
        t0 = time.time()
        frames = rolling_forecasts(r, w, start, levels, every, models, cfg.ewma_burn_in)
        print(f"{name}: {time.time() - t0:.0f}s")
        rows += evaluate(frames, name, levels, cfg.significance)
        if primary is None:
            primary = frames

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports"
    res.to_csv(out / "phase3_backtest_results.csv", index=False)
    # primary forecasts for later phases / dashboard
    long = pd.concat({LABEL[m]: f for m, f in primary.items()}, names=["model", "date"])
    long.to_csv(out / "phase3_primary_forecasts.csv.gz", float_format="%.8g")

    def fmt(df):
        d = df.copy()
        d["alpha"] = d["alpha"].map("{:.0%}".format)
        d["violation_rate"] = d["violation_rate"].map("{:.2%}".format)
        d["expected"] = d["expected"].map("{:.1f}".format)
        for c in ("kupiec_p", "ind_p", "cc_p"):
            d[c] = d[c].map(lambda v: f"{v:.3f}" if v >= 0.001 else f"{v:.1e}")
        d["es_ratio"] = d["es_ratio"].map("{:.2f}".format)
        for c in [c for c in d.columns if c.startswith("reject")]:
            d[c] = d[c].map({True: "REJECT", False: "-"})
        keep = ["model", "alpha", "violations", "expected", "violation_rate", "clustered (n11)",
                "kupiec_p", "ind_p", "cc_p"] + [c for c in d.columns if c.startswith("reject")] + ["es_ratio"]
        return d[keep].to_markdown(index=False)

    # violations by calendar year, primary set-up, 99%
    yearly = pd.DataFrame({LABEL[m]: hits(f, 0.99).groupby(f.index.year).sum() for m, f in primary.items()})
    yearly.index.name = "year"
    days = primary["historical"]["realized"].groupby(primary["historical"].index.year).size()
    yearly.insert(0, "days", days)
    yearly.insert(1, "expected@99%", (days * 0.01).round(1))

    g = primary["garch_t"]
    sections = "\n\n".join(f"### {s}\n{fmt(res[res.setup == s])}" for s in res.setup.unique())
    first, last = r.index[start].date(), r.index[-1].date()
    report = f"""# Phase 3 Report: Rolling VaR backtest

Out-of-sample forecast dates: {first} to {last} ({len(r) - start} days), identical for all set-ups.
Each forecast for day t uses only returns before t. Hits: realised loss > VaR.

Tests: Kupiec POF (chi2(1)), Christoffersen independence (chi2(1)) and conditional coverage
(chi2(2)). "REJECT" means the model is rejected at that significance level.
`es_ratio` = mean realised loss on violation days / mean forecast ES on those days
(descriptive; ~1 is good, >1 means ES understated). It is not a formal ES test.

{sections}

## Primary set-up: 99% VaR violations by year
{yearly.to_markdown()}

## Primary GARCH-t parameter path
- nu: median {np.nanmedian(g['nu']):.1f}, range {np.nanmin(g['nu']):.1f} to {np.nanmax(g['nu']):.1f}
- persistence alpha+beta: median {g['persistence'].median():.4f}; share of days with >= 0.999: {(g['persistence'] >= 0.999).mean():.1%}
- primary EWMA lambda: median {primary['ewma']['lambda'].median():.4f}, range {primary['ewma']['lambda'].min():.4f} to {primary['ewma']['lambda'].max():.4f}

## Notes
- Daily-refit sensitivity covers GARCH only (Historical/Normal/EWMA have no refit choice beyond lambda).
- Average 99% VaR of the primary GARCH-t: ${primary['garch_t']['var_99'].mean() * V:,.0f}; Historical: ${primary['historical']['var_99'].mean() * V:,.0f}.
"""
    (out / "phase3_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

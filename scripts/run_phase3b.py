"""Phase 3b: tail-model extensions to the Phase 3 rolling VaR/ES backtest.

New models (see ``mre.backtest.rolling``): GARCH-skew-t, GJR-t, GJR-skew-t, EGARCH-t,
FHS with GARCH-t filter, FHS with GJR-t filter. They are backtested on exactly the
Phase 3 out-of-sample dates together with the Phase 3 models for comparison.

Set-ups (same as Phase 3):
  Primary:      W = models.estimation_window (500), refit every 21 days
  Sensitivity:  daily refit (GARCH family only) at W = 500; W = 250 and 1000 with refit 21
For FHS the residual pool is the last W standardised residuals, so the W = 250 / 1000
set-ups double as the FHS residual-window sensitivity.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from mre.backtest.coverage_tests import backtest_summary
from mre.backtest.rolling import ALL_MODELS, GARCH_SPECS, es_check, hits, rolling_forecasts
from mre.config import PROJECT_ROOT, load_config
from mre.data.loader import get_prices
from mre.data.quality import align_prices
from mre.portfolio import portfolio_returns, simple_returns

LABEL = {"historical": "Historical", "normal": "Normal", "ewma": "EWMA (MLE lambda)",
         "garch_normal": "GARCH-Normal", "garch_t": "GARCH-t",
         "garch_skewt": "GARCH-skew-t", "gjr_t": "GJR-t", "gjr_skewt": "GJR-skew-t",
         "egarch_t": "EGARCH-t", "fhs_garch": "FHS (GARCH-t filter)", "fhs_gjr": "FHS (GJR-t filter)"}
NEW = {"garch_skewt", "gjr_t", "gjr_skewt", "egarch_t", "fhs_garch", "fhs_gjr"}


def evaluate(frames, setup, levels, sig_levels):
    rows = []
    for m, f in frames.items():
        for a in levels:
            s = backtest_summary(hits(f, a).to_numpy(), 1 - a)
            row = {"setup": setup, "model": LABEL[m], "new": m in NEW, "alpha": a, **s,
                   "es_ratio": es_check(f, a)["ratio"],
                   "avg_var_usd": float(f[f"var_{int(round(a * 100))}"].mean())}
            for sl in sig_levels:
                tag = f"{int(sl * 100)}%"
                row[f"reject_pof@{tag}"] = s["kupiec_p"] < sl
                row[f"reject_cc@{tag}"] = s["cc_p"] < sl
            rows.append(row)
    return rows


def fmt(df, V):
    d = df.copy()
    d["alpha"] = d["alpha"].map("{:.0%}".format)
    d["violation_rate"] = d["violation_rate"].map("{:.2%}".format)
    d["expected"] = d["expected"].map("{:.1f}".format)
    for c in ("kupiec_p", "ind_p", "cc_p"):
        d[c] = d[c].map(lambda v: f"{v:.3f}" if v >= 0.001 else f"{v:.1e}")
    d["es_ratio"] = d["es_ratio"].map("{:.2f}".format)
    d["avg VaR $"] = (df["avg_var_usd"] * V).map("{:,.0f}".format)
    d["model"] = [f"**{m}**" if n else m for m, n in zip(df["model"], df["new"])]
    for c in [c for c in d.columns if c.startswith("reject")]:
        d[c] = d[c].map({True: "REJECT", False: "-"})
    keep = ["model", "alpha", "violations", "expected", "violation_rate", "clustered (n11)",
            "kupiec_p", "ind_p", "cc_p"] + [c for c in d.columns if c.startswith("reject")] + \
           ["es_ratio", "avg VaR $"]
    return d[keep].to_markdown(index=False)


def main(skip_daily: bool = False) -> None:
    cfg = load_config()
    V = cfg.portfolio_value
    prices = align_prices(get_prices(cfg.tickers, cfg.start, cfg.end, cfg.cache_path, cfg.auto_adjust))
    r = portfolio_returns(simple_returns(prices), cfg.weights)
    start = cfg.oos_start_index
    levels = cfg.confidence_levels
    W, every = cfg.estimation_window, cfg.refit_every
    garch_family = tuple(m for m in ALL_MODELS if m in GARCH_SPECS)

    setups = [(f"Primary: W={W}, refit {every}d", W, every, ALL_MODELS)]
    if not skip_daily:
        setups.append((f"Sens: W={W}, refit daily", W, cfg.sensitivity_refit_every, garch_family))
    for w in cfg.sensitivity_windows:
        setups.append((f"Sens: W={w}, refit {every}d", w, every, ALL_MODELS))

    rows, primary, failed, timing, by_window = [], None, {}, {}, {}
    for name, w, ev, models in setups:
        t0 = time.time()
        frames = rolling_forecasts(r, w, start, levels, ev, models, cfg.ewma_burn_in)
        timing[name] = time.time() - t0
        print(f"{name}: {timing[name]:.0f}s", flush=True)
        rows += evaluate(frames, name, levels, cfg.significance)
        refit_days = frames[models[-1]].index[::ev] if ev > 1 else frames[models[-1]].index
        failed[name] = {LABEL[m]: int((frames[m].loc[refit_days, "refit_used"] == 0).sum())
                        for m in models if m in GARCH_SPECS}
        if primary is None:
            primary = frames
        if ev == every:
            by_window.setdefault(w, frames)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports"
    res.to_csv(out / "phase3b_backtest_results.csv", index=False)
    long = pd.concat({LABEL[m]: primary[m] for m in NEW if m in primary}, names=["model", "date"])
    long.to_csv(out / "phase3b_primary_forecasts.csv.gz", float_format="%.8g")

    # official model forecasts (config var_model): primary / alternative / previous
    roles = []
    for role in ("primary", "alternative", "previous"):
        spec = cfg.raw["var_model"][role]
        f = by_window[int(spec["window"])][spec["model"]][[f"{k}_{int(a * 100)}" for a in levels for k in ("var", "es")]
                                                           + ["realized", "sigma"]]
        roles.append(f.assign(role=role, model=f"{LABEL[spec['model']]}, W={spec['window']}"))
    off = pd.concat(roles).rename_axis("date").reset_index()
    off.to_csv(out / "var_model_forecasts.csv.gz", index=False, float_format="%.8g")

    yearly = pd.DataFrame({LABEL[m]: hits(f, 0.99).groupby(f.index.year).sum() for m, f in primary.items()
                           if m in NEW or m in ("historical", "garch_t")})
    yearly.index.name = "year"
    days = primary["historical"]["realized"].groupby(primary["historical"].index.year).size()
    yearly.insert(0, "days", days)
    yearly.insert(1, "expected@99%", (days * 0.01).round(1))

    par_rows = []
    for m in ("garch_t", "garch_skewt", "gjr_t", "gjr_skewt", "egarch_t"):
        f = primary[m]
        refit = f.index[::every]
        row = {"model": LABEL[m], "persistence (median)": f.loc[refit, "persistence"].median()}
        if "nu" in f:
            row["nu (median)"] = f.loc[refit, "nu"].median()
        if "lam" in f:
            row["skew lambda (median)"] = f.loc[refit, "lam"].median()
            row["share lambda<0"] = (f.loc[refit, "lam"] < 0).mean()
        if "gamma" in f:
            row["gamma (median)"] = f.loc[refit, "gamma"].median()
        par_rows.append(row)
    params = pd.DataFrame(par_rows).set_index("model")

    sections = "\n\n".join(f"### {s}\n{fmt(res[res.setup == s], V)}" for s in res.setup.unique())
    fail_txt = "\n".join(f"- {s}: " + ", ".join(f"{k} {v}" for k, v in d.items() if v) if any(d.values())
                         else f"- {s}: none" for s, d in failed.items())
    first, last = r.index[start].date(), r.index[-1].date()
    report = f"""# Phase 3b Report: tail-model extensions to the rolling VaR backtest

Out-of-sample forecast dates: {first} to {last} ({len(r) - start} days), identical to Phase 3.
Each forecast for day t uses only returns before t. New models in **bold**.
Tests: Kupiec POF, Christoffersen independence and conditional coverage; "REJECT" at the stated level.
`es_ratio` = mean realised loss on violation days / mean forecast ES on those days (descriptive, ~1 good).
`avg VaR $` = average forecast VaR on the $1,000,000 portfolio.

{sections}

## Primary set-up: 99% VaR violations by year
{yearly.to_markdown()}

## Primary set-up: parameter medians over refits
{params.to_markdown(floatfmt=".3f")}

## Refits rejected as numerically implausible (previous parameters kept)
{fail_txt}

Run time per set-up (s): {", ".join(f"{k}: {v:.0f}" for k, v in timing.items())}

## Notes
- Phase 3 models are recomputed here and reproduce the Phase 3 numbers exactly.
- FHS residual pool = the last W standardised residuals (W = estimation window), filtered with
  the parameters of the latest refit and rolled forward daily; W = 250 / 1000 is the pool-size sensitivity.
- EGARCH is estimated with alpha >= 0 and beta <= 0.999 and from four starting points (best plausible
  likelihood kept): the unrestricted optimum on some 500-day windows had alpha < 0, beta ~ 1 and
  exploded out of sample. A refit with no plausible optimum would keep the previous parameters.
- Six new candidates were evaluated on the same sample; p-values are not adjusted for this.
"""
    (out / "phase3b_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main(skip_daily="--skip-daily" in sys.argv)

"""Phase 3c: conditional EVT (FHS-EVT) and multivariate DCC-GARCH in the rolling VaR backtest.

Same out-of-sample dates (return #1000 onwards), refit every 21 days, coverage tests and
ES ratio as Phase 3 / 3b. Each forecast for day t uses only returns before t.

Set-ups: W = 500 (Phase 3 primary) and W = 1000 (where FHS-GJR passed every test in 3b).
W = 250 is not run: a 90% threshold would leave only 25 exceedances for the GPD.

Models
  benchmarks   GJR-skew-t, FHS-GJR (Phase 3b, recomputed)
  FHS-EVT      GJR-t filter on the portfolio, GPD tail on the residual pool
  DCC-*        GJR-t marginals per asset + DCC(1,1) correlations -> portfolio sigma;
               portfolio tail Normal / FHS / EVT
  CCC-*        ablation: same marginals, constant correlation (a = b = 0)
EVT threshold: 90% quantile of residual losses (primary); 85% and 95% sensitivity.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mre.backtest.coverage_tests import backtest_summary
from mre.backtest.rolling import es_check, hits, rolling_forecasts
from mre.backtest.rolling_dcc import rolling_dcc_forecasts
from mre.backtest.traffic_light import traffic_light
from mre.config import PROJECT_ROOT, load_config
from mre.data.loader import get_prices
from mre.data.quality import align_prices
from mre.portfolio import portfolio_returns, simple_returns

WINDOWS = (500, 1000)
Q_PRIMARY, Q_SENS = 0.90, (0.85, 0.95)
BENCH = ("GJR-skew-t", "FHS-GJR")
LABEL = {"gjr_skewt": "GJR-skew-t", "fhs_gjr": "FHS-GJR", "fhs_evt": "FHS-EVT",
         "dcc_normal": "DCC-Normal", "dcc_fhs": "DCC-FHS", "dcc_evt": "DCC-EVT"}


def evaluate(frames, setup, W, levels, sig_levels, V):
    rows = []
    for name, f in frames.items():
        for a in levels:
            s = backtest_summary(hits(f, a).to_numpy(), 1 - a)
            row = {"setup": setup, "W": W, "model": name, "benchmark": name in BENCH, "alpha": a, **s,
                   "es_ratio": es_check(f, a)["ratio"],
                   "avg_var_usd": float(f[f"var_{int(round(a * 100))}"].mean()) * V}
            for sl in sig_levels:
                tag = f"{int(sl * 100)}%"
                row[f"reject_pof@{tag}"] = s["kupiec_p"] < sl
                row[f"reject_cc@{tag}"] = s["cc_p"] < sl
            rows.append(row)
    return rows


def fmt(df):
    d = df.copy()
    d["alpha"] = d["alpha"].map("{:.0%}".format)
    d["violation_rate"] = d["violation_rate"].map("{:.2%}".format)
    d["expected"] = d["expected"].map("{:.1f}".format)
    for c in ("kupiec_p", "ind_p", "cc_p"):
        d[c] = d[c].map(lambda v: f"{v:.3f}" if v >= 0.001 else f"{v:.1e}")
    d["es_ratio"] = d["es_ratio"].map("{:.2f}".format)
    d["avg VaR $"] = df["avg_var_usd"].map("{:,.0f}".format)
    d["model"] = [m if b else f"**{m}**" for m, b in zip(df["model"], df["benchmark"])]
    for c in [c for c in d.columns if c.startswith("reject")]:
        d[c] = d[c].map({True: "REJECT", False: "-"})
    keep = ["model", "alpha", "violations", "expected", "violation_rate", "clustered (n11)",
            "kupiec_p", "ind_p", "cc_p"] + [c for c in d.columns if c.startswith("reject")] + \
           ["es_ratio", "avg VaR $"]
    return d[keep].to_markdown(index=False)


def run_window(R, r, W, cfg):
    start, levels, every = cfg.oos_start_index, cfg.confidence_levels, cfg.refit_every
    frames, timing = {}, {}
    t0 = time.time()
    uni = rolling_forecasts(r, W, start, levels, every, ("gjr_skewt", "fhs_gjr", "fhs_evt"),
                            evt_threshold=Q_PRIMARY)
    for m, f in uni.items():
        frames[LABEL[m] if m != "fhs_evt" else f"FHS-EVT (u={Q_PRIMARY:.0%})"] = f
    for q in Q_SENS:
        frames[f"FHS-EVT (u={q:.0%})"] = rolling_forecasts(r, W, start, levels, every, ("fhs_evt",),
                                                           evt_threshold=q)["fhs_evt"]
    timing["univariate"] = time.time() - t0
    t0 = time.time()
    for corr in ("dcc", "ccc"):
        d = rolling_dcc_forecasts(R, cfg.weights, W, start, levels, every, evt_threshold=Q_PRIMARY,
                                  correlation=corr)
        for m, f in d.items():
            name = LABEL[m].replace("DCC", corr.upper())
            frames[name + (f" (u={Q_PRIMARY:.0%})" if m == "dcc_evt" else "")] = f
    for q in Q_SENS:
        frames[f"DCC-EVT (u={q:.0%})"] = rolling_dcc_forecasts(
            R, cfg.weights, W, start, levels, every, models=("dcc_evt",), evt_threshold=q)["dcc_evt"]
    timing["multivariate"] = time.time() - t0
    return frames, timing


ORDER = ["GJR-skew-t", "FHS-GJR", "FHS-EVT (u=85%)", "FHS-EVT (u=90%)", "FHS-EVT (u=95%)",
         "CCC-Normal", "CCC-FHS", "CCC-EVT (u=90%)", "DCC-Normal", "DCC-FHS",
         "DCC-EVT (u=85%)", "DCC-EVT (u=90%)", "DCC-EVT (u=95%)"]
EPISODES = {"2010 flash crash / euro crisis (2010)": ("2010-01-01", "2010-12-31"),
            "2011 US downgrade (Aug-Oct 2011)": ("2011-08-01", "2011-10-31"),
            "2018 Q4 / Feb 2018 volmageddon": ("2018-01-01", "2018-12-31"),
            "COVID (Feb-Apr 2020)": ("2020-02-01", "2020-04-30"),
            "2022 stock-bond sell-off": ("2022-01-01", "2022-12-31"),
            "Apr 2025 tariff shock": ("2025-03-01", "2025-05-31")}


def main() -> None:
    cfg = load_config()
    V = cfg.portfolio_value
    R = simple_returns(align_prices(get_prices(cfg.tickers, cfg.start, cfg.end, cfg.cache_path,
                                               cfg.auto_adjust)))
    r = portfolio_returns(R, cfg.weights)
    out = PROJECT_ROOT / "reports"
    rows, all_frames, timing = [], {}, {}
    for W in WINDOWS:
        frames, tm = run_window(R, r, W, cfg)
        frames = {k: frames[k] for k in ORDER}
        all_frames[W], timing[W] = frames, tm
        print(f"W={W}: {tm}", flush=True)
        rows += evaluate(frames, f"W={W}, refit {cfg.refit_every}d", W, cfg.confidence_levels,
                         cfg.significance, V)
    res = pd.DataFrame(rows)
    res.to_csv(out / "phase3c_backtest_results.csv", index=False)
    long = pd.concat({(W, m): f[[c for c in f.columns if c.startswith(("var_", "es_", "sigma", "xi", "corr_",
                                                                       "dcc_", "realized"))]]
                      for W, fr in all_frames.items() for m, f in fr.items()},
                     names=["W", "model", "date"])
    long.to_csv(out / "phase3c_forecasts.csv.gz", float_format="%.8g")

    sections = []
    for s in res.setup.unique():
        sub = res[res.setup == s]
        for a in cfg.confidence_levels:
            sections.append(f"### {s}, {a:.0%} VaR\n{fmt(sub[sub.alpha == a])}")
    sections = "\n\n".join(sections)

    key = ["GJR-skew-t", "FHS-GJR", "FHS-EVT (u=90%)", "CCC-FHS", "DCC-FHS", "DCC-EVT (u=90%)"]
    yearly_txt, episode_txt, overlap_txt = [], [], []
    for W, fr in all_frames.items():
        base = fr["FHS-GJR"]
        days = base["realized"].groupby(base.index.year).size()
        y = pd.DataFrame({m: hits(fr[m], 0.99).groupby(fr[m].index.year).sum() for m in key})
        y.insert(0, "expected@99%", (days * 0.01).round(1))
        y.index.name = "year"
        yearly_txt.append(f"### W={W}\n{y.to_markdown()}")
        ep = pd.DataFrame({m: {e: int(hits(fr[m], 0.99).loc[a:b].sum()) for e, (a, b) in EPISODES.items()}
                           for m in key})
        ep.insert(0, "days", [len(base.loc[a:b]) for a, b in EPISODES.values()])
        episode_txt.append(f"### W={W}\n{ep.to_markdown()}")
        h0, h1 = hits(fr["FHS-GJR"], 0.99).astype(bool), hits(fr["DCC-EVT (u=90%)"], 0.99).astype(bool)
        only0, only1 = base.index[h0 & ~h1], base.index[h1 & ~h0]
        overlap_txt.append(f"- W={W}: both {int((h0 & h1).sum())}; FHS-GJR only {len(only0)} "
                           f"({', '.join(str(d.date()) for d in only0)}); DCC-EVT only {len(only1)} "
                           f"({', '.join(str(d.date()) for d in only1) or 'none'})")

    par_rows = []
    for W, fr in all_frames.items():
        for m in ("FHS-EVT (u=85%)", "FHS-EVT (u=90%)", "FHS-EVT (u=95%)", "DCC-EVT (u=85%)",
                  "DCC-EVT (u=90%)", "DCC-EVT (u=95%)"):
            f = fr[m]
            par_rows.append({"W": W, "model": m, "xi median": f["xi"].median(),
                             "xi 5%": f["xi"].quantile(0.05), "xi 95%": f["xi"].quantile(0.95),
                             "share of days xi at bound": f["xi_at_bound"].mean()})
    xi_tab = pd.DataFrame(par_rows)
    dcc_rows = []
    for W, fr in all_frames.items():
        f = fr["DCC-FHS"]
        refit = f.index[::cfg.refit_every]
        c = f["corr_SPY_TLT"]
        dcc_rows.append({"W": W, "a median": f.loc[refit, "dcc_a"].median(),
                         "b median": f.loc[refit, "dcc_b"].median(),
                         "a+b median": (f.loc[refit, "dcc_a"] + f.loc[refit, "dcc_b"]).median(),
                         "failed refits": int((f.loc[refit, "refit_used"] == 0).sum()),
                         "corr SPY-TLT 2019 mean": c.loc["2019"].mean(),
                         "corr SPY-TLT 2022 mean": c.loc["2022"].mean(),
                         "CCC corr SPY-TLT 2022 mean": fr["CCC-FHS"]["corr_SPY_TLT"].loc["2022"].mean()})
    dcc_tab = pd.DataFrame(dcc_rows)

    tl_in = pd.concat([f.assign(model=f"{m}, W={W}").rename_axis("date").reset_index()
                       for W, fr in all_frames.items() for m, f in fr.items()
                       if m in ("FHS-GJR", "GJR-skew-t", "DCC-FHS", "CCC-EVT (u=90%)", "DCC-EVT (u=90%)")])
    tl = traffic_light(tl_in)
    tl[["green", "yellow", "red"]] = tl[["green", "yellow", "red"]].map("{:.1%}".format)

    first, last = r.index[cfg.oos_start_index].date(), r.index[-1].date()
    n_models = len(ORDER) - len(BENCH)
    report = f"""# Phase 3c Report: conditional EVT and DCC-GARCH

Out-of-sample forecast dates: {first} to {last} ({len(r) - cfg.oos_start_index} days), identical to Phase 3 / 3b.
Refit every {cfg.refit_every} days on the last W returns; filters updated daily with fixed parameters in between.
Each forecast for day t uses only returns before t. New models in **bold**, benchmarks plain.
Tests: Kupiec POF, Christoffersen independence and conditional coverage; "REJECT" at the stated level.
`es_ratio` = mean realised loss on violation days / mean forecast ES on those days (descriptive, ~1 good).
`avg VaR $` = average forecast VaR on the $1,000,000 portfolio.

{sections}

## 99% violations by year
{chr(10).join(yearly_txt)}

## 99% violations in stress episodes
{chr(10).join(episode_txt)}

## 99% violation days: FHS-GJR vs DCC-EVT (u=90%)
{chr(10).join(overlap_txt)}

## Basel traffic light (99% VaR, rolling 250 days; green 0-4, yellow 5-9, red 10+)
{tl.to_markdown()}

## GPD shape xi on the residual pool (daily fits)
{xi_tab.to_markdown(index=False, floatfmt=".3f")}

## DCC parameters (medians over refits) and the SPY-TLT conditional correlation
{dcc_tab.to_markdown(index=False, floatfmt=".3f")}

Run time (s): {", ".join(f"W={W}: " + ", ".join(f"{k} {v:.0f}" for k, v in t.items()) for W, t in timing.items())}

## Notes
- FHS-GJR and GJR-skew-t are recomputed and reproduce Phase 3b exactly.
- GPD: MLE with xi in [-0.5, 0.9], threshold = empirical quantile of the residual losses in the pool
  (W residuals), refitted every day as the pool rolls.
- DCC: GJR-t marginals per asset (refit every {cfg.refit_every} days), correlation targeting
  (Qbar = sample second moment of the window's standardised residuals), (a, b) by two-step Gaussian
  quasi-ML with a + b <= 0.999. A refit where any marginal or the DCC stage fails keeps the previous parameters.
- CCC = the same marginals with a = b = 0; it separates "model each asset's volatility" from
  "let correlations move".
- {n_models} new model variants per window (x2 windows, x2 levels) were evaluated on the same sample.
  p-values are not adjusted for this; with ~{2 * n_models} 99% tests at 5%, one or two passes/rejections by
  chance are expected, so differences of a few violations between close variants are not evidence.
- Decision (2026-11-20): FHS-GJR (W = 1000) stays the official model; DCC-EVT (W = 1000, u = 90%) is a
  monitored challenger with promotion rules fixed in advance (docs/model_validation_report.md, sections 6.6 and 12).
"""
    (out / "phase3c_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

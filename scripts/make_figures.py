"""Static figures for the README and the model validation report (reads reports/*.csv)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mre.config import PROJECT_ROOT  # noqa: E402

REP = PROJECT_ROOT / "reports"
FIG = REP / "figures"

# validated categorical palette (light mode), fixed order
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
ASSET_COLORS = {"SPY": C["blue"], "QQQ": C["orange"], "TLT": C["aqua"], "GLD": C["yellow"]}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def fig_var_backtest():
    f = pd.read_csv(REP / "var_model_forecasts.csv.gz", parse_dates=["date"])
    g = f[f["role"] == "primary"].set_index("date")
    prev = f[f["role"] == "previous"].set_index("date")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(g.index, g["realized"] * 100, color="#b9b8b2", lw=0.6, label="Realised daily return")
    ax.plot(prev.index, -prev["var_99"] * 100, color=C["orange"], lw=0.9, alpha=0.8,
            label=f"Previous: {prev['model'].iloc[0]} 99% VaR")
    ax.plot(g.index, -g["var_99"] * 100, color=C["blue"], lw=1.2, label=f"Primary: {g['model'].iloc[0]} 99% VaR")
    hit = g["realized"] < -g["var_99"]
    ax.scatter(g.index[hit], g.loc[hit, "realized"] * 100, s=16, color=C["red"], zorder=3,
               edgecolor=SURF, linewidth=0.8,
               label=f"Primary violation ({int(hit.sum())} vs {0.01 * len(g):.0f} expected; previous {int((prev['realized'] < -prev['var_99']).sum())})")
    ax.set_ylabel("Daily return (%)")
    ax.set_title("Out-of-sample 99% VaR backtest, FHS with GJR-GARCH-t filter (1,000-day window) vs previous GARCH-t")
    ax.legend(loc="lower left", ncol=2)
    save(fig, "var_backtest_primary.png")


def fig_violations():
    r = pd.read_csv(REP / "phase3b_backtest_results.csv")
    p = r[r["setup"].str.startswith("Primary") & (r["alpha"] == 0.99)].copy()
    w1 = r[(r["setup"] == "Sens: W=1000, refit 21d") & (r["alpha"] == 0.99) & (r["model"] == "FHS (GJR-t filter)")].copy()
    w1["model"] = "FHS (GJR-t filter), W=1000 [primary]"
    p = pd.concat([p, w1], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(p))
    ax.barh(y, p["violations"], color=C["blue"], height=0.55)
    ax.axvline(p["expected"].iloc[0], color=INK, lw=1.2, ls="--")
    ax.text(p["expected"].iloc[0] + 1, -0.55, f"expected {p['expected'].iloc[0]:.1f}", color=INK2, va="bottom")
    for yi, (v, pv) in enumerate(zip(p["violations"], p["kupiec_p"])):
        ax.text(v + 1, yi, f"{v}  (Kupiec p={pv:.1g})", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, p["model"])
    ax.set_xlim(0, p["violations"].max() * 1.45)
    ax.set_ylim(len(p) - 0.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_title("99% VaR violations, 4,030 days (W=500 unless stated)", loc="right")
    save(fig, "violations_99.png")


def fig_risk_contribution():
    rc = pd.read_csv(REP / "phase4_rolling_risk_contribution.csv", index_col=0, parse_dates=True)
    rc = rc.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for col in rc.columns:
        ax.plot(rc.index, rc[col] * 100, color=ASSET_COLORS[col], lw=1.6, label=col)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Share of portfolio variance (%)")
    ax.set_title("Risk contribution by asset (EWMA covariance, monthly average), 40/30/20/10 portfolio")
    ax.legend(ncol=4, loc="upper right")
    ax.set_ylim(-28, 88)
    save(fig, "risk_contribution.png")


def fig_stress():
    s = pd.read_csv(REP / "phase4_historical_scenarios.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(s))
    ax.barh(y - 0.18, s["buy_and_hold"] * 100, height=0.34, color=C["blue"], label="Buy and hold")
    ax.barh(y + 0.18, s["rebalanced"] * 100, height=0.34, color=C["orange"], label="Daily rebalanced")
    for yi, v in zip(y, s["buy_and_hold"]):
        ax.text(v * 100 - 0.6, yi - 0.18, f"{v:.1%}", va="center", ha="right", color=INK2, fontsize=8.5)
    ax.set_yticks(y, s["scenario"])
    ax.invert_yaxis()
    ax.set_xlim(min(s["rebalanced"].min(), s["buy_and_hold"].min()) * 100 * 1.25, 0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Portfolio return over scenario (%)")
    ax.set_title("Historical stress scenarios on the 40/30/20/10 portfolio")
    ax.legend(loc="lower left")
    save(fig, "stress_scenarios.png")


def fig_allocation():
    r = pd.read_csv(REP / "phase5_strategy_returns.csv.gz", index_col=0, parse_dates=True)
    pick = {
        [c for c in r.columns if c.startswith("Benchmark")][0]: ("40/30/20/10 benchmark", C["orange"]),
        [c for c in r.columns if c.startswith("RP (no VT")][0]: ("Risk parity, fully invested", C["aqua"]),
        [c for c in r.columns if "[primary]" in c][0]: ("Risk parity + 10% vol target", C["blue"]),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for col, (label, color) in pick.items():
        w = (1 + r[col]).cumprod()
        axes[0].plot(w.index, w, color=color, lw=1.6, label=label)
        axes[1].plot(w.index, (w / w.cummax() - 1) * 100, color=color, lw=1.2)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[0].set_title("Walk-forward allocation, net of 5 bp costs, 2009-12 to 2025-12")
    axes[0].legend(loc="upper left")
    axes[1].set_ylabel("Drawdown (%)")
    save(fig, "allocation.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_var_backtest()
    fig_violations()
    fig_risk_contribution()
    fig_stress()
    fig_allocation()
    print("figures written to", FIG)

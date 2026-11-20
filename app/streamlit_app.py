"""Market Risk Engine dashboard.

Run:  streamlit run app/streamlit_app.py

Backtest / allocation pages read the result files produced by scripts/run_phase*.py
(so the dashboard shows exactly what the reports show). Risk, stress and attribution
for user-chosen weights are recomputed live from the cached price snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from mre.attribution.euler import parametric_decomposition  # noqa: E402
from mre.config import load_config  # noqa: E402
from mre.data.loader import get_prices  # noqa: E402
from mre.data.quality import align_prices  # noqa: E402
from mre.models.ewma import fit_ewma_cov_lambda  # noqa: E402
from mre.models.garch import fhs_var_es, fit_garch, garch_var_es  # noqa: E402
from mre.models.shrinkage import ledoit_wolf_constant_correlation  # noqa: E402
from mre.portfolio import portfolio_returns, simple_returns  # noqa: E402
from mre.risk.var import historical_es, historical_var, normal_es, normal_var  # noqa: E402
from mre.stress.scenarios import conditional_shock, historical_scenario, hypothetical_shock  # noqa: E402

REP = ROOT / "reports"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#9a9993"

st.set_page_config(page_title="Market Risk Engine", layout="wide")


# ----------------------------------------------------------------- data
@st.cache_data
def load_returns():
    cfg = load_config(ROOT / "config.yaml")
    prices = align_prices(get_prices(cfg.tickers, cfg.start, cfg.end, cfg.cache_path, cfg.auto_adjust))
    return cfg, prices, simple_returns(prices)


@st.cache_data
def read_csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(REP / name, **kw)


@st.cache_data
def ewma_cov(window_returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    fit, nxt = fit_ewma_cov_lambda(window_returns.to_numpy())
    return pd.DataFrame(nxt, index=window_returns.columns, columns=window_returns.columns), fit.lam


@st.cache_data
def garch_t_var(r: np.ndarray, alpha: float) -> tuple[float, float]:
    return garch_var_es(fit_garch(r, "t"), alpha)


@st.cache_data
def primary_var(r: np.ndarray, alpha: float) -> tuple[float, float]:
    """Official model (config var_model.primary): FHS with a GJR-GARCH-t filter."""
    f = fit_garch(r, "t", "gjr")
    return fhs_var_es(f, f.std_resid, alpha)


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def style(fig: go.Figure, height=380, ytitle=None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title=ytitle,
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.5)")
    return fig


cfg, prices, R = load_returns()
T = cfg.tickers
W = cfg.estimation_window
PW = int(cfg.raw["var_model"]["primary"]["window"])

# ----------------------------------------------------------------- sidebar
st.sidebar.title("Portfolio")
V = st.sidebar.number_input("Portfolio value (USD)", 10_000.0, 1e10, cfg.portfolio_value, step=100_000.0)
raw_w = {t: st.sidebar.slider(t, 0.0, 1.0, float(cfg.weights[t]), 0.05) for t in T}
tot = sum(raw_w.values())
if tot == 0:
    st.sidebar.error("At least one weight must be positive.")
    st.stop()
w = pd.Series({t: v / tot for t, v in raw_w.items()})
if abs(tot - 1) > 1e-9:
    st.sidebar.caption(f"Weights sum to {tot:.2f}; normalised to 100%.")
st.sidebar.caption(f"Data: {R.index[0].date()} to {R.index[-1].date()}, {len(R):,} daily returns (Yahoo adjusted close snapshot).")
is_default = all(abs(w[t] - cfg.weights[t]) < 1e-9 for t in T)

st.title("Market Risk Analytics & Model Validation Engine")
st.caption("SPY / QQQ / TLT / GLD. VaR and ES are 1-day, reported as positive losses.")

tab_risk, tab_bt, tab_stress, tab_attr, tab_alloc = st.tabs(
    ["Risk overview", "VaR backtest", "Stress testing", "Risk attribution", "Allocation"])

# ----------------------------------------------------------------- risk overview
with tab_risk:
    rp = portfolio_returns(R, w)
    win = rp.iloc[-W:].to_numpy()
    alpha = st.radio("Confidence level", [0.95, 0.99], index=1, horizontal=True, format_func=lambda a: f"{a:.0%}")
    pv, pe = primary_var(rp.iloc[-PW:].to_numpy(), alpha)
    gv, ge = garch_t_var(win, alpha)
    rows = [
        (f"FHS-GJR [primary], next day (last {PW})", pv, pe),
        (f"GARCH(1,1)-t [previous], next day (last {W})", gv, ge),
        ("Historical, full sample", historical_var(rp, alpha), historical_es(rp, alpha)),
        (f"Historical, last {W} days", historical_var(win, alpha), historical_es(win, alpha)),
        (f"Normal, last {W} days", normal_var(win, alpha), normal_es(win, alpha)),
    ]
    c = st.columns(5)
    for col, (name, v, e) in zip(c, rows):
        col.metric(f"VaR · {name}", money(v * V))
        col.caption(f"ES {money(e * V)}")
    st.caption(f"As of {R.index[-1].date()}. Full-sample numbers are in-sample descriptions; "
               "the FHS-GJR and GARCH-t numbers are one-day-ahead forecasts. The primary model was changed "
               "from GARCH-t to FHS-GJR in Phase 3b (see the VaR backtest tab).")
    fig = go.Figure()
    fig.add_histogram(x=rp * 100, nbinsx=150, marker_color=SERIES[0], name="Daily return",
                      hovertemplate="%{x:.2f}%: %{y} days<extra></extra>")
    fig.add_vline(x=-historical_var(rp, alpha) * 100, line_dash="dash", line_color=SERIES[7],
                  annotation_text=f"Historical {alpha:.0%} VaR", annotation_position="top left")
    st.plotly_chart(style(fig, 320, "Days").update_layout(hovermode="closest", showlegend=False,
                                                          xaxis_title="Daily portfolio return (%)"),
                    use_container_width=True)

# ----------------------------------------------------------------- backtest
with tab_bt:
    res = read_csv("phase3b_backtest_results.csv")
    off = read_csv("var_model_forecasts.csv.gz", parse_dates=["date"])
    off["model"] = off["model"] + " [" + off["role"] + "]"
    others = pd.concat([read_csv("phase3_primary_forecasts.csv.gz", parse_dates=["date"]),
                        read_csv("phase3b_primary_forecasts.csv.gz", parse_dates=["date"])])
    others["model"] = others["model"] + ", W=500"
    fc = pd.concat([off, others], ignore_index=True)
    st.caption("Out-of-sample rolling forecasts for the 40/30/20/10 portfolio (Phases 3 and 3b), "
               "2009-12-23 to 2025-12-31. Each forecast uses only returns before the forecast date. "
               "Primary model: FHS with a GJR-GARCH-t filter on a 1,000-day window (the only model that "
               "passes Kupiec, independence and conditional coverage at both 95% and 99%).")
    c1, c2 = st.columns(2)
    names = fc["model"].unique().tolist()
    model = c1.selectbox("Model", names, index=0)
    a_bt = c2.radio("Level", [0.95, 0.99], index=1, horizontal=True, format_func=lambda a: f"{a:.0%}", key="bt_a")
    k = int(round(a_bt * 100))
    g = fc[fc["model"] == model].set_index("date")
    hit = g["realized"] < -g[f"var_{k}"]
    fig = go.Figure()
    fig.add_scatter(x=g.index, y=g["realized"] * 100, mode="lines", line=dict(color=MUTED, width=0.7), name="Realised return")
    fig.add_scatter(x=g.index, y=-g[f"var_{k}"] * 100, mode="lines", line=dict(color=SERIES[0], width=1.5), name=f"-VaR {a_bt:.0%}")
    fig.add_scatter(x=g.index[hit], y=g.loc[hit, "realized"] * 100, mode="markers",
                    marker=dict(color=SERIES[7], size=7, line=dict(color="white", width=1)),
                    name=f"Violation ({int(hit.sum())}, expected {len(g) * (1 - a_bt):.0f})")
    st.plotly_chart(style(fig, 420, "Daily return (%)"), use_container_width=True)
    setup = st.selectbox("Set-up", res["setup"].unique().tolist())
    tbl = res[(res["setup"] == setup)][["model", "alpha", "violations", "expected", "violation_rate",
                                        "kupiec_p", "ind_p", "cc_p", "es_ratio"]]
    st.dataframe(tbl.style.format({"alpha": "{:.0%}", "expected": "{:.1f}", "violation_rate": "{:.2%}",
                                   "kupiec_p": "{:.2g}", "ind_p": "{:.2g}", "cc_p": "{:.2g}", "es_ratio": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("p-values below 0.05 reject correct coverage (Kupiec), independence (Christoffersen) "
               "or both (conditional coverage). ES ratio = mean loss on violation days / mean forecast ES.")

# ----------------------------------------------------------------- stress
with tab_stress:
    st.subheader("Historical scenarios")
    rows = []
    for sc in cfg.raw["stress"]["historical"]:
        bh = historical_scenario(prices, w, sc["start"], sc["end"], V, sc["name"], "buy_and_hold")
        rb = historical_scenario(prices, w, sc["start"], sc["end"], V, sc["name"], "rebalanced")
        rows.append({"Scenario": sc["name"], "Period": f"{bh.start.date()} to {bh.end.date()}",
                     "Buy & hold": bh.portfolio_return, "P&L": bh.pnl, "Rebalanced": rb.portfolio_return,
                     "Max drawdown": bh.max_drawdown})
    st.dataframe(pd.DataFrame(rows).style.format({"Buy & hold": "{:.1%}", "Rebalanced": "{:.1%}",
                                                  "Max drawdown": "{:.1%}", "P&L": money}),
                 hide_index=True, use_container_width=True)

    st.subheader("Hypothetical instantaneous shock")
    cols = st.columns(len(T))
    shocks = {t: col.slider(f"{t} shock (%)", -50, 50, d, 1, key=f"sh_{t}") / 100
              for col, t, d in zip(cols, T, [-20, -20, 5, 3])}
    hs = hypothetical_shock(w, shocks, V)
    st.metric("Portfolio P&L", money(hs.pnl), f"{hs.portfolio_return:+.2%}", delta_color="normal")

    st.subheader("Conditional shock")
    cov_now, lam_now = ewma_cov(R.iloc[-W:])
    lead = st.selectbox("Shocked asset", T, index=0)
    size = st.slider(f"{lead} move (%)", -30, 30, -10, 1) / 100
    cs = conditional_shock(cov_now, w, {lead: size}, V)
    st.dataframe(pd.DataFrame({"Implied move": pd.Series(cs.asset_returns), "P&L": pd.Series(cs.contributions)})
                 .style.format({"Implied move": "{:+.2%}", "P&L": money}), use_container_width=True)
    st.caption(f"Other assets move by E[r | {lead}] under the current EWMA covariance "
               f"(lambda {lam_now:.3f}, last {W} days). Portfolio P&L {money(cs.pnl)}.")

# ----------------------------------------------------------------- attribution
with tab_attr:
    c1, c2, c3 = st.columns(3)
    cov_name = c1.selectbox("Covariance", ["EWMA (MLE lambda)", f"Sample ({W}d)", f"Ledoit-Wolf const-corr ({W}d)"])
    a_at = c2.radio("Level", [0.95, 0.99], index=1, horizontal=True, format_func=lambda a: f"{a:.0%}", key="at_a")
    measure = c3.radio("Measure", ["var", "es"], horizontal=True, format_func=str.upper)
    win_a = R.iloc[-W:]
    cov = {"EWMA (MLE lambda)": cov_now, f"Sample ({W}d)": win_a.cov(),
           f"Ledoit-Wolf const-corr ({W}d)": pd.DataFrame(
               ledoit_wolf_constant_correlation(win_a.to_numpy()).covariance, index=T, columns=T)}[cov_name]
    d = parametric_decomposition(cov, w, a_at, V, measure=measure)
    st.metric(f"Portfolio {measure.upper()} {a_at:.0%}", money(d.attrs["total"]))
    st.caption(f"Diversification benefit vs sum of standalone: {money(d.attrs['diversification_benefit'])}")
    fig = go.Figure()
    fig.add_bar(x=T, y=d["weight"] * 100, name="Weight", marker_color=SERIES[0])
    fig.add_bar(x=T, y=d["pct_contribution"] * 100, name="Share of risk", marker_color=SERIES[1])
    st.plotly_chart(style(fig, 320, "%").update_layout(hovermode="x", barmode="group"), use_container_width=True)
    st.dataframe(d[["weight", "standalone", "marginal", "component", "pct_contribution", "incremental"]]
                 .style.format({"weight": "{:.1%}", "pct_contribution": "{:.1%}", "standalone": "${:,.0f}",
                                "marginal": "${:,.0f}", "component": "${:,.0f}", "incremental": "${:,.0f}"}),
                 use_container_width=True)
    st.caption("Euler decomposition of Gaussian VaR/ES (zero mean); components sum to the total. "
               "Marginal = change per +100% weight; incremental = reduction from closing the position.")
    if is_default:
        rc = read_csv("phase4_rolling_risk_contribution.csv", index_col=0, parse_dates=True).resample("ME").mean()
        fig = go.Figure()
        for i, t in enumerate(T):
            fig.add_scatter(x=rc.index, y=rc[t] * 100, mode="lines", name=t, line=dict(color=SERIES[i], width=1.6))
        st.plotly_chart(style(fig, 340, "Share of variance (%)"), use_container_width=True)
        st.caption("Risk contribution over time for the default weights (EWMA, fixed lambda 0.94).")

# ----------------------------------------------------------------- allocation
with tab_alloc:
    perf = read_csv("phase5_results.csv", index_col=0)
    rets = read_csv("phase5_strategy_returns.csv.gz", index_col=0, parse_dates=True)
    path = read_csv("phase5_primary_path.csv.gz", header=[0, 1], index_col=0, parse_dates=True)
    st.caption("Walk-forward results (Phase 5): targets for day t use data up to t-1; 5 bp costs; "
               "cash earns the 13-week T-bill rate.")
    default = [c for c in rets.columns if "[primary]" in c or c.startswith("RP (no VT") or c.startswith("Benchmark")]
    pick = st.multiselect("Strategies", rets.columns.tolist(), default=default)
    if pick:
        fig = go.Figure()
        for i, col in enumerate(pick[:8]):
            wealth = (1 + rets[col]).cumprod()
            fig.add_scatter(x=wealth.index, y=wealth, mode="lines", name=col, line=dict(color=SERIES[i], width=1.6))
        st.plotly_chart(style(fig, 380, "Growth of $1").update_yaxes(type="log"), use_container_width=True)
        if len(pick) > 8:
            st.caption("Only the first 8 strategies are plotted.")
    pct = ["CAGR", "vol", "max DD", "VaR99", "ES99", "max 63d vol", "2020", "2022", "avg cash", "turnover/yr"]
    st.dataframe(perf.style.format({**{c: "{:.1%}" for c in pct}, "Sharpe (excess)": "{:.2f}",
                                    "Calmar": "{:.2f}", "trades/yr": "{:.1f}"}), use_container_width=True)
    hold = path["weights"].copy()
    hold["cash"] = 1 - hold.sum(axis=1)  # risky weights + cash = 1 by construction
    hold = hold.resample("ME").mean()
    fig = go.Figure()
    for i, col in enumerate(hold.columns):
        fig.add_scatter(x=hold.index, y=hold[col] * 100, mode="lines", stackgroup="w", name=col,
                        line=dict(width=0.5, color=SERIES[i] if col != "cash" else MUTED))
    st.plotly_chart(style(fig, 320, "Holdings (%)"), use_container_width=True)
    st.caption("Primary strategy (risk parity + 10% vol target, 5pp threshold): average monthly holdings.")

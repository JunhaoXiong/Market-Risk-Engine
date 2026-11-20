"""Phase 6: extra validation statistics for the model validation report.

Reads the out-of-sample forecasts already produced by Phases 3 and 3b (no re-estimation):
the official VaR models from config ``var_model`` (primary FHS-GJR W=1000, alternative
GJR-skew-t W=500, previous GARCH-t W=500) plus the other Phase 3 models, and computes the
Basel traffic-light zones for the 99% VaR on rolling 250-day windows.

Basel zones for 250 observations at 99%: green 0-4 exceptions, yellow 5-9, red 10+.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mre.backtest.traffic_light import traffic_light

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    off = pd.read_csv(REPORTS / "var_model_forecasts.csv.gz", parse_dates=["date"])
    off["model"] = off["model"] + " [" + off["role"] + "]"
    p3 = pd.read_csv(REPORTS / "phase3_primary_forecasts.csv.gz", parse_dates=["date"])
    p3 = p3[p3["model"] != "GARCH-t"]  # already included as [previous]
    p3["model"] = p3["model"] + ", W=500"
    fc = pd.concat([off, p3], ignore_index=True)
    tl = traffic_light(fc)
    tl.to_csv(REPORTS / "phase6_traffic_light.csv")
    fmt = tl.copy()
    for c in ["green", "yellow", "red"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.1%}")
    last = fc["date"].max().date()
    lines = [
        "# Phase 6: Basel traffic light (99% VaR, rolling 250 days)",
        "",
        "Source: out-of-sample forecasts 2009-12-23 to 2025-12-31, refit every 21 days "
        "(official models from config var_model, other Phase 3 models at W=500).",
        "Zones per 250-day window: green 0-4 exceptions, yellow 5-9, red 10+.",
        f"Shares are over all {int(tl['windows'].iloc[0])} overlapping windows; last window ends {last}.",
        "",
        fmt.to_markdown(),
        "",
    ]
    (REPORTS / "phase6_traffic_light.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

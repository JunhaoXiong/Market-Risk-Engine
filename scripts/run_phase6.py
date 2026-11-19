"""Phase 6: extra validation statistics for the model validation report.

Reads the Phase 3 primary out-of-sample forecasts (no re-estimation) and computes the
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
    fc = pd.read_csv(REPORTS / "phase3_primary_forecasts.csv.gz", parse_dates=["date"])
    tl = traffic_light(fc)
    tl.to_csv(REPORTS / "phase6_traffic_light.csv")
    fmt = tl.copy()
    for c in ["green", "yellow", "red"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.1%}")
    last = fc["date"].max().date()
    lines = [
        "# Phase 6: Basel traffic light (99% VaR, rolling 250 days)",
        "",
        "Source: Phase 3 primary set-up (W=500, refit 21d) out-of-sample forecasts.",
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

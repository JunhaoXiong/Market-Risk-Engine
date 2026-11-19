"""Basel traffic-light test for 99% VaR exceptions (BCBS 1996 backtesting framework)."""
from __future__ import annotations

import pandas as pd


def basel_zone(n: int) -> str:
    if n <= 4:
        return "green"
    if n <= 9:
        return "yellow"
    return "red"


def traffic_light(fc: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    rows = []
    for model, g in fc.groupby("model", sort=False):
        g = g.sort_values("date")
        hits = (-g["realized"] > g["var_99"]).astype(int)
        counts = hits.rolling(window).sum().dropna().astype(int)
        zones = counts.map(basel_zone)
        share = zones.value_counts(normalize=True)
        rows.append({
            "model": model,
            "windows": len(counts),
            "green": share.get("green", 0.0),
            "yellow": share.get("yellow", 0.0),
            "red": share.get("red", 0.0),
            "max exceptions": int(counts.max()),
            "last 250d exceptions": int(counts.iloc[-1]),
            "last zone": zones.iloc[-1],
        })
    return pd.DataFrame(rows).set_index("model")

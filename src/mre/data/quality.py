"""Data quality checks for a wide (dates x tickers) price panel.

Philosophy: checks *report* problems; they never silently change data. The only
transformation offered is ``align_prices``, which keeps dates on which every
asset has a valid price (an inner join). We deliberately do not forward-fill:
filling a missing price creates an artificial zero return followed by a catch-up
jump, which distorts the tails that VaR is trying to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"


@dataclass
class Issue:
    severity: str
    check: str
    detail: str


@dataclass
class QualityReport:
    issues: list[Issue] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when there are no ERROR-level issues."""
        return not any(i.severity == ERROR for i in self.issues)

    def add(self, severity: str, check: str, detail: str) -> None:
        self.issues.append(Issue(severity, check, detail))

    def by_check(self, check: str) -> list[Issue]:
        return [i for i in self.issues if i.check == check]

    def to_markdown(self) -> str:
        lines = [f"**Status:** {'PASSED' if self.passed else 'FAILED'}", ""]
        for k, v in self.summary.items():
            lines.append(f"- {k}: {v}")
        lines += ["", "| Severity | Check | Detail |", "|---|---|---|"]
        lines += [f"| {i.severity} | {i.check} | {i.detail} |" for i in self.issues] or [
            "| - | - | no issues |"
        ]
        return "\n".join(lines)


def _longest_constant_runs(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Return (start, end, length) for each run of identical consecutive values."""
    s = series.dropna()
    if s.empty:
        return []
    run_id = (s != s.shift()).cumsum()
    runs = []
    for _, grp in s.groupby(run_id):
        runs.append((grp.index[0], grp.index[-1], len(grp)))
    return runs


def check_prices(
    prices: pd.DataFrame,
    stale_run_days: int = 5,
    extreme_return_abs: float = 0.15,
    max_calendar_gap_days: int = 5,
) -> QualityReport:
    rep = QualityReport()

    if prices.empty:
        rep.add(ERROR, "empty", "price frame is empty")
        return rep

    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex):
        rep.add(ERROR, "index_type", f"index must be DatetimeIndex, got {type(idx).__name__}")
        return rep

    n_dup = int(idx.duplicated().sum())
    if n_dup:
        rep.add(ERROR, "duplicate_dates", f"{n_dup} duplicated dates")
    if not idx.is_monotonic_increasing:
        rep.add(ERROR, "unsorted_index", "dates are not sorted ascending")

    non_numeric = [c for c in prices.columns if not pd.api.types.is_numeric_dtype(prices[c])]
    if non_numeric:
        rep.add(ERROR, "non_numeric", f"non-numeric columns: {non_numeric}")
        return rep

    for col in prices.columns:
        s = prices[col]
        first_valid = s.first_valid_index()
        if first_valid is None:
            rep.add(ERROR, "missing_values", f"{col}: no valid prices")
            continue
        # Missing values after the asset's first valid price are genuine gaps.
        n_missing = int(s.loc[first_valid:].isna().sum())
        n_leading = int(s.loc[:first_valid].isna().sum())
        if n_leading:
            rep.add(WARNING, "late_start", f"{col}: first valid price on {first_valid.date()}")
        if n_missing:
            rep.add(WARNING, "missing_values", f"{col}: {n_missing} missing prices after first valid date")

        n_nonpos = int((s <= 0).sum())
        if n_nonpos:
            rep.add(ERROR, "non_positive_price", f"{col}: {n_nonpos} prices <= 0")

        for start, end, length in _longest_constant_runs(s):
            if length >= stale_run_days:
                rep.add(
                    WARNING,
                    "stale_prices",
                    f"{col}: {length} identical prices {start.date()}..{end.date()}",
                )

        rets = s.dropna().pct_change().dropna()
        rets = rets[np.isfinite(rets)]
        extreme = rets[rets.abs() > extreme_return_abs]
        for dt, r in extreme.items():
            rep.add(WARNING, "extreme_return", f"{col}: {r:+.2%} on {dt.date()} (kept, not removed)")

    gaps = pd.Series(idx[1:] - idx[:-1], index=idx[1:]).dt.days
    for dt, g in gaps[gaps > max_calendar_gap_days].items():
        rep.add(WARNING, "calendar_gap", f"{g} calendar days without data before {dt.date()}")

    all_valid = prices.notna().all(axis=1)
    rep.summary = {
        "rows": len(prices),
        "first_date": idx.min().date().isoformat(),
        "last_date": idx.max().date().isoformat(),
        "tickers": list(prices.columns),
        "rows_with_all_assets": int(all_valid.sum()),
        "rows_dropped_by_alignment": int((~all_valid).sum()),
    }
    return rep


def align_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates where every asset has a valid, positive price."""
    clean = prices.sort_index()
    clean = clean[~clean.index.duplicated(keep="first")]
    clean = clean.where(clean > 0)
    return clean.dropna(how="any")

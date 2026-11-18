"""Load and validate the project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    portfolio_value: float
    weights: dict[str, float]
    start: str
    end: str
    auto_adjust: bool
    cache_path: Path
    stale_run_days: int
    extreme_return_abs: float
    max_calendar_gap_days: int
    confidence_levels: tuple[float, ...]
    horizon_days: int
    quantile_method: str
    normal_include_mean: bool
    estimation_window: int = 1000
    ewma_burn_in: int = 30
    garch_primary_dist: str = "t"
    garch_benchmark_dist: str = "normal"
    oos_start_index: int = 1000
    refit_every: int = 21
    sensitivity_windows: tuple[int, ...] = (250, 1000)
    sensitivity_refit_every: int = 1
    significance: tuple[float, ...] = (0.05, 0.01)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def tickers(self) -> list[str]:
        return list(self.weights)


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    weights = {str(k): float(v) for k, v in raw["portfolio"]["weights"].items()}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Portfolio weights must sum to 1, got {sum(weights.values()):.6f}")

    levels = tuple(float(c) for c in raw["risk"]["confidence_levels"])
    if not all(0.5 < c < 1.0 for c in levels):
        raise ValueError(f"Confidence levels must be in (0.5, 1): {levels}")

    cache = Path(raw["data"]["cache_path"])
    if not cache.is_absolute():
        cache = path.parent / cache

    return Config(
        portfolio_value=float(raw["portfolio"]["value"]),
        weights=weights,
        start=str(raw["data"]["start"]),
        end=str(raw["data"]["end"]),
        auto_adjust=bool(raw["data"]["auto_adjust"]),
        cache_path=cache,
        stale_run_days=int(raw["quality"]["stale_run_days"]),
        extreme_return_abs=float(raw["quality"]["extreme_return_abs"]),
        max_calendar_gap_days=int(raw["quality"]["max_calendar_gap_days"]),
        confidence_levels=levels,
        horizon_days=int(raw["risk"]["horizon_days"]),
        quantile_method=str(raw["risk"]["quantile_method"]),
        normal_include_mean=bool(raw["risk"]["normal_include_mean"]),
        estimation_window=int(raw.get("models", {}).get("estimation_window", 1000)),
        ewma_burn_in=int(raw.get("models", {}).get("ewma_burn_in", 30)),
        garch_primary_dist=str(raw.get("models", {}).get("garch_primary_dist", "t")),
        garch_benchmark_dist=str(raw.get("models", {}).get("garch_benchmark_dist", "normal")),
        oos_start_index=int(raw.get("backtest", {}).get("oos_start_index", 1000)),
        refit_every=int(raw.get("backtest", {}).get("refit_every", 21)),
        sensitivity_windows=tuple(int(x) for x in raw.get("backtest", {}).get("sensitivity_windows", [250, 1000])),
        sensitivity_refit_every=int(raw.get("backtest", {}).get("sensitivity_refit_every", 1)),
        significance=tuple(float(x) for x in raw.get("backtest", {}).get("significance", [0.05, 0.01])),
        raw=raw,
    )

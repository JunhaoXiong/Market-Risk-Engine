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
        raw=raw,
    )

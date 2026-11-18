import pytest

from mre.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.tickers == ["SPY", "QQQ", "TLT", "GLD"]
    assert cfg.weights == {"SPY": 0.4, "QQQ": 0.3, "TLT": 0.2, "GLD": 0.1}
    assert cfg.portfolio_value == 1_000_000
    assert cfg.confidence_levels == (0.95, 0.99)
    assert cfg.cache_path.is_absolute()
    assert cfg.estimation_window == 1000
    assert cfg.garch_primary_dist == "t"


def test_bad_weights_rejected(tmp_path):
    text = (
        "portfolio: {value: 1, weights: {A: 0.5, B: 0.6}}\n"
        "data: {start: '2020-01-01', end: '2020-12-31', auto_adjust: true, cache_path: p.csv}\n"
        "quality: {stale_run_days: 5, extreme_return_abs: 0.15, max_calendar_gap_days: 5}\n"
        "risk: {confidence_levels: [0.99], horizon_days: 1, quantile_method: linear, normal_include_mean: true}\n"
    )
    p = tmp_path / "c.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(p)

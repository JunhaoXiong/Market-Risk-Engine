"""Smoke test: the dashboard runs end-to-end without exceptions (headless)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


@pytest.mark.skipif(not (REPORTS / "phase5_strategy_returns.csv.gz").exists(),
                    reason="run scripts/run_phase*.py first")
def test_dashboard_renders():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    assert len(at.metric) >= 4


@pytest.mark.skipif(not (REPORTS / "phase5_strategy_returns.csv.gz").exists(),
                    reason="run scripts/run_phase*.py first")
def test_changing_weights_recomputes():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    before = at.metric[0].value
    at.sidebar.slider[0].set_value(1.0).run()   # SPY weight -> 100% before normalisation
    assert not at.exception, at.exception
    assert at.metric[0].value != before

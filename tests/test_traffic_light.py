import numpy as np
import pandas as pd

from mre.backtest.traffic_light import basel_zone, traffic_light


def test_zone_boundaries():
    assert [basel_zone(n) for n in (0, 4, 5, 9, 10, 30)] == [
        "green", "green", "yellow", "yellow", "red", "red"]


def test_traffic_light_counts_exceptions_in_last_window():
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    realized = np.zeros(n)
    realized[-6:] = -0.05          # 6 exceptions in the last 250 days
    fc = pd.DataFrame({"model": "m", "date": dates, "var_99": 0.02, "realized": realized})
    tl = traffic_light(fc)
    assert tl.loc["m", "windows"] == n - 249
    assert tl.loc["m", "last 250d exceptions"] == 6
    assert tl.loc["m", "last zone"] == "yellow"
    assert tl.loc["m", "max exceptions"] == 6


def test_loss_equal_to_var_is_not_an_exception():
    dates = pd.bdate_range("2020-01-01", periods=250)
    fc = pd.DataFrame({"model": "m", "date": dates, "var_99": 0.02, "realized": -0.02})
    assert traffic_light(fc).loc["m", "last 250d exceptions"] == 0

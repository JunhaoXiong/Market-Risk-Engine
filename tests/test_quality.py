import numpy as np
import pandas as pd

from mre.data.quality import align_prices, check_prices


def test_clean_data_passes(clean_prices):
    rep = check_prices(clean_prices)
    assert rep.passed
    assert rep.issues == []
    assert rep.summary["rows_dropped_by_alignment"] == 0


def test_duplicate_dates_are_errors(clean_prices):
    dup = pd.concat([clean_prices, clean_prices.iloc[[10]]]).sort_index()
    rep = check_prices(dup)
    assert not rep.passed
    assert rep.by_check("duplicate_dates")


def test_non_positive_price_is_error(clean_prices):
    bad = clean_prices.copy()
    bad.iloc[50, 1] = 0.0
    rep = check_prices(bad)
    assert not rep.passed
    assert "QQQ" in rep.by_check("non_positive_price")[0].detail


def test_missing_and_late_start_flagged(clean_prices):
    bad = clean_prices.copy()
    bad.iloc[:5, 3] = np.nan     # GLD starts late
    bad.iloc[100, 0] = np.nan    # SPY gap
    rep = check_prices(bad)
    assert rep.passed  # warnings only
    assert rep.by_check("late_start")
    assert "SPY: 1 missing" in rep.by_check("missing_values")[0].detail
    assert rep.summary["rows_dropped_by_alignment"] == 6


def test_stale_prices_flagged(clean_prices):
    bad = clean_prices.copy()
    bad.iloc[60:67, 2] = bad.iloc[60, 2]  # 7 identical prices
    rep = check_prices(bad, stale_run_days=5)
    stale = rep.by_check("stale_prices")
    assert len(stale) == 1 and "TLT: 7 identical" in stale[0].detail


def test_extreme_return_flagged_not_removed(clean_prices):
    bad = clean_prices.copy()
    bad.iloc[120:, 0] *= 0.7  # -30% jump
    rep = check_prices(bad, extreme_return_abs=0.15)
    assert len(rep.by_check("extreme_return")) == 1
    assert len(align_prices(bad)) == len(bad)  # extreme day is kept


def test_calendar_gap_flagged(clean_prices):
    bad = clean_prices.drop(clean_prices.index[30:40])
    rep = check_prices(bad, max_calendar_gap_days=5)
    assert rep.by_check("calendar_gap")


def test_checks_do_not_mutate_input(clean_prices):
    before = clean_prices.copy()
    check_prices(clean_prices)
    pd.testing.assert_frame_equal(before, clean_prices)


def test_align_inner_joins_and_drops_non_positive(clean_prices):
    bad = clean_prices.copy()
    bad.iloc[3, 0] = np.nan
    bad.iloc[7, 2] = -1.0
    out = align_prices(bad)
    assert len(out) == len(bad) - 2
    assert out.notna().all().all() and (out > 0).all().all()


def test_empty_frame_fails():
    rep = check_prices(pd.DataFrame())
    assert not rep.passed

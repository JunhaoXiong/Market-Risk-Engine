import sys
import types

import pandas as pd
import pytest

from mre.data import loader


def test_save_load_roundtrip(tmp_path, clean_prices):
    path = tmp_path / "prices.csv"
    meta = loader.save_prices(clean_prices, path, {"source": "test"})
    assert meta["rows"] == len(clean_prices)
    assert loader.metadata_path(path).exists()
    back = loader.load_prices(path)
    pd.testing.assert_frame_equal(back, clean_prices, check_freq=False, atol=1e-6)


def test_hash_mismatch_detected(tmp_path, clean_prices):
    path = tmp_path / "prices.csv"
    loader.save_prices(clean_prices, path, {"source": "test"})
    with open(path, "a") as fh:
        fh.write("2030-01-01,1,1,1,1\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        loader.load_prices(path)


def test_get_prices_uses_cache_without_network(tmp_path, clean_prices, monkeypatch):
    path = tmp_path / "prices.csv"
    loader.save_prices(clean_prices, path, {"source": "test"})

    def boom(*a, **k):
        raise AssertionError("network should not be touched when cache exists")

    monkeypatch.setattr(loader, "download_prices", boom)
    out = loader.get_prices(["GLD", "SPY"], "2020-01-01", "2021-01-01", path)
    assert list(out.columns) == ["GLD", "SPY"]


def test_download_end_is_inclusive_and_parses_multiindex(monkeypatch):
    calls = {}
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "TLT"]], names=["Price", "Ticker"])
    fake = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]], index=idx, columns=cols)

    def fake_download(tickers, start, end, **kwargs):
        calls.update(tickers=tickers, start=start, end=end, **kwargs)
        return fake

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    out = loader.download_prices(["TLT", "SPY"], "2024-01-01", "2024-01-03")
    assert calls["end"] == "2024-01-04"  # yfinance end is exclusive
    assert calls["auto_adjust"] is True
    assert list(out.columns) == ["TLT", "SPY"]
    assert out.loc["2024-01-03", "TLT"] == 2.1


def test_download_empty_raises(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
    )
    with pytest.raises(RuntimeError, match="No data"):
        loader.download_prices(["SPY"], "2024-01-01", "2024-01-03")

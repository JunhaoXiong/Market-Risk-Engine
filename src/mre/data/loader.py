"""Price download and local caching.

Design notes
------------
* Yahoo re-computes adjusted prices retroactively whenever a new dividend is paid,
  so the "same" download on two different days can differ slightly. We therefore
  snapshot the download to CSV and record metadata (download time, library version,
  SHA-256 of the file). All analysis reads the snapshot, never the live API.
* yfinance treats ``end`` as exclusive; the config's ``end`` is inclusive, so we add
  one calendar day when calling the API.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def download_prices(
    tickers: list[str], start: str, end_inclusive: str, auto_adjust: bool = True
) -> pd.DataFrame:
    """Download daily close prices (adjusted if ``auto_adjust``) from Yahoo Finance."""
    import yfinance as yf  # imported lazily so tests never need network access

    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {tickers}; check network access to Yahoo Finance.")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(prices, pd.DataFrame):
        prices = prices.to_frame(tickers[0])
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"Tickers missing from download: {missing}")

    prices = prices[tickers].copy()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices.index.name = "Date"
    prices.columns.name = None
    return prices


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def save_prices(prices: pd.DataFrame, csv_path: str | Path, source_info: dict) -> dict:
    """Write prices to CSV plus a JSON metadata sidecar. Returns the metadata."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(csv_path, float_format="%.6f")
    meta = {
        **source_info,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(prices)),
        "columns": list(prices.columns),
        "first_date": prices.index.min().strftime("%Y-%m-%d"),
        "last_date": prices.index.max().strftime("%Y-%m-%d"),
        "sha256": _sha256(csv_path),
    }
    metadata_path(csv_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_prices(csv_path: str | Path, verify_hash: bool = True) -> pd.DataFrame:
    """Load cached prices; optionally verify the file matches its recorded SHA-256."""
    csv_path = Path(csv_path)
    meta_file = metadata_path(csv_path)
    if verify_hash and meta_file.exists():
        expected = json.loads(meta_file.read_text(encoding="utf-8"))["sha256"]
        actual = _sha256(csv_path)
        if actual != expected:
            raise ValueError(f"{csv_path} hash mismatch: file changed since it was downloaded.")
    prices = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"])
    return prices.sort_index()


def get_prices(
    tickers: list[str],
    start: str,
    end_inclusive: str,
    cache_path: str | Path,
    auto_adjust: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return prices from the cache, downloading only if the cache is absent or ``refresh``."""
    cache_path = Path(cache_path)
    if refresh or not cache_path.exists():
        import yfinance as yf

        prices = download_prices(tickers, start, end_inclusive, auto_adjust)
        save_prices(
            prices,
            cache_path,
            {
                "source": "yfinance",
                "yfinance_version": getattr(yf, "__version__", "unknown"),
                "auto_adjust": auto_adjust,
                "requested_start": start,
                "requested_end_inclusive": end_inclusive,
            },
        )
    prices = load_prices(cache_path)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Cached file lacks tickers {missing}; rerun with refresh=True.")
    return prices[tickers]

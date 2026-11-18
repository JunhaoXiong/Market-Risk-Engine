import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_prices() -> pd.DataFrame:
    """Synthetic, well-behaved price panel (business days, 4 assets)."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300, name="Date")
    rets = rng.normal(0.0003, 0.01, size=(len(idx), 4))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=["SPY", "QQQ", "TLT", "GLD"])

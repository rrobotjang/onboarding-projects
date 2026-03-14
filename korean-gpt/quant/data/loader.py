"""
Market Data Loader
Fetches OHLCV data from various sources (CSV, API, or generated).
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    return df


def generate_synthetic_ohlcv(
    ticker: str = "SYN",
    days: int = 252,
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV data for backtesting.
    Uses geometric Brownian motion for realistic price paths.
    """
    rng = np.random.default_rng(seed)
    
    returns = rng.normal(loc=0.0003, scale=volatility, size=days)
    close = start_price * np.exp(np.cumsum(returns))
    
    # Derive OHLV from close
    high = close * (1 + rng.uniform(0, 0.015, days))
    low = close * (1 - rng.uniform(0, 0.015, days))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.integers(100_000, 5_000_000, size=days)

    df = pd.DataFrame({
        "timestamp": pd.bdate_range(start="2023-01-02", periods=days),
        "ticker": ticker,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume,
    })
    return df


def generate_multi_asset(
    tickers: list[str] = None,
    days: int = 252,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic data for multiple assets."""
    if tickers is None:
        tickers = ["005930.KS", "000660.KS", "035420.KS", "BTC-KRW", "ETH-KRW"]

    assets = {}
    for i, ticker in enumerate(tickers):
        vol = 0.015 + (i * 0.005)  # increasing vol per asset
        price = 50_000 + (i * 20_000)
        assets[ticker] = generate_synthetic_ohlcv(
            ticker=ticker, days=days, start_price=price, volatility=vol, seed=42 + i
        )
    return assets


if __name__ == "__main__":
    data = generate_multi_asset()
    for ticker, df in data.items():
        print(f"📊 {ticker}: {len(df)} bars | "
              f"Close range [{df['close'].min():.0f} ~ {df['close'].max():.0f}]")

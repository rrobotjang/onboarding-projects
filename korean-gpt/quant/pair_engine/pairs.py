"""
Pairs Trading Engine
Implements statistical arbitrage via cointegration and z-score mean-reversion.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    hedge_ratio: float = None,
) -> Tuple[pd.Series, float]:
    """
    Compute the spread between two price series.
    If no hedge_ratio is given, uses OLS regression to estimate it.
    """
    if hedge_ratio is None:
        # Simple OLS hedge ratio: y = β * x + ε
        x = series_b.values.reshape(-1, 1)
        y = series_a.values
        beta = np.linalg.lstsq(x, y, rcond=None)[0][0]
        hedge_ratio = beta

    spread = series_a - hedge_ratio * series_b
    return spread, hedge_ratio


def compute_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
    """Rolling z-score of the spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std


def generate_pair_signals(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """
    Generate trading signals from z-score:
      +1 = short spread (z > entry)
      -1 = long spread  (z < -entry)
       0 = flat          (|z| < exit)
    """
    signal = pd.Series(0, index=zscore.index, dtype=float)
    signal[zscore > entry_z] = -1.0   # spread too high → short
    signal[zscore < -entry_z] = 1.0   # spread too low  → long
    signal[zscore.abs() < exit_z] = 0.0

    # Forward-fill positions between entry and exit
    signal = signal.replace(0, np.nan).ffill().fillna(0)
    return signal


if __name__ == "__main__":
    # Quick demo with random walk
    rng = np.random.default_rng(42)
    n = 200
    common = rng.normal(0, 1, n).cumsum()
    a = pd.Series(100 + common + rng.normal(0, 0.5, n).cumsum(), name="A")
    b = pd.Series(50 + 0.5 * common + rng.normal(0, 0.3, n).cumsum(), name="B")

    spread, hr = compute_spread(a, b)
    z = compute_zscore(spread, lookback=20)
    sig = generate_pair_signals(z)

    print(f"Hedge ratio : {hr:.4f}")
    print(f"Signal dist : {sig.value_counts().to_dict()}")

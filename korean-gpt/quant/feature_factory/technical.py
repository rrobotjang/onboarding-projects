"""
Expanded Technical Indicators
RSI, MACD, Bollinger Bands + ATR, Stochastic, Williams %R, CCI, ADX, OBV, VWAP
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Original indicators
# ═══════════════════════════════════════════════════════════════

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, slow: int = 26, fast: int = 12, signal: int = 9) -> pd.DataFrame:
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    df['bb_mid'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * std_dev)
    return df


# ═══════════════════════════════════════════════════════════════
# New indicators
# ═══════════════════════════════════════════════════════════════

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — volatility indicator."""
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift(1)).abs()
    l_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df[f'atr_{period}'] = tr.rolling(window=period).mean()
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    df['stoch_k'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R — momentum indicator (-100 to 0)."""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    df[f'williams_r_{period}'] = ((high_max - df['close']) / (high_max - low_min)) * -100
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df[f'cci_{period}'] = (tp - sma) / (0.015 * mad)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — trend strength."""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = add_atr(df.copy(), period)[f'atr_{period}']

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df[f'adx_{period}'] = dx.ewm(span=period, adjust=False).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    direction = np.sign(df['close'].diff())
    df['obv'] = (direction * df['volume']).cumsum()
    # Normalized OBV rate of change
    df['obv_roc'] = df['obv'].pct_change(periods=10)
    return df


def add_vwap(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume-Weighted Average Price (rolling)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
    return df


def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Price Rate of Change (momentum)."""
    df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
    return df


def add_sma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """SMA crossover signal (Golden/Death Cross)."""
    df[f'sma_{fast}'] = df['close'].rolling(fast).mean()
    df[f'sma_{slow}'] = df['close'].rolling(slow).mean()
    df['sma_cross'] = df[f'sma_{fast}'] - df[f'sma_{slow}']
    return df


def add_trend_filter(df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
    """Long-term trend filter (SMA 200)."""
    if f'sma_{period}' not in df.columns:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
    return df


# ═══════════════════════════════════════════════════════════════
# Wrapper to apply ALL indicators
# ═══════════════════════════════════════════════════════════════

def wrap_generators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all technical indicators."""
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_stochastic(df)
    df = add_williams_r(df)
    df = add_cci(df)
    df = add_adx(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_momentum(df)
    df = add_sma_crossover(df)
    df = add_trend_filter(df)
    return df

#!/usr/bin/env python3
"""
run_strategy.py — Enhanced Backtest with Real KRX Data + Regime Detection
Pipeline:  Real CSV → Features (12) → Regime Detection → Adaptive Signals (10) → MVO → Execution
"""

import os, sys, numpy as np, pandas as pd

from quant.feature_factory.factory import FeatureFactory
from quant.feature_factory.technical import (
    add_rsi, add_macd, add_bollinger_bands,
    add_atr, add_stochastic, add_williams_r,
    add_cci, add_adx, add_obv, add_vwap,
    add_momentum, add_sma_crossover,
)
from quant.signals.ensemble import SignalEnsemble
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.execution.broker import PaperBroker, Order


# ═══════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════

def load_krx_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('timestamp').reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════
# 2. Signal Generators — Regime-Adaptive
# ═══════════════════════════════════════════════════════════════

# --- Mean-Reversion signals ---
def sig_rsi(row) -> float:
    rsi = row.get("rsi_14", 50)
    if pd.isna(rsi): return 0.0
    if rsi > 70: return -1.0
    elif rsi < 30: return 1.0
    return (50 - rsi) / 50

def sig_bb(row) -> float:
    close, upper, lower, mid = row.get("close", 0), row.get("bb_upper", 0), row.get("bb_lower", 0), row.get("bb_mid", 0)
    if pd.isna(upper) or pd.isna(lower) or upper == lower: return 0.0
    return float(np.clip((mid - close) / (upper - lower) * 2, -1, 1))

def sig_stoch(row) -> float:
    k, d = row.get("stoch_k", 50), row.get("stoch_d", 50)
    if pd.isna(k) or pd.isna(d): return 0.0
    if k > 80 and k < d: return -1.0
    elif k < 20 and k > d: return 1.0
    return float(np.clip((50 - k) / 50, -1, 1))

def sig_williams(row) -> float:
    wr = row.get("williams_r_14", -50)
    if pd.isna(wr): return 0.0
    if wr < -80: return 1.0
    elif wr > -20: return -1.0
    return float(np.clip((-50 - wr) / 50, -1, 1))

def sig_cci(row) -> float:
    cci = row.get("cci_20", 0)
    if pd.isna(cci): return 0.0
    if cci > 100: return -1.0
    elif cci < -100: return 1.0
    return float(np.clip(-cci / 100, -1, 1))

def sig_vwap(row) -> float:
    close, vwap = row.get("close", 0), row.get("vwap", 0)
    if pd.isna(vwap) or vwap == 0: return 0.0
    return float(np.clip(-(close - vwap) / vwap * 20, -1, 1))

# --- Trend-Following signals ---
def sig_macd(row) -> float:
    """MACD as trend signal: histogram direction."""
    hist = row.get("macd_hist", 0)
    macd = row.get("macd", 0)
    if pd.isna(hist) or pd.isna(macd): return 0.0
    sig = np.sign(macd) * min(abs(hist) / max(abs(macd), 1), 1.0)
    return float(np.clip(sig, -1, 1))

def sig_adx_trend(row) -> float:
    """ADX trend direction & strength."""
    adx = row.get("adx_14", 0)
    plus_di, minus_di = row.get("plus_di", 0), row.get("minus_di", 0)
    if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di): return 0.0
    if adx < 20: return 0.0
    direction = 1.0 if plus_di > minus_di else -1.0
    return direction * min(adx / 40, 1.0)

def sig_obv(row) -> float:
    roc = row.get("obv_roc", 0)
    if pd.isna(roc): return 0.0
    return float(np.clip(roc * 10, -1, 1))

def sig_momentum(row) -> float:
    """ROC momentum — pure trend following."""
    roc = row.get("roc_10", 0)
    if pd.isna(roc): return 0.0
    return float(np.clip(roc / 5, -1, 1))

def sig_sma_cross(row) -> float:
    """SMA crossover — trend confirmation."""
    cross = row.get("sma_cross", 0)
    close = row.get("close", 1)
    if pd.isna(cross): return 0.0
    return float(np.clip(cross / (close * 0.05), -1, 1))


MEAN_REV_SIGNALS = {'sig_rsi': sig_rsi, 'sig_bb': sig_bb, 'sig_stoch': sig_stoch,
                    'sig_williams': sig_williams, 'sig_cci': sig_cci, 'sig_vwap': sig_vwap}

TREND_SIGNALS = {'sig_macd': sig_macd, 'sig_adx': sig_adx_trend, 'sig_obv': sig_obv,
                 'sig_momentum': sig_momentum, 'sig_sma_cross': sig_sma_cross}

ALL_SIGNALS = {**MEAN_REV_SIGNALS, **TREND_SIGNALS}


def get_regime_weights(adx_val: float) -> dict:
    """
    Regime detection: ADX > 25 → trending market (favor trend signals).
    ADX < 20 → range-bound (favor mean-reversion).
    """
    if pd.isna(adx_val):
        adx_val = 20

    # Trend strength [0, 1]
    trend_strength = np.clip((adx_val - 15) / 30, 0, 1)
    mr_weight = 1 - trend_strength  # mean-reversion weight
    tf_weight = trend_strength       # trend-following weight

    weights = {}
    # Mean-reversion signals
    for name in MEAN_REV_SIGNALS:
        weights[name] = mr_weight * 0.12
    # Trend-following signals
    for name in TREND_SIGNALS:
        weights[name] = tf_weight * 0.15

    # Normalize to sum=1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


# ═══════════════════════════════════════════════════════════════
# 3. Main Backtest
# ═══════════════════════════════════════════════════════════════

def run_backtest(optimizer_method: str = "mean_variance", rebal_freq: int = 1, pos_pct: float = 0.45):
    print("=" * 65)
    print(f"📈  Quant-GPT Enhanced Backtest (Real KRX + Regime Adaptive)")
    print(f"   Optimizer: {optimizer_method} | Rebal: {rebal_freq}d | Pos: {pos_pct*100:.0f}%")
    print("=" * 65)

    # --- Load real data ---
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_krx")
    files = {
        "005930.KS": os.path.join(data_dir, "005930_KS.csv"),
        "000660.KS": os.path.join(data_dir, "000660_KS.csv"),
        "035420.KS": os.path.join(data_dir, "035420_KS.csv"),
    }
    tickers = list(files.keys())
    assets_raw = {}
    for ticker, path in files.items():
        df = load_krx_csv(path)
        assets_raw[ticker] = df
        print(f"  📊 {ticker}: {len(df)} bars | "
              f"₩{df['close'].iloc[0]:,.0f} → ₩{df['close'].iloc[-1]:,.0f}")

    # --- Feature Factory ---
    factory = FeatureFactory()
    for name, func in [
        ("rsi", add_rsi), ("macd", add_macd), ("bb", add_bollinger_bands),
        ("atr", add_atr), ("stoch", add_stochastic), ("wr", add_williams_r),
        ("cci", add_cci), ("adx", add_adx), ("obv", add_obv),
        ("vwap", add_vwap), ("mom", add_momentum), ("sma_cross", add_sma_crossover),
    ]:
        factory.register_generator(name, func)

    enriched = {t: factory.create_features(df) for t, df in assets_raw.items()}

    # --- Setup ---
    optimizer = PortfolioOptimizer(method=optimizer_method)
    broker = PaperBroker(initial_capital=100_000_000)

    n_bars = min(len(df) for df in enriched.values())
    warmup = 60
    returns_window = []

    print(f"\n🚀 Running ({n_bars - warmup} days, regime-adaptive signals)...\n")

    for day_idx in range(warmup, n_bars):
        asset_signals = {}
        prices = {}
        daily_returns = {}

        for ticker in tickers:
            row = enriched[ticker].iloc[day_idx]
            prev_close = enriched[ticker].iloc[day_idx - 1]['close']
            prices[ticker] = row['close']
            daily_returns[ticker] = (row['close'] / prev_close) - 1

            # Regime-adaptive: compute signal weights based on ADX
            adx_val = row.get('adx_14', 20)
            regime_weights = get_regime_weights(adx_val)

            # Compute all signals
            sig_values = {name: func(row) for name, func in ALL_SIGNALS.items()}

            # Weighted ensemble with regime-adapted weights
            signal = sum(sig_values[k] * regime_weights.get(k, 0) for k in sig_values)
            asset_signals[ticker] = float(np.clip(signal, -1, 1))

        returns_window.append(daily_returns)
        if len(returns_window) > 60:
            optimizer.set_returns_history(pd.DataFrame(returns_window[-120:]))

        nav = broker.mark_to_market(prices)

        if (day_idx - warmup) % rebal_freq == 0 and len(returns_window) > 60:
            weights = optimizer.optimize_weights(asset_signals)

            for ticker, weight in weights.items():
                target_value = abs(weight) * nav * pos_pct
                target_qty = int(target_value / prices[ticker])
                current_qty = broker.get_net_position(ticker)
                ts = enriched[ticker].iloc[day_idx]['timestamp'].strftime('%Y-%m-%d')

                if weight > 0:
                    # Target: LONG position
                    if current_qty < 0:
                        # Currently short → cover first
                        broker.submit_order(Order(ticker, 'cover', abs(current_qty), prices[ticker], ts))
                        current_qty = 0
                    # Buy to reach target
                    delta = target_qty - current_qty
                    if delta > 0:
                        broker.submit_order(Order(ticker, 'buy', delta, prices[ticker], ts))
                    elif delta < 0:
                        broker.submit_order(Order(ticker, 'sell', abs(delta), prices[ticker], ts))

                elif weight < 0:
                    # Target: SHORT position
                    if current_qty > 0:
                        # Currently long → sell first
                        broker.submit_order(Order(ticker, 'sell', current_qty, prices[ticker], ts))
                        current_qty = 0
                    # Short to reach target
                    current_short = abs(min(current_qty, 0))
                    delta = target_qty - current_short
                    if delta > 0:
                        broker.submit_order(Order(ticker, 'short', delta, prices[ticker], ts))
                    elif delta < 0:
                        broker.submit_order(Order(ticker, 'cover', abs(delta), prices[ticker], ts))

                else:
                    # weight ≈ 0 → flatten
                    if current_qty > 0:
                        broker.submit_order(Order(ticker, 'sell', current_qty, prices[ticker], ts))
                    elif current_qty < 0:
                        broker.submit_order(Order(ticker, 'cover', abs(current_qty), prices[ticker], ts))

    # --- Final MTM ---
    final_prices = {t: enriched[t].iloc[n_bars - 1]['close'] for t in tickers}
    broker.mark_to_market(final_prices)

    # --- Results ---
    summary = broker.get_summary()
    eq = pd.Series(broker.equity_curve)
    daily_ret = eq.pct_change().dropna()

    sharpe = max_dd = ann_vol = win_rate = 0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        max_dd = ((eq / eq.cummax()) - 1).min() * 100
        ann_vol = daily_ret.std() * np.sqrt(252) * 100
        win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100

    print("\n" + "=" * 65)
    print("🏁  Backtest Results (Long/Short)")
    print("=" * 65)
    print(f"  Period          : {enriched[tickers[0]].iloc[warmup]['timestamp'].date()}"
          f" ~ {enriched[tickers[0]].iloc[n_bars-1]['timestamp'].date()}")
    print(f"  Initial Capital : ₩{broker.initial_capital:>15,.0f}")
    print(f"  Final NAV       : ₩{summary['nav']:>15,.0f}")
    print(f"  Total Return    :  {summary['total_return']:>14.2f}%")
    print(f"  Total Trades    :  {summary['total_trades']:>14d}")
    print(f"  Long Positions  :  {summary.get('long_positions', 0):>14d}")
    print(f"  Short Positions :  {summary.get('short_positions', 0):>14d}")
    print(f"  ────────────────────────────────────")
    print(f"  Sharpe Ratio    :  {sharpe:>14.2f}")
    print(f"  Max Drawdown    :  {max_dd:>13.2f}%")
    print(f"  Annual Vol      :  {ann_vol:>13.2f}%")
    print(f"  Win Rate (daily):  {win_rate:>13.1f}%")
    print("=" * 65)

    return {"method": optimizer_method, "return": summary["total_return"],
            "sharpe": sharpe, "maxdd": max_dd, "trades": summary["total_trades"]}


if __name__ == "__main__":
    print("\n🔄 Running all 3 optimizers (LONG/SHORT enabled)...\n")
    results = []
    for m in ["mean_variance", "risk_parity", "kelly"]:
        r = run_backtest(optimizer_method=m, rebal_freq=1, pos_pct=0.45)
        results.append(r)
        print()

    print("\n" + "=" * 65)
    print("📊 OPTIMIZER COMPARISON (Regime-Adaptive)")
    print("=" * 65)
    print(f"{'Method':<18} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        print(f"{r['method']:<18} {r['return']:>9.2f}% {r['sharpe']:>7.2f}  {r['maxdd']:>8.2f}%  {r['trades']:>7d}")
    print("=" * 65)

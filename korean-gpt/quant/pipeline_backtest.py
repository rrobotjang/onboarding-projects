#!/usr/bin/env python3
"""
pipeline_backtest.py — CLI-driven end-to-end backtest pipeline
Usage:
  python3 pipeline_backtest.py --start 2022-01-01 --end 2024-12-31 \
      --symbols 005930.KS,000660.KS,035420.KS --rebalance 5 --borrow 0.02 --kelly 0.5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add parent dir so quant package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.feature_factory.factory import FeatureFactory
from quant.feature_factory.technical import (
    add_rsi, add_macd, add_bollinger_bands,
    add_atr, add_stochastic, add_williams_r,
    add_cci, add_adx, add_obv, add_vwap,
    add_momentum, add_sma_crossover, add_trend_filter
)
from quant.signals.ensemble import SignalEnsemble
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.execution.broker import PaperBroker, Order


# ───────────────────────────────────────────
# Data
# ───────────────────────────────────────────

def load_or_download(ticker: str, start: str, end: str, data_dir: str) -> pd.DataFrame:
    """Load from CSV cache or download via yfinance."""
    safe_name = ticker.replace(".", "_")
    csv_path = os.path.join(data_dir, f"{safe_name}.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        try:
            import yfinance as yf
            print(f"  ⬇ Downloading {ticker}...")
            # Use longer start date for warmup
            warmup_start = (pd.to_datetime(start) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
            df = yf.download(ticker, start=warmup_start, end=end, auto_adjust=True)
            df = df.reset_index()
            if hasattr(df.columns, 'levels'):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            os.makedirs(data_dir, exist_ok=True)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"  ❌ Failed to download {ticker}: {e}")
            return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('timestamp').reset_index(drop=True)

    # Filter by date range (ensure enough data for warmup)
    if 'timestamp' in df.columns:
        # Keep data before start for technical indicator warmup
        df = df[df['timestamp'] <= end].reset_index(drop=True)

    return df


# ───────────────────────────────────────────
# Signals (regime-adaptive)
# ───────────────────────────────────────────

def sig_rsi(row):
    rsi = row.get("rsi_14", 50)
    if pd.isna(rsi): return 0.0
    if rsi > 70: return -1.0
    elif rsi < 30: return 1.0
    return (50 - rsi) / 50

def sig_bb(row):
    close, upper, lower, mid = row.get("close", 0), row.get("bb_upper", 0), row.get("bb_lower", 0), row.get("bb_mid", 0)
    if pd.isna(upper) or pd.isna(lower) or upper == lower: return 0.0
    return float(np.clip((mid - close) / (upper - lower) * 2, -1, 1))

def sig_stoch(row):
    k, d = row.get("stoch_k", 50), row.get("stoch_d", 50)
    if pd.isna(k) or pd.isna(d): return 0.0
    if k > 80 and k < d: return -1.0
    elif k < 20 and k > d: return 1.0
    return float(np.clip((50 - k) / 50, -1, 1))

def sig_williams(row):
    wr = row.get("williams_r_14", -50)
    if pd.isna(wr): return 0.0
    if wr < -80: return 1.0
    elif wr > -20: return -1.0
    return float(np.clip((-50 - wr) / 50, -1, 1))

def sig_cci(row):
    cci = row.get("cci_20", 0)
    if pd.isna(cci): return 0.0
    return float(np.clip(-cci / 100, -1, 1))

def sig_vwap(row):
    close, vwap = row.get("close", 0), row.get("vwap", 0)
    if pd.isna(vwap) or vwap == 0: return 0.0
    return float(np.clip(-(close - vwap) / vwap * 20, -1, 1))

def sig_macd(row):
    hist, macd = row.get("macd_hist", 0), row.get("macd", 0)
    if pd.isna(hist) or pd.isna(macd): return 0.0
    return float(np.clip(np.sign(macd) * min(abs(hist) / max(abs(macd), 1), 1.0), -1, 1))

def sig_adx(row):
    adx = row.get("adx_14", 0)
    plus_di, minus_di = row.get("plus_di", 0), row.get("minus_di", 0)
    if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di): return 0.0
    if adx < 20: return 0.0
    direction = 1.0 if plus_di > minus_di else -1.0
    return direction * min(adx / 40, 1.0)

def sig_obv(row):
    roc = row.get("obv_roc", 0)
    if pd.isna(roc): return 0.0
    return float(np.clip(roc * 10, -1, 1))

def sig_momentum(row):
    roc = row.get("roc_10", 0)
    if pd.isna(roc): return 0.0
    return float(np.clip(roc / 5, -1, 1))

def sig_sma_cross(row):
    cross, close = row.get("sma_cross", 0), row.get("close", 1)
    if pd.isna(cross): return 0.0
    return float(np.clip(cross / (close * 0.05), -1, 1))


MEAN_REV = {'sig_rsi': sig_rsi, 'sig_bb': sig_bb, 'sig_stoch': sig_stoch,
            'sig_williams': sig_williams, 'sig_cci': sig_cci, 'sig_vwap': sig_vwap}
TREND_FL = {'sig_macd': sig_macd, 'sig_adx': sig_adx, 'sig_obv': sig_obv,
            'sig_momentum': sig_momentum, 'sig_sma_cross': sig_sma_cross}
ALL_SIGS = {**MEAN_REV, **TREND_FL}


def regime_weights(adx_val):
    if pd.isna(adx_val): adx_val = 20
    t = np.clip((adx_val - 15) / 30, 0, 1)
    w = {}
    for n in MEAN_REV: w[n] = (1 - t) * 0.12
    for n in TREND_FL: w[n] = t * 0.15
    s = sum(w.values())
    return {k: v / s for k, v in w.items()} if s > 0 else w


# ───────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────

def run_pipeline(args):
    print("=" * 65)
    print(f"📈  Quant-GPT Pipeline Backtest (Trend-Filtered)")
    print(f"   Symbols  : {', '.join(args.symbols)}")
    print(f"   Period   : {args.start} ~ {args.end}")
    print(f"   Optimizer: kelly={args.kelly} | Rebal: {args.rebalance}d | Borrow: {args.borrow*100:.1f}%")
    print("=" * 65)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_krx")

    # 1. Load data
    assets_raw = {}
    for sym in args.symbols:
        df = load_or_download(sym, args.start, args.end, data_dir)
        if df.empty:
            continue
        assets_raw[sym] = df

    if len(assets_raw) < 1:
        print("❌ No data loaded. Exiting.")
        return

    tickers = list(assets_raw.keys())

    # 2. Feature Factory
    factory = FeatureFactory()
    for name, func in [
        ("rsi", add_rsi), ("macd", add_macd), ("bb", add_bollinger_bands),
        ("atr", add_atr), ("stoch", add_stochastic), ("wr", add_williams_r),
        ("cci", add_cci), ("adx", add_adx), ("obv", add_obv),
        ("vwap", add_vwap), ("mom", add_momentum), ("sma_x", add_sma_crossover),
        ("trend", add_trend_filter),
    ]:
        factory.register_generator(name, func)

    enriched = {t: factory.create_features(df) for t, df in assets_raw.items()}

    # 3. Setup
    optimizer = PortfolioOptimizer(method="kelly")
    broker = PaperBroker(initial_capital=args.capital, margin_rate=1.5)
    broker.short_borrow_cost_annual = args.borrow

    # Synchronize start index based on start date
    start_dt = pd.to_datetime(args.start)
    
    # Calculate global range of bars
    min_bars = min(len(df) for df in enriched.values())
    warmup = 205 # SMA 200 + margin
    
    # Find the index where timestamp >= start_dt
    start_indices = []
    for t in tickers:
        idx = enriched[t][enriched[t]['timestamp'] >= start_dt].index
        if len(idx) > 0: start_indices.append(idx[0])
    
    real_start_idx = max(start_indices) if start_indices else warmup
    if real_start_idx < warmup: real_start_idx = warmup

    returns_window = []
    pos_pct = args.kelly

    print(f"\n🚀 Running from index {real_start_idx} (approx {enriched[tickers[0]].iloc[real_start_idx]['timestamp'].date()})...\n")

    for day_idx in range(real_start_idx, min_bars):
        asset_signals = {}
        prices = {}
        daily_returns = {}

        for ticker in tickers:
            row = enriched[ticker].iloc[day_idx]
            prev_close = enriched[ticker].iloc[day_idx - 1]['close']
            prices[ticker] = row['close']
            daily_returns[ticker] = (row['close'] / prev_close) - 1

            # 1. Base Signal
            rw = regime_weights(row.get('adx_14', 20))
            sv = {name: func(row) for name, func in ALL_SIGS.items()}
            signal = sum(sv[k] * rw.get(k, 0) for k in sv)
            
            # 2. TREND FILTER (SMA 200)
            sma_200 = row.get('sma_200')
            if pd.notna(sma_200):
                if signal > 0 and prices[ticker] < sma_200: # Long in downtrend
                    signal *= 0.2
                elif signal < 0 and prices[ticker] > sma_200: # Short in uptrend
                    signal *= 0.2

            asset_signals[ticker] = float(np.clip(signal, -1, 1))

        returns_window.append(daily_returns)
        if len(returns_window) > 60:
            optimizer.set_returns_history(pd.DataFrame(returns_window[-120:]))

        nav = broker.mark_to_market(prices)

        if (day_idx - real_start_idx) % args.rebalance == 0 and len(returns_window) > 60:
            weights = optimizer.optimize_weights(asset_signals)

            for ticker, weight in weights.items():
                target_value = abs(weight) * nav * pos_pct
                target_qty = int(target_value / prices[ticker])
                current_qty = broker.get_net_position(ticker)
                ts = enriched[ticker].iloc[day_idx]['timestamp'].strftime('%Y-%m-%d')

                if weight > 0:
                    if current_qty < 0:
                        broker.submit_order(Order(ticker, 'cover', abs(current_qty), prices[ticker], ts))
                        current_qty = 0
                    delta = target_qty - current_qty
                    if delta > 0:
                        broker.submit_order(Order(ticker, 'buy', delta, prices[ticker], ts))
                    elif delta < 0:
                        broker.submit_order(Order(ticker, 'sell', abs(delta), prices[ticker], ts))
                elif weight < 0:
                    if current_qty > 0:
                        broker.submit_order(Order(ticker, 'sell', current_qty, prices[ticker], ts))
                        current_qty = 0
                    current_short = abs(min(current_qty, 0))
                    delta = target_qty - current_short
                    if delta > 0:
                        broker.submit_order(Order(ticker, 'short', delta, prices[ticker], ts))
                    elif delta < 0:
                        broker.submit_order(Order(ticker, 'cover', abs(delta), prices[ticker], ts))
                else:
                    if current_qty > 0:
                        broker.submit_order(Order(ticker, 'sell', current_qty, prices[ticker], ts))
                    elif current_qty < 0:
                        broker.submit_order(Order(ticker, 'cover', abs(current_qty), prices[ticker], ts))

    # Final MTM
    final_prices = {t: enriched[t].iloc[min_bars - 1]['close'] for t in tickers}
    broker.mark_to_market(final_prices)

    # Results
    summary = broker.get_summary()
    eq = pd.Series(broker.equity_curve)
    dr = eq.pct_change().dropna()

    sharpe = max_dd = ann_vol = win_rate = 0
    if len(dr) > 1 and dr.std() > 0:
        sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
        max_dd = ((eq / eq.cummax()) - 1).min() * 100
        ann_vol = dr.std() * np.sqrt(252) * 100
        win_rate = (dr > 0).sum() / len(dr) * 100

    print("\n" + "=" * 65)
    print("🏁  Pipeline Backtest Results")
    print("=" * 65)
    print(f"  Period          : {args.start} ~ {args.end}")
    print(f"  Assets          : {', '.join(tickers)}")
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
    print(f"  Borrow Cost     :  {args.borrow*100:>13.1f}%/yr")
    print("=" * 65)
    print("✅ Pipeline complete!\n")


# ───────────────────────────────────────────
# CLI
# ───────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quant-GPT Pipeline Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diversified Global Core
  python3 pipeline_backtest.py --symbols QQQ,BTC-USD,GLD,TLT --kelly 0.3

  # US High Growth
  python3 pipeline_backtest.py --symbols AAPL,MSFT,TSLA,NVDA
        """,
    )
    parser.add_argument("--start", default="2022-01-01", help="Start date (default: 2022-01-01)")
    parser.add_argument("--end", default="2024-12-31", help="End date (default: 2024-12-31)")
    parser.add_argument("--symbols", default="QQQ,BTC-USD,GLD,TLT", help="CSV tickers (default: QQQ,BTC-USD,GLD,TLT)")
    parser.add_argument("--rebalance", type=int, default=1, help="Rebalance days (default: 1)")
    parser.add_argument("--borrow", type=float, default=0.02, help="Short borrow cost (default: 0.02)")
    parser.add_argument("--kelly", type=float, default=0.3, help="Kelly fraction (default: 0.3)")
    parser.add_argument("--capital", type=float, default=100_000_000, help="Initial capital")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.symbols = [s.strip() for s in args.symbols.split(",")]
    run_pipeline(args)

#!/usr/bin/env python3
"""
intraday_pipeline.py — 15m Resolution Backtest + Sentiment Signal
Targets Sharpe 2.0+ by combining high-frequency technicals with news sentiment.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# Add parent dir so quant package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.feature_factory.factory import FeatureFactory
from quant.feature_factory.technical import (
    add_rsi, add_macd, add_bollinger_bands,
    add_atr, add_stochastic, add_williams_r,
    add_cci, add_adx, add_obv, add_vwap,
    add_momentum, add_sma_crossover, add_trend_filter
)
from quant.feature_factory.sentiment import add_sentiment_signal, FileSentimentScorer
from quant.data.chunk_loader import download_intraday_chunks
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.execution.broker import PaperBroker, Order


def load_intraday(ticker: str, interval: str = "15m", period: str = "60d") -> pd.DataFrame:
    """Download intraday data."""
    print(f"  ⬇ Downloading {ticker} ({interval}, {period})...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df = df.reset_index()
    if hasattr(df.columns, 'levels'):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    
    df.columns = [c.strip().lower() for c in df.columns]
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
        
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['close']).sort_values('timestamp').reset_index(drop=True)


# --- Signals (Technical + Sentiment) ---

def sig_rsi(row):
    rsi = row.get("rsi_14", 50)
    if pd.isna(rsi): return 0.0
    return (50 - rsi) / 50

def sig_bb(row):
    close, upper, lower, mid = row.get("close", 0), row.get("bb_upper", 0), row.get("bb_lower", 0), row.get("bb_mid", 0)
    if pd.isna(upper) or pd.isna(lower) or upper == lower: return 0.0
    return float(np.clip((mid - close) / (upper - lower) * 2, -1, 1))

def sig_macd(row):
    hist, macd = row.get("macd_hist", 0), row.get("macd", 0)
    if pd.isna(hist) or pd.isna(macd): return 0.0
    return float(np.clip(np.sign(macd) * min(abs(hist) / max(abs(macd), 1), 1.0), -1, 1))

ALL_SIGS = {'rsi': sig_rsi, 'bb': sig_bb, 'macd': sig_macd}

def run_intraday_backtest(args):
    print("=" * 65)
    print(f"🚀  Quant-GPT INTRADAY Alpha Pipeline")
    print(f"   Symbols  : {', '.join(args.symbols)}")
    print(f"   Interval : {args.interval} | Kelly: {args.kelly}")
    print("=" * 65)

    # 1. Load data
    assets_raw = {}
    for sym in args.symbols:
        # Use chunk loader to bypass yf limits
        # args.period like '60d'
        days = int(args.period.replace('d', ''))
        df = download_intraday_chunks(sym, interval=args.interval, days=days)
        if df.empty: continue
        assets_raw[sym] = df

    if not assets_raw:
        print("❌ No data. Exiting.")
        return {}

    tickers = list(assets_raw.keys())

    # 2. Features
    factory = FeatureFactory()
    # ... (indicators)
    for name, func in [
        ("rsi", add_rsi), ("macd", add_macd), ("bb", add_bollinger_bands),
        ("atr", add_atr), ("stoch", add_stochastic), ("wr", add_williams_r),
        ("cci", add_cci), ("adx", add_adx), ("obv", add_obv),
        ("vwap", add_vwap), ("mom", add_momentum), ("sma_x", add_sma_crossover),
        ("trend", add_trend_filter),
    ]:
        factory.register_generator(name, func)

    # Sentiment Scorer selection
    sentiment_source = "simulation"
    if hasattr(args, 'sentiment_file') and args.sentiment_file:
        sentiment_source = FileSentimentScorer(args.sentiment_file)

    enriched = {}
    for t, df in assets_raw.items():
        print(f"  🛠 Processing features for {t}...")
        df_feat = factory.create_features(df)
        df_feat = add_sentiment_signal(df_feat, t, source=sentiment_source)
        enriched[t] = df_feat

    # 3. Setup
    optimizer = PortfolioOptimizer(method="kelly")
    broker = PaperBroker(initial_capital=args.capital)
    
    min_bars = min(len(df) for df in enriched.values())
    warmup = 205
    pos_pct = args.kelly

    print(f"\n⚡ Processing {min_bars - warmup} bars...\n")

    for idx in range(warmup, min_bars):
        asset_signals = {}
        prices = {}
        
        for t in tickers:
            row = enriched[t].iloc[idx]
            prices[t] = row['close']
            
            # Combine Technical Signals
            sv = {name: func(row) for name, func in ALL_SIGS.items()}
            tech_sig = np.mean(list(sv.values()))
            
            # Sentiment Alpha
            sent_sig = row.get('sentiment', 0.0)
            
            # Final combined signal: Tech (0.3) + Sent (0.7)
            # Sentiment is weighted heavily for 'Information Asymmetry' test
            combined = 0.3 * tech_sig + 0.7 * sent_sig
            
            # Trend Filter (SMA 200)
            sma_200 = row.get('sma_200')
            if pd.notna(sma_200):
                if combined > 0 and prices[t] < sma_200: combined *= 0.2
                elif combined < 0 and prices[t] > sma_200: combined *= 0.2
            
            asset_signals[t] = float(np.clip(combined, -1, 1))

        nav = broker.mark_to_market(prices)

        if (idx - warmup) % args.rebalance == 0:
            weights = optimizer.optimize_weights(asset_signals)

            for t, w in weights.items():
                target_qty = int(abs(w) * nav * pos_pct / prices[t])
                current_qty = broker.get_net_position(t)
                ts = enriched[t].iloc[idx]['timestamp']
                
                if w > 0:
                    if current_qty < 0: broker.submit_order(Order(t, 'cover', abs(current_qty), prices[t], str(ts)))
                    delta = target_qty - broker.get_net_position(t)
                    if delta > 0: broker.submit_order(Order(t, 'buy', delta, prices[t], str(ts)))
                    elif delta < 0: broker.submit_order(Order(t, 'sell', abs(delta), prices[t], str(ts)))
                elif w < 0:
                    if current_qty > 0: broker.submit_order(Order(t, 'sell', current_qty, prices[t], str(ts)))
                    delta = target_qty - abs(broker.get_net_position(t))
                    if delta > 0: broker.submit_order(Order(t, 'short', delta, prices[t], str(ts)))
                    elif delta < 0: broker.submit_order(Order(t, 'cover', abs(delta), prices[t], str(ts)))

    # Results
    summary = broker.get_summary()
    eq = pd.Series(broker.equity_curve)
    dr = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252 * (6.5 * 4)) if not dr.empty else 0 # Annualize for 15m (6.5h/day)
    
    print("-" * 65)
    print(f"🏁  Results: Return {summary['total_return']}% | Sharpe {sharpe:.2f}")
    print("-" * 65)

    return {
        'sharpe': sharpe,
        'return': summary['total_return'],
        'max_dd': ((eq / eq.cummax()) - 1).min() * 100,
        'trades': summary['total_trades'],
        'params': vars(args)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="NVDA,BTC-USD")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--end_date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--kelly", type=float, default=0.45)
    parser.add_argument("--rebalance", type=int, default=1)
    parser.add_argument("--capital", type=float, default=100_000_000)
    parser.add_argument("--sentiment_file", default=None, help="Path to sentiment CSV/JSON")
    args = parser.parse_args()
    args.symbols = args.symbols.split(",")
    run_intraday_backtest(args)

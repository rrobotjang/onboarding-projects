#!/usr/bin/env python3
"""Grid search for optimal strategy parameters."""

import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from quant.feature_factory.factory import FeatureFactory
from quant.feature_factory import technical as tech
from quant.signals.ensemble import SignalEnsemble
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.execution.broker import PaperBroker, Order
from run_strategy import load_krx_csv, SIGNAL_FUNCS, SIGNAL_WEIGHTS


def quick_backtest(method, rebal, pos_pct) -> dict:
    data_dir = os.path.join(os.path.dirname(__file__), "data_krx")
    files = {
        "005930.KS": os.path.join(data_dir, "005930_KS.csv"),
        "000660.KS": os.path.join(data_dir, "000660_KS.csv"),
        "035420.KS": os.path.join(data_dir, "035420_KS.csv"),
    }
    tickers = list(files.keys())
    assets_raw = {t: load_krx_csv(p) for t, p in files.items()}

    factory = FeatureFactory()
    for name, func in [
        ("rsi", tech.add_rsi), ("macd", tech.add_macd), ("bb", tech.add_bollinger_bands),
        ("atr", tech.add_atr), ("stoch", tech.add_stochastic), ("wr", tech.add_williams_r),
        ("cci", tech.add_cci), ("adx", tech.add_adx), ("obv", tech.add_obv),
        ("vwap", tech.add_vwap), ("mom", tech.add_momentum), ("sma", tech.add_sma_crossover),
    ]:
        factory.register_generator(name, func)

    enriched = {t: factory.create_features(df) for t, df in assets_raw.items()}

    ensemble = SignalEnsemble(method="weighted")
    ensemble.set_weights(SIGNAL_WEIGHTS)
    optimizer = PortfolioOptimizer(method=method)
    broker = PaperBroker(initial_capital=100_000_000)

    n_bars = min(len(df) for df in enriched.values())
    warmup = 60
    returns_window = []

    for day_idx in range(warmup, n_bars):
        asset_signals = {}
        prices = {}
        daily_returns = {}

        for ticker in tickers:
            row = enriched[ticker].iloc[day_idx]
            prev_close = enriched[ticker].iloc[day_idx - 1]["close"]
            prices[ticker] = row["close"]
            daily_returns[ticker] = (row["close"] / prev_close) - 1

            sig_values = {name: func(row) for name, func in SIGNAL_FUNCS.items()}
            sig_df = pd.DataFrame([sig_values])
            result = ensemble.generate_unified_signal(sig_df, list(SIGNAL_FUNCS.keys()))
            asset_signals[ticker] = result["final_signal"].iloc[0]

        returns_window.append(daily_returns)
        if len(returns_window) > 60:
            optimizer.set_returns_history(pd.DataFrame(returns_window[-120:]))

        nav = broker.mark_to_market(prices)

        if (day_idx - warmup) % rebal == 0 and len(returns_window) > 60:
            weights = optimizer.optimize_weights(asset_signals)
            for ticker, weight in weights.items():
                target_value = abs(weight) * nav * pos_pct
                target_qty = int(target_value / prices[ticker])
                current_pos = broker.positions.get(ticker)
                current_qty = current_pos.quantity if current_pos else 0
                delta = target_qty - current_qty
                ts = enriched[ticker].iloc[day_idx]["timestamp"].strftime("%Y-%m-%d")
                if delta > 0 and weight > 0:
                    broker.submit_order(Order(ticker, "buy", delta, prices[ticker], ts))
                elif delta < 0 or weight < 0:
                    sell_qty = min(abs(delta), current_qty)
                    if sell_qty > 0:
                        broker.submit_order(Order(ticker, "sell", sell_qty, prices[ticker], ts))

    final_prices = {t: enriched[t].iloc[n_bars - 1]["close"] for t in tickers}
    broker.mark_to_market(final_prices)
    summary = broker.get_summary()
    eq = pd.Series(broker.equity_curve)
    dr = eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    max_dd = ((eq / eq.cummax()) - 1).min() * 100
    return {"method": method, "rebal": rebal, "pos": pos_pct,
            "return": summary["total_return"], "sharpe": sharpe, "maxdd": max_dd,
            "trades": summary["total_trades"]}


if __name__ == "__main__":
    results = []
    configs = [
        (m, r, p)
        for m in ["mean_variance", "risk_parity", "kelly"]
        for r in [1, 3, 5]
        for p in [0.20, 0.30, 0.40, 0.50]
    ]
    total = len(configs)
    for i, (m, r, p) in enumerate(configs):
        print(f"[{i+1}/{total}] {m} | rebal={r}d | pos={p*100:.0f}%", end=" → ", flush=True)
        res = quick_backtest(m, r, p)
        print(f"Return: {res['return']:.2f}% | Sharpe: {res['sharpe']:.2f} | MaxDD: {res['maxdd']:.2f}%")
        results.append(res)

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print("\n" + "=" * 75)
    print("🏆 TOP 5 CONFIGS BY SHARPE RATIO")
    print("=" * 75)
    print(f"{'Rank':<5} {'Method':<18} {'Rebal':<7} {'Pos%':<7} {'Return':<10} {'Sharpe':<8} {'MaxDD':<10}")
    print("-" * 75)
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}   {r['method']:<18} {r['rebal']}d     {r['pos']*100:.0f}%    "
              f"{r['return']:>7.2f}%  {r['sharpe']:>6.2f}   {r['maxdd']:>7.2f}%")
    print("=" * 75)

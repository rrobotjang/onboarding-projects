import time
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.feature_factory.sentiment import LiveSentimentScorer

def test_live_scoring():
    tickers = ['NVDA', 'BTC-USD']
    print("🚀 Initializing Live Sentiment Scorer...")
    scorer = LiveSentimentScorer(tickers)
    
    print("\n📡 Listening for live news events...")
    # Simulate an intraday trading loop ticking every few seconds
    for bar in range(5):
        print(f"\n--- 📊 Trading Bar {bar + 1} | Timestamp: {pd.Timestamp.now()} ---")
        for ticker in tickers:
            score = scorer.get_score(ticker, pd.Timestamp.now())
            if score > 0.3:
                action = "BUY Signal 🟢"
            elif score < -0.3:
                action = "SELL Signal 🔴"
            else:
                action = "NEUTRAL ⚪"
            
            print(f"[{ticker}] Live Sentiment Score: {score:+.2f} -> {action}")
        
        time.sleep(1.5)

if __name__ == "__main__":
    test_live_scoring()

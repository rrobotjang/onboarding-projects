import random
import time
from datetime import datetime
from typing import Dict, List, Optional

class LiveNewsStreamer:
    """
    Simulates a WebSocket or REST polling connection to a live news provider.
    (e.g., Bloomberg, Reuters, NewsAPI)
    """
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.bullish_keywords = ['surge', 'beats', 'upgrade', 'launch', 'record', 'growth', 'acquire']
        self.bearish_keywords = ['misses', 'downgrade', 'lawsuit', 'drop', 'delay', 'loss', 'investigation']
        self.neutral_keywords = ['reports', 'holds', 'announces', 'meeting', 'review']

    def fetch_latest_news(self, ticker: str) -> Optional[Dict]:
        """
        Poll the 'API' for the latest headline regarding a specific ticker.
        In a real scenario, this would be an HTTP GET request.
        """
        # Simulate API latency
        time.sleep(0.1)
        
        # 30% chance of NO news at this exact moment
        if random.random() < 0.3:
            return None
            
        # Generate a mock headline
        sentiment_type = random.choice(['bullish', 'bearish', 'neutral', 'neutral'])
        
        if sentiment_type == 'bullish':
            word = random.choice(self.bullish_keywords)
        elif sentiment_type == 'bearish':
            word = random.choice(self.bearish_keywords)
        else:
            word = random.choice(self.neutral_keywords)
            
        headline = f"{ticker} {word} expectations in latest quarter."
        
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'headline': headline,
            'source': 'MockNewsAPI'
        }

if __name__ == "__main__":
    streamer = LiveNewsStreamer(['NVDA', 'BTC-USD'])
    print("📡 Connecting to Live News Feeds...")
    for _ in range(3):
        for ticker in ['NVDA', 'BTC-USD']:
            news = streamer.fetch_latest_news(ticker)
            if news:
                print(f"[{news['timestamp']}] {news['ticker']}: {news['headline']}")
        time.sleep(1)

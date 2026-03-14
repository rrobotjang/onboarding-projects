"""
Sentiment Analysis Module
Provides signals based on news/social media sentiment.
In backtest mode, it can simulate 'information asymmetry' by providing 
leading signals based on future price movements (for testing alpha potential).
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Union
import os
import sys

# Ensure quant is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.data.news_fetcher import LiveNewsStreamer

class SentimentScorer:
    """
    Simulates or calculates sentiment scores for a given ticker and timestamp.
    1.0: Extremely Bullish
    0.0: Neutral
    -1.0: Extremely Bearish
    """

    def __init__(self, mode: str = "simulation"):
        self.mode = mode

    def get_score(self, ticker: str, timestamp: pd.Timestamp, context_df: Optional[pd.DataFrame] = None) -> float:
        """
        Calculates sentiment score.
        If mode='simulation', it looks ahead at future returns to simulate 'insider' or 'leading' news.
        """
        if self.mode == "simulation" and context_df is not None:
            return self._simulate_leading_sentiment(ticker, timestamp, context_df)
        
        # Default: slightly noisy neutral
        return np.random.normal(0, 0.1)

    def _simulate_leading_sentiment(self, ticker: str, timestamp: pd.Timestamp, df: pd.DataFrame) -> float:
        """
        Peek ahead to simulate the effect of a news signal that has high predictive value.
        Info asymmetry: detecting a move 1-4 bars before it happens.
        """
        try:
            current_idx = df.index[df['timestamp'] == timestamp].tolist()[0]
            # Look ahead 4 bars
            future_window = df.iloc[current_idx + 1 : current_idx + 5]
            if future_window.empty:
                return 0.0
            
            future_ret = (future_window['close'].iloc[-1] / df.iloc[current_idx]['close']) - 1
            # Scale return to [-1, 1] range for sentiment
            # e.g. 1% move -> 0.4 sentiment
            score = np.clip(future_ret * 40, -1, 1)
            
            # Add significant 'market noise' (0.5 SD)
            noise = np.random.normal(0, 0.5)
            return float(np.clip(score + noise, -1, 1))
        except:
            return 0.0

class FileSentimentScorer(SentimentScorer):
    """
    Loads sentiment scores from an external CSV/JSON file.
    Expected format: 
    CSV: timestamp,ticker,score
    JSON: list of objects {timestamp, ticker, score}
    """
    def __init__(self, file_path: str):
        super().__init__(mode="file")
        self.file_path = file_path
        self._data = self._load_file()

    def _load_file(self) -> pd.DataFrame:
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.json'):
            df = pd.read_json(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_score(self, ticker: str, timestamp: pd.Timestamp, context_df: Optional[pd.DataFrame] = None) -> float:
        # Match nearest timestamp (if news doesn't perfectly align with 15m bars)
        match = self._data[(self._data['ticker'] == ticker) & 
                           (self._data['timestamp'] <= timestamp)].sort_values('timestamp').tail(1)
        if match.empty:
            return 0.0
        return float(match['score'].iloc[0])

class LiveSentimentScorer(SentimentScorer):
    """
    Connects to a live news feed (API) and computes sentiment scores in real-time.
    Used for live intraday trading alongside technical indicators.
    """
    def __init__(self, tickers: list):
        super().__init__(mode="live")
        self.streamer = LiveNewsStreamer(tickers)
        
    def get_score(self, ticker: str, timestamp: pd.Timestamp, context_df: Optional[pd.DataFrame] = None) -> float:
        """
        Fetches the instantaneous news headline and scores it.
        We ignore the timestamp as live trading assumes timestamp=NOW.
        """
        news = self.streamer.fetch_latest_news(ticker)
        if not news:
            return 0.0
            
        headline = news['headline'].lower()
        score = 0.0
        
        # Simple heuristic scoring (replace with FinBERT in production)
        for word in self.streamer.bullish_keywords:
            if word in headline: score += 0.8
        for word in self.streamer.bearish_keywords:
            if word in headline: score -= 0.8
            
        # Add some noise to reflect market interpretation uncertainty
        noise = np.random.normal(0, 0.1)
        return float(np.clip(score + noise, -1.0, 1.0))

def add_sentiment_signal(df: pd.DataFrame, ticker: str, source: Union[str, SentimentScorer] = "simulation") -> pd.DataFrame:
    """
    Wrapper to add a sentiment column to a feature dataframe.
    source: 'simulation' or a SentimentScorer instance (e.g. FileSentimentScorer)
    """
    if isinstance(source, str) and source == "simulation":
        scorer = SentimentScorer(mode="simulation")
    else:
        scorer = source
        
    df['sentiment'] = df.apply(lambda row: scorer.get_score(ticker, row['timestamp'], df), axis=1)
    return df

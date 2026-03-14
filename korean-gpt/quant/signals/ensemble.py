
import pandas as pd
import numpy as np
from typing import List, Dict, Union

class SignalEnsemble:
    """
    Combines multiple alpha signals into a single unified trading signal.
    Supports simple averaging, weighted averaging, and rank-based voting.
    """
    def __init__(self, method: str = "mean"):
        self.method = method
        self.weights: Dict[str, float] = {}

    def set_weights(self, weights: Dict[str, float]):
        """Sets weights for weighted averaging."""
        self.weights = weights

    def generate_unified_signal(self, df: pd.DataFrame, signal_cols: List[str]) -> pd.DataFrame:
        """
        Combines signals from signal_cols into a 'final_signal' column.
        """
        if self.method == "mean":
            df['final_signal'] = df[signal_cols].mean(axis=1)
        
        elif self.method == "weighted":
            if not self.weights:
                print("⚠️ No weights set, falling back to mean.")
                df['final_signal'] = df[signal_cols].mean(axis=1)
            else:
                weighted_sum = sum(df[col] * self.weights.get(col, 1.0) for col in signal_cols)
                total_weight = sum(self.weights.get(col, 1.0) for col in signal_cols)
                df['final_signal'] = weighted_sum / total_weight
                
        elif self.method == "rank":
            # Normalized rank-based voting
            ranks = df[signal_cols].rank(axis=1, pct=True)
            df['final_signal'] = ranks.mean(axis=1)
            
        # Clipping/Normalization
        df['final_signal'] = df['final_signal'].clip(-1, 1)
        
        return df

if __name__ == "__main__":
    # Sample Test
    data = {
        'sig_rsi': [0.1, -0.2, 0.5, 0.8, -0.4],
        'sig_macd': [0.2, -0.1, 0.4, 0.7, -0.3],
        'sig_llm': [0.5, 0.3, -0.1, 0.9, -0.2]
    }
    sample_df = pd.DataFrame(data)
    
    ensemble = SignalEnsemble(method="weighted")
    ensemble.set_weights({'sig_rsi': 0.2, 'sig_macd': 0.3, 'sig_llm': 0.5})
    
    result = ensemble.generate_unified_signal(sample_df, ['sig_rsi', 'sig_macd', 'sig_llm'])
    print(result)

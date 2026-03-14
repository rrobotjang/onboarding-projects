
import pandas as pd
from typing import List, Callable, Dict

class FeatureFactory:
    """
    A modular factory to generate features for quantitative trading.
    It allows registering and applying multiple feature generators to a dataframe.
    """
    def __init__(self):
        self.generators: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}

    def register_generator(self, name: str, func: Callable[[pd.DataFrame], pd.DataFrame]):
        """Registers a new feature generator function."""
        print(f"✅ Registered generator: {name}")
        self.generators[name] = func

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all registered generators to the dataframe."""
        feature_df = df.copy()
        for name, func in self.generators.items():
            print(f"🚀 Applying generator: {name}...")
            feature_df = func(feature_df)
        return feature_df

if __name__ == "__main__":
    # Sample Test
    import numpy as np
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'open': np.random.randn(10).cumsum() + 100,
        'high': np.random.randn(10).cumsum() + 105,
        'low': np.random.randn(10).cumsum() + 95,
        'close': np.random.randn(10).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 10)
    }
    sample_df = pd.DataFrame(data)
    
    factory = FeatureFactory()
    
    # Simple dummy generator
    def pct_change_gen(df):
        df['returns'] = df['close'].pct_change()
        return df
        
    factory.register_generator("returns", pct_change_gen)
    result = factory.create_features(sample_df)
    print(result.head())

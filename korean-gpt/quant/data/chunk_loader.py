import yfinance as yf
import pandas as pd
import os
from datetime import timedelta, datetime

def download_intraday_chunks(ticker: str, interval: str = "15m", days: int = 100, end_date: Union[datetime, str, None] = None, data_dir: str = "data_intraday") -> pd.DataFrame:
    """
    Downloads intraday data in chunks to bypass yfinance limits.
    """
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
    os.makedirs(data_dir, exist_ok=True)
    all_chunks = []
    
    end_date = datetime.now()
    # 15m limit is 60 days, we use 50-day chunks for safety
    chunk_size = 50 if interval == "15m" else 6 
    
    print(f"🚀 Chunk-loading {days} days of {interval} data for {ticker}...")
    
    for i in range(0, days, chunk_size):
        chunk_end = end_date - timedelta(days=i)
        chunk_start = chunk_end - timedelta(days=chunk_size)
        
        start_str = chunk_start.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        print(f"  📅 Chunk ({start_str} to {end_str})")
        try:
            chunk = yf.download(ticker, start=start_str, end=end_str, interval=interval, auto_adjust=True, progress=False)
            if not chunk.empty:
                all_chunks.append(chunk)
        except Exception as e:
            print(f"  ⚠️ Error on chunk: {e}")
            
    if not all_chunks:
        return pd.DataFrame()
        
    df = pd.concat(all_chunks).sort_index()
    # Handle yfinance multi-index if multi-ticker (though we pass one ticker)
    if hasattr(df.columns, 'levels'):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    
    # Standardize to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Drop duplicates in case of overlap
    df = df[~df.index.duplicated(keep='first')]
    
    # Save for persistence
    safe_name = ticker.replace(".", "_")
    csv_path = os.path.join(data_dir, f"{safe_name}_{interval}.csv")
    df.to_csv(csv_path)
    
    df.index.name = 'timestamp'
    return df.reset_index()

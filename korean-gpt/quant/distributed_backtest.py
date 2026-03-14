import pandas as pd
import numpy as np
import argparse
import itertools
import os
import sys
from typing import List, Dict, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent dir so quant package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.intraday_pipeline import run_intraday_backtest

def backtest_worker(params: Dict[str, Any]):
    """
    Worker function for a single backtest run using Multiprocessing.
    """
    from io import StringIO
    import sys
    
    # Caputre output to keep the main console clean
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Re-construct Namespace for argparse compatibility in the pipeline
        args_ns = argparse.Namespace(**params)
        results = run_intraday_backtest(args_ns)
    except Exception as e:
        results = {'error': str(e), 'params': params}
    finally:
        sys.stdout = old_stdout
        
    return results

class DistributedBacktester:
    """
    Orchestrates distributed backtests and parameter sweeps using standard Multiprocessing.
    """
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers

    def run_sweep(self, base_args: Dict[str, Any], sweep_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Runs a grid search over specified parameters.
        Example sweep_params: {'kelly': [0.1, 0.3, 0.5], 'rebalance': [1, 5]}
        """
        keys = sweep_params.keys()
        values = sweep_params.values()
        combinations = list(itertools.product(*values))
        
        task_list = []
        for combo in combinations:
            task_args = base_args.copy()
            for k, v in zip(keys, combo):
                task_args[k] = v
            # Ensure symbols list is handled correctly if passed as string
            if isinstance(task_args['symbols'], str):
                task_args['symbols'] = [s.strip() for s in task_args['symbols'].split(",")]
            task_list.append(task_args)
            
        print(f"🛰  Launching {len(task_list)} parallel backtest tasks using ProcessPoolExecutor...")
        
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(backtest_worker, arg): arg for arg in task_list}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"  ❌ Worker failed: {e}")
                    
        return results

if __name__ == "__main__":
    tester = DistributedBacktester()
    
    # Example usage for parameter sweep
    base = {
        'symbols': ['NVDA', 'BTC-USD'],
        'interval': '1d',
        'period': '500d',
        'end_date': '2024-12-01',
        'capital': 100_000_000,
        'kelly': 0.3,
        'rebalance': 1,
        'borrow': 0.02
    }
    
    sweep = {
        'kelly': [0.2, 0.4],
        'rebalance': [1, 2]
    }
    
    final_results = tester.run_sweep(base, sweep)
    
    print("\n✅ Sweep Results Summary:")
    print("-" * 45)
    for res in sorted(final_results, key=lambda x: x.get('sharpe', 0), reverse=True):
        if 'error' in res:
            print(f"❌ Error: {res['error']}")
        else:
            print(f"📊 Kelly: {res['params']['kelly']} | Rebal: {res['params']['rebalance']} | Sharpe: {res['sharpe']:.2f}")

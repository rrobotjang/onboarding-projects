"""
Advanced Portfolio Optimizer
Supports: Equal Weight, Signal-Weighted, Mean-Variance (Markowitz), Risk Parity, Kelly Criterion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, method: str = "equal_weight"):
        self.method = method
        self._returns_history: Optional[pd.DataFrame] = None

    def set_returns_history(self, returns_df: pd.DataFrame):
        """Set historical returns matrix for covariance-based methods."""
        self._returns_history = returns_df

    # ─────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────
    def optimize_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        assets = list(signals.keys())
        scores = np.array(list(signals.values()))

        if self.method == "equal_weight":
            return self._equal_weight(assets)

        elif self.method == "signal_weighted":
            return self._signal_weighted(assets, scores)

        elif self.method == "mean_variance":
            return self._mean_variance(assets, scores)

        elif self.method == "risk_parity":
            return self._risk_parity(assets)

        elif self.method == "kelly":
            return self._kelly(assets, scores)

        return self._equal_weight(assets)

    # ─────────────────────────────────────────────────────
    # 1. Equal Weight
    # ─────────────────────────────────────────────────────
    def _equal_weight(self, assets: List[str]) -> Dict[str, float]:
        w = 1.0 / len(assets)
        return {a: w for a in assets}

    # ─────────────────────────────────────────────────────
    # 2. Signal-Weighted
    # ─────────────────────────────────────────────────────
    def _signal_weighted(self, assets: List[str], scores: np.ndarray) -> Dict[str, float]:
        abs_s = np.abs(scores)
        total = abs_s.sum()
        if total == 0:
            return self._equal_weight(assets)
        return {a: (abs_s[i] / total) * np.sign(scores[i]) for i, a in enumerate(assets)}

    # ─────────────────────────────────────────────────────
    # 3. Mean-Variance Optimization (Markowitz)
    # ─────────────────────────────────────────────────────
    def _mean_variance(self, assets: List[str], scores: np.ndarray) -> Dict[str, float]:
        """Max Sharpe Ratio via scipy.optimize."""
        n = len(assets)

        if self._returns_history is None or len(self._returns_history) < 30:
            return self._signal_weighted(assets, scores)

        ret_cols = [a for a in assets if a in self._returns_history.columns]
        if len(ret_cols) < n:
            return self._signal_weighted(assets, scores)

        mu = self._returns_history[ret_cols].mean().values * 252  # annualize
        cov = self._returns_history[ret_cols].cov().values * 252

        # Blend signal views into expected returns (High-conviction Black-Litterman)
        signal_mu = scores * np.abs(mu).max() * 0.8
        blended_mu = 0.3 * mu + 0.7 * signal_mu  # 70% signal conviction

        def neg_sharpe(w):
            port_ret = w @ blended_mu
            port_vol = np.sqrt(w @ cov @ w)
            return -(port_ret / (port_vol + 1e-8))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}]
        bounds = [(-0.8, 0.8)] * n  # allow more concentration
        x0 = np.ones(n) / n

        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            w = result.x
        else:
            w = np.ones(n) / n

        return {assets[i]: float(w[i]) for i in range(n)}

    # ─────────────────────────────────────────────────────
    # 4. Risk Parity
    # ─────────────────────────────────────────────────────
    def _risk_parity(self, assets: List[str]) -> Dict[str, float]:
        """Inverse-volatility weighting (simplified Risk Parity)."""
        n = len(assets)

        if self._returns_history is None or len(self._returns_history) < 30:
            return self._equal_weight(assets)

        ret_cols = [a for a in assets if a in self._returns_history.columns]
        if len(ret_cols) < n:
            return self._equal_weight(assets)

        vols = self._returns_history[ret_cols].std().values
        inv_vol = 1.0 / (vols + 1e-8)
        w = inv_vol / inv_vol.sum()
        return {assets[i]: float(w[i]) for i in range(n)}

    # ─────────────────────────────────────────────────────
    # 5. Kelly Criterion
    # ─────────────────────────────────────────────────────
    def _kelly(self, assets: List[str], scores: np.ndarray) -> Dict[str, float]:
        """Simplified Kelly: f* = μ / σ² (clipped)."""
        n = len(assets)

        if self._returns_history is None or len(self._returns_history) < 30:
            return self._signal_weighted(assets, scores)

        ret_cols = [a for a in assets if a in self._returns_history.columns]
        if len(ret_cols) < n:
            return self._signal_weighted(assets, scores)

        mu = self._returns_history[ret_cols].mean().values
        var = self._returns_history[ret_cols].var().values

        kelly_f = mu / (var + 1e-8)
        # Scale by signal direction — full Kelly with clipping
        kelly_f = kelly_f * np.sign(scores) * 0.5
        kelly_f = np.clip(kelly_f, -0.8, 0.8)

        total = np.abs(kelly_f).sum()
        if total == 0:
            return self._equal_weight(assets)
        w = kelly_f / total
        return {assets[i]: float(w[i]) for i in range(n)}


if __name__ == "__main__":
    # Quick smoke test
    signals = {'A': 0.8, 'B': 0.4, 'C': -0.2}

    for method in ['equal_weight', 'signal_weighted', 'mean_variance', 'risk_parity', 'kelly']:
        opt = PortfolioOptimizer(method=method)
        # MVO/RP/Kelly need returns history
        rng = np.random.default_rng(42)
        ret_df = pd.DataFrame(rng.normal(0.001, 0.02, (100, 3)), columns=['A', 'B', 'C'])
        opt.set_returns_history(ret_df)
        w = opt.optimize_weights(signals)
        print(f"{method:>20s}: {w}")

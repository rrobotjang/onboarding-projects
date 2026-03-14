"""
Execution Layer — Long/Short Paper Broker
Supports: buy, sell, short_sell, cover (buy-to-cover)
Tracks: margin, short P&L, borrowing cost
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class Order:
    ticker: str
    side: str          # "buy", "sell", "short", "cover"
    quantity: float
    price: float
    timestamp: str
    order_type: str = "market"
    status: str = "pending"


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0       # positive = long, negative = short
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0


class PaperBroker:
    """
    Paper trading broker with full long/short support.
    - buy:   open or add to long position
    - sell:  reduce or close long position
    - short: open or add to short position (borrow & sell)
    - cover: reduce or close short position (buy & return)
    """

    def __init__(self, initial_capital: float = 100_000_000, margin_rate: float = 1.5):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.margin_rate = margin_rate  # margin requirement for shorts (150%)
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[dict] = []
        self.equity_curve: List[float] = []
        self.short_borrow_cost_annual = 0.02  # 2% annual borrow cost

    def submit_order(self, order: Order) -> bool:
        """Execute an order immediately at the given price."""
        cost = order.quantity * order.price

        if order.side == "buy":
            if cost > self.capital:
                return False  # insufficient capital
            self.capital -= cost
            pos = self.positions.get(order.ticker, Position(ticker=order.ticker))
            total_qty = pos.quantity + order.quantity
            if total_qty > 0:
                old_cost = pos.avg_price * max(pos.quantity, 0)
                pos.avg_price = (old_cost + cost) / total_qty
            pos.quantity = total_qty
            self.positions[order.ticker] = pos

        elif order.side == "sell":
            pos = self.positions.get(order.ticker)
            if pos is None or pos.quantity < order.quantity:
                return False  # insufficient long position
            pos.quantity -= order.quantity
            self.capital += cost
            if pos.quantity == 0:
                del self.positions[order.ticker]

        elif order.side == "short":
            # Short sell: borrow shares and sell at market price
            # Margin required = cost * margin_rate
            margin_required = cost * self.margin_rate
            if margin_required > self.capital:
                return False  # insufficient margin
            # Receive cash from selling borrowed shares
            self.capital += cost
            pos = self.positions.get(order.ticker, Position(ticker=order.ticker))
            if pos.quantity >= 0 and pos.quantity > 0:
                # Already long — close long first, then go short
                return False  # don't allow flipping in one order
            old_short = abs(min(pos.quantity, 0))
            new_short = old_short + order.quantity
            old_cost = pos.avg_price * old_short
            pos.avg_price = (old_cost + cost) / new_short
            pos.quantity = -new_short  # negative = short
            self.positions[order.ticker] = pos

        elif order.side == "cover":
            # Buy-to-cover: return borrowed shares
            pos = self.positions.get(order.ticker)
            if pos is None or pos.quantity >= 0:
                return False  # no short position to cover
            short_qty = abs(pos.quantity)
            cover_qty = min(order.quantity, short_qty)
            # Pay to buy back shares
            cover_cost = cover_qty * order.price
            if cover_cost > self.capital:
                return False  # insufficient capital to cover
            self.capital -= cover_cost
            pos.quantity += cover_qty  # moves toward 0
            if pos.quantity == 0:
                del self.positions[order.ticker]

        order.status = "filled"
        self.trade_log.append({
            "timestamp": order.timestamp,
            "ticker": order.ticker,
            "side": order.side,
            "qty": order.quantity,
            "price": order.price,
        })
        return True

    def mark_to_market(self, prices: Dict[str, float]):
        """Update unrealized P&L for all positions (long & short)."""
        nav = self.capital
        for ticker, pos in self.positions.items():
            if ticker in prices:
                current_price = prices[ticker]
                if pos.quantity > 0:
                    # Long: profit when price goes up
                    pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity
                    nav += current_price * pos.quantity
                elif pos.quantity < 0:
                    # Short: profit when price goes down
                    short_qty = abs(pos.quantity)
                    pos.unrealized_pnl = (pos.avg_price - current_price) * short_qty
                    # Short liability = current market value of borrowed shares
                    nav -= current_price * short_qty
                    # Daily borrow cost
                    daily_cost = pos.avg_price * short_qty * self.short_borrow_cost_annual / 252
                    self.capital -= daily_cost
        self.equity_curve.append(nav)
        return nav

    def get_net_position(self, ticker: str) -> float:
        """Get net position quantity (positive=long, negative=short, 0=flat)."""
        pos = self.positions.get(ticker)
        return pos.quantity if pos else 0.0

    def get_summary(self) -> dict:
        nav = self.equity_curve[-1] if self.equity_curve else self.capital
        long_count = sum(1 for p in self.positions.values() if p.quantity > 0)
        short_count = sum(1 for p in self.positions.values() if p.quantity < 0)
        return {
            "capital": round(self.capital, 2),
            "nav": round(nav, 2),
            "total_return": round((nav / self.initial_capital - 1) * 100, 2),
            "open_positions": len(self.positions),
            "long_positions": long_count,
            "short_positions": short_count,
            "total_trades": len(self.trade_log),
        }


if __name__ == "__main__":
    broker = PaperBroker(initial_capital=10_000_000)

    # Long trade
    broker.submit_order(Order("005930.KS", "buy", 100, 50000, "2024-01-02"))
    nav = broker.mark_to_market({"005930.KS": 52000})
    print(f"After buy+price up: NAV ₩{nav:,.0f}")

    # Short trade
    broker.submit_order(Order("035420.KS", "short", 10, 300000, "2024-01-02"))
    nav = broker.mark_to_market({"005930.KS": 52000, "035420.KS": 280000})
    print(f"After short+price down: NAV ₩{nav:,.0f} (short profit!)")

    # Cover
    broker.submit_order(Order("035420.KS", "cover", 10, 280000, "2024-01-03"))
    nav = broker.mark_to_market({"005930.KS": 52000, "035420.KS": 280000})

    summary = broker.get_summary()
    print(f"\n📊 Summary: {summary}")

"""Order book reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from sortedcontainers import SortedDict


@dataclass
class _OrderState:
    side: int
    price: float
    qty: float


class OrderBookRector:
    """Reconstructs an order book snapshot stream from order/trade events."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._orders: Dict[int, _OrderState] = {}
        self._bids: SortedDict[float, float] = SortedDict()
        self._asks: SortedDict[float, float] = SortedDict()
        self._cum_trade_qty = 0.0
        self._cum_trade_amt = 0.0
        self._last_price = None

    @staticmethod
    def author() -> str:
        return "OpenAI Assistant"

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _update_level(self, book: SortedDict[float, float], price: float, delta_qty: float) -> None:
        new_qty = book.get(price, 0.0) + delta_qty
        if new_qty <= 0:
            book.pop(price, None)
        else:
            book[price] = new_qty

    def _apply_insert(self, row: pd.Series) -> None:
        order_flag = str(row.get("msg_order_flag", ""))
        if order_flag != "2":
            self._log(f"Ignored non-limit order insert flag={order_flag}")
            return
        order_id = int(row["order_id"])
        side = int(row["order_side"])
        price = float(row["order_opx"])
        qty = float(row["order_qty"])
        self._orders[order_id] = _OrderState(side=side, price=price, qty=qty)
        book = self._bids if side == 1 else self._asks
        self._update_level(book, price, qty)
        self._log(f"Insert order {order_id} side={side} price={price} qty={qty}")

    def _apply_cancel(self, row: pd.Series) -> None:
        order_id = int(row["order_id"])
        cancel_qty = float(row["order_qty"])
        order_state = self._orders.get(order_id)
        if not order_state:
            self._log(f"Cancel ignored for unknown order {order_id}")
            return
        cancel_qty = min(cancel_qty, order_state.qty)
        order_state.qty -= cancel_qty
        book = self._bids if order_state.side == 1 else self._asks
        self._update_level(book, order_state.price, -cancel_qty)
        if order_state.qty <= 0:
            self._orders.pop(order_id, None)
        self._log(f"Cancel order {order_id} qty={cancel_qty}")

    def _apply_trade(self, row: pd.Series) -> None:
        trade_qty = float(row["trade_tvl"])
        trade_price = float(row["trade_tpx"])
        self._cum_trade_qty += trade_qty
        self._cum_trade_amt += trade_qty * trade_price
        self._last_price = trade_price

        self._update_level(self._bids, trade_price, -trade_qty)
        self._update_level(self._asks, trade_price, -trade_qty)
        self._log(f"Trade qty={trade_qty} price={trade_price}")

    def _top_levels(self, book: SortedDict[float, float], depth: int, reverse: bool) -> List[Tuple[float, float]]:
        if not reverse:
            return list(book.items()[:depth])
        levels: List[Tuple[float, float]] = []
        for price in reversed(book.keys()):
            levels.append((price, book[price]))
            if len(levels) >= depth:
                break
        return levels

    def _build_snapshot(self, row: pd.Series) -> Dict[str, object]:
        snapshot: Dict[str, object] = {
            "skey": row.get("skey"),
            "hhmmss_nano": row.get("hhmmss_nano"),
            "yyyymmdd": row.get("yyyymmdd"),
            "obe_seq_num": row.get("obe_seq_num"),
            "cum_trade_tvl": self._cum_trade_qty,
            "cum_trade_amt": self._cum_trade_amt,
            "last": self._last_price,
        }
        asks = self._top_levels(self._asks, 10, reverse=False)
        bids = self._top_levels(self._bids, 10, reverse=True)
        for idx in range(1, 11):
            ask_price, ask_qty = (asks[idx - 1] if idx <= len(asks) else (None, None))
            bid_price, bid_qty = (bids[idx - 1] if idx <= len(bids) else (None, None))
            snapshot[f"ask_{idx}_opx"] = ask_price
            snapshot[f"ask_{idx}_qty"] = ask_qty
            snapshot[f"bid_{idx}_opx"] = bid_price
            snapshot[f"bid_{idx}_qty"] = bid_qty
        return snapshot

    def apply_construction(self, md_order: pd.DataFrame, md_trade: pd.DataFrame) -> pd.DataFrame:
        order_events = md_order.copy()
        order_events["_event_type"] = "order"
        trade_events = md_trade.copy()
        trade_events["_event_type"] = "trade"
        events = pd.concat([order_events, trade_events], ignore_index=True, sort=False)

        sort_keys = [key for key in ["yyyymmdd", "hhmmss_nano", "obe_seq_num", "msg_seq_num"] if key in events.columns]
        if sort_keys:
            events = events.sort_values(sort_keys, kind="mergesort")

        snapshots: List[Dict[str, object]] = []
        for _, row in events.iterrows():
            if row["_event_type"] == "order":
                msg_type = str(row.get("msg_order_type", ""))
                if msg_type == "1":
                    self._apply_insert(row)
                elif msg_type == "3":
                    self._apply_cancel(row)
                else:
                    self._log(f"Unknown order type {msg_type}")
            else:
                self._apply_trade(row)
            snapshots.append(self._build_snapshot(row))

        return pd.DataFrame(snapshots)

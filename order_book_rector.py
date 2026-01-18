"""Order book reconstruction utilities."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

_TIME_WINDOW_START = 92500 * 1_000_000_000
_TIME_WINDOW_END = 145700 * 1_000_000_000


@dataclass
class _OrderState:
    side: int
    price: Optional[float]
    qty: float
    is_limit: bool


class OrderBookRector:
    """Reconstructs an order book snapshot stream from order/trade events."""

    def __init__(
        self,
        verbose: bool = False,
        limit_state_detector: Optional[Callable[[pd.Series], Optional[str]]] = None,
    ) -> None:
        self.verbose = verbose
        self._orders: Dict[int, _OrderState] = {}
        self._bids: SortedDict[float, float] = SortedDict()
        self._asks: SortedDict[float, float] = SortedDict()
        self._cum_trade_qty = 0.0
        self._cum_trade_amt = 0.0
        self._last_price = None
        self._limit_state_detector = limit_state_detector

    @staticmethod
    def author() -> str:
        return "orderbook_rector"

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
        order_id = int(row["order_id"])
        side = int(row["order_side"])
        qty = float(row["order_qty"])
        order_flag = str(row.get("msg_order_flag", ""))
        if order_flag == "3":
            best_bid = next(reversed(self._bids.keys()), None) if self._bids else None
            best_ask = next(iter(self._asks.keys()), None) if self._asks else None
            reference_price = best_bid if side == 1 else best_ask
            if reference_price is None:
                self._orders[order_id] = _OrderState(side=side, price=None, qty=qty, is_limit=False)
                self._log(f"Insert best-of-own order {order_id} side={side} qty={qty} (no reference)")
                return
            self._orders[order_id] = _OrderState(side=side, price=reference_price, qty=qty, is_limit=True)
            book = self._bids if side == 1 else self._asks
            self._update_level(book, reference_price, qty)
            self._log(f"Insert best-of-own order {order_id} side={side} price={reference_price} qty={qty}")
            return
        if order_flag == "2":
            price = float(row["order_opx"])
            self._orders[order_id] = _OrderState(side=side, price=price, qty=qty, is_limit=True)
            book = self._bids if side == 1 else self._asks
            self._update_level(book, price, qty)
            self._log(f"Insert order {order_id} side={side} price={price} qty={qty}")
            return
        self._orders[order_id] = _OrderState(side=side, price=None, qty=qty, is_limit=False)
        self._log(f"Insert non-limit order {order_id} side={side} qty={qty}")

    def _apply_cancel(self, row: pd.Series) -> None:
        order_id = int(row["order_id"])
        cancel_qty = float(row["order_qty"])
        order_state = self._orders.get(order_id)
        if not order_state:
            self._log(f"Cancel ignored for unknown order {order_id}")
            return
        cancel_qty = min(cancel_qty, order_state.qty)
        order_state.qty -= cancel_qty
        if order_state.is_limit and order_state.price is not None:
            book = self._bids if order_state.side == 1 else self._asks
            self._update_level(book, order_state.price, -cancel_qty)
        if order_state.qty <= 0:
            self._orders.pop(order_id, None)
        self._log(f"Cancel order {order_id} qty={cancel_qty}")

    def _apply_trade(self, row: pd.Series) -> None:
        def get_field(name: str) -> object:
            if hasattr(row, "get"):
                return row.get(name)
            return getattr(row, name)

        trade_qty = float(get_field("trade_tvl"))
        trade_price = float(get_field("trade_tpx"))
        self._cum_trade_qty += trade_qty
        self._cum_trade_amt += trade_qty * trade_price
        self._last_price = trade_price

        bid_order_id = get_field("bid_order_id")
        if bid_order_id is not None and not pd.isna(bid_order_id):
            order_id_int = int(bid_order_id)
            order_state = self._orders.get(order_id_int)
            if not order_state:
                self._log(
                    f"Trade references unknown order_id={order_id_int} "
                    f"(side=1); cannot locate level"
                )
            else:
                if order_state.side != 1:
                    self._log(
                        f"Warning: order_id={order_id_int} side mismatch: "
                        f"state={order_state.side} expected=1"
                    )
                fill_qty = min(trade_qty, order_state.qty)
                order_state.qty -= fill_qty
                if order_state.price is None and not order_state.is_limit:
                    order_state.price = trade_price
                    order_state.is_limit = True
                    if order_state.qty > 0:
                        self._update_level(self._bids, order_state.price, order_state.qty)
                else:
                    if order_state.price is not None:
                        self._update_level(self._bids, order_state.price, -fill_qty)
                if order_state.qty <= 0:
                    self._orders.pop(order_id_int, None)

        ask_order_id = get_field("ask_order_id")
        if ask_order_id is not None and not pd.isna(ask_order_id):
            order_id_int = int(ask_order_id)
            order_state = self._orders.get(order_id_int)
            if not order_state:
                self._log(
                    f"Trade references unknown order_id={order_id_int} "
                    f"(side=2); cannot locate level"
                )
            else:
                if order_state.side != 2:
                    self._log(
                        f"Warning: order_id={order_id_int} side mismatch: "
                        f"state={order_state.side} expected=2"
                    )
                fill_qty = min(trade_qty, order_state.qty)
                order_state.qty -= fill_qty
                if order_state.price is None and not order_state.is_limit:
                    order_state.price = trade_price
                    order_state.is_limit = True
                    if order_state.qty > 0:
                        self._update_level(self._asks, order_state.price, order_state.qty)
                else:
                    if order_state.price is not None:
                        self._update_level(self._asks, order_state.price, -fill_qty)
                if order_state.qty <= 0:
                    self._orders.pop(order_id_int, None)
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
        def get_field(name: str, default: object = None) -> object:
            if hasattr(row, "get"):
                return row.get(name, default)
            return getattr(row, name, default)

        limit_side = None
        if hasattr(row, "get") and "limit_side" in row:
            limit_side = row.get("limit_side")
        elif self._limit_state_detector is not None:
            limit_side = self._limit_state_detector(row)

        snapshot: Dict[str, object] = {
            "skey": get_field("skey"),
            "hhmmss_nano": get_field("hhmmss_nano"),
            "yyyymmdd": get_field("yyyymmdd"),
            "obe_seq_num": get_field("obe_seq_num"),
            "cum_trade_tvl": self._cum_trade_qty,
            "cum_trade_amt": self._cum_trade_amt,
            "last": self._last_price,
        }
        msg_seq_num = get_field("msg_seq_num")
        if msg_seq_num is not None:
            snapshot["msg_seq_num"] = msg_seq_num
        # Snapshot-level simulated matching is a projection used only to
        # enforce non-crossing views and does not represent real execution.
        depth = 10
        buffer_depth = 10
        bids = [list(item) for item in self._top_levels(self._bids, depth + buffer_depth, reverse=True)]
        asks = [list(item) for item in self._top_levels(self._asks, depth + buffer_depth, reverse=False)]
        bid_idx = 0
        ask_idx = 0
        while bid_idx < len(bids) and ask_idx < len(asks) and bids[bid_idx][0] >= asks[ask_idx][0]:
            match_qty = min(bids[bid_idx][1], asks[ask_idx][1])
            bids[bid_idx][1] -= match_qty
            asks[ask_idx][1] -= match_qty
            if bids[bid_idx][1] <= 0:
                bid_idx += 1
            if asks[ask_idx][1] <= 0:
                ask_idx += 1
        bids = bids[bid_idx:]
        asks = asks[ask_idx:]
        bids.sort(key=lambda item: item[0], reverse=True)
        asks.sort(key=lambda item: item[0])
        bids = [tuple(item) for item in bids[:depth]]
        asks = [tuple(item) for item in asks[:depth]]
        for idx in range(1, 11):
            ask_price, ask_qty = (asks[idx - 1] if idx <= len(asks) else (None, None))
            bid_price, bid_qty = (bids[idx - 1] if idx <= len(bids) else (None, None))
            if limit_side == "ask":
                ask_price, ask_qty = (np.nan, 0)
            elif limit_side == "bid":
                bid_price, bid_qty = (np.nan, 0)
            snapshot[f"ask_{idx}_opx"] = ask_price
            snapshot[f"ask_{idx}_qty"] = ask_qty
            snapshot[f"bid_{idx}_opx"] = bid_price
            snapshot[f"bid_{idx}_qty"] = bid_qty
        return snapshot

    @staticmethod
    def _within_time_window(row: pd.Series) -> bool:
        if "hhmmss_nano" not in row:
            return True
        value = row.get("hhmmss_nano")
        if pd.isna(value):
            return False
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return False
        return _TIME_WINDOW_START <= value_int <= _TIME_WINDOW_END

    def apply_construction(self, md_order: pd.DataFrame, md_trade: pd.DataFrame) -> pd.DataFrame:
        order_events = md_order.copy()
        order_events["_event_type"] = "order"
        trade_events = md_trade.copy()
        trade_events["_event_type"] = "trade"
        events = pd.concat([order_events, trade_events], ignore_index=True, sort=False)

        for col in ["msg_order_type", "msg_order_flag", "order_side"]:
            if col in events.columns:
                events[col] = pd.to_numeric(events[col], errors="coerce").fillna(-1).astype(int)

        sort_keys = [key for key in ["yyyymmdd", "hhmmss_nano", "obe_seq_num", "msg_seq_num"] if key in events.columns]
        if sort_keys:
            events = events.sort_values(sort_keys, kind="mergesort")

        snapshots: List[Dict[str, object]] = []
        class _RowView:
            __slots__ = ("_row",)

            def __init__(self, row_obj: object) -> None:
                self._row = row_obj

            def __getitem__(self, key: str) -> object:
                return getattr(self._row, key)

            def get(self, key: str, default: object = None) -> object:
                return getattr(self._row, key, default)

            def __contains__(self, key: str) -> bool:
                return hasattr(self._row, key)

        for row in events.itertuples(index=False):
            row_view = _RowView(row)
            if row._event_type == "order":
                msg_type = row.msg_order_type
                if msg_type == 1:
                    self._apply_insert(row_view)
                elif msg_type == 3:
                    self._apply_cancel(row_view)
                else:
                    self._log(f"Unknown order type {msg_type}")
            else:
                self._apply_trade(row_view)
            if self._within_time_window(row_view):
                snapshots.append(self._build_snapshot(row_view))

        return pd.DataFrame(snapshots)


def _reconstruct_group(payload: Tuple[tuple, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    _, order_group, trade_group = payload
    reconstructor = OrderBookRector()
    return reconstructor.apply_construction(order_group, trade_group)


def reconstruct_many_symbols(
    md_order: pd.DataFrame,
    md_trade: pd.DataFrame,
    *,
    n_jobs: int = 4,
    backend: str = "process",
) -> pd.DataFrame:
    """Reconstruct snapshots for multiple symbols concurrently."""
    group_keys = ["skey"]
    if "yyyymmdd" in md_order.columns and "yyyymmdd" in md_trade.columns:
        group_keys.append("yyyymmdd")

    order_groups = {key: group.copy() for key, group in md_order.groupby(group_keys, sort=False)}
    trade_groups = {key: group.copy() for key, group in md_trade.groupby(group_keys, sort=False)}
    all_keys = sorted(set(order_groups.keys()) | set(trade_groups.keys()))

    empty_order = md_order.iloc[:0].copy()
    empty_trade = md_trade.iloc[:0].copy()
    payloads = [
        (key, order_groups.get(key, empty_order), trade_groups.get(key, empty_trade)) for key in all_keys
    ]

    if backend not in {"process", "thread"}:
        raise ValueError("backend must be 'process' or 'thread'")

    if n_jobs <= 1 or len(payloads) <= 1:
        results = [_reconstruct_group(payload) for payload in payloads]
    else:
        if backend == "process":
            executor_cls = concurrent.futures.ProcessPoolExecutor
        else:
            executor_cls = concurrent.futures.ThreadPoolExecutor
        with executor_cls(max_workers=n_jobs) as executor:
            results = list(executor.map(_reconstruct_group, payloads))

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    sort_keys = [
        key
        for key in ["skey", "yyyymmdd", "hhmmss_nano", "obe_seq_num", "msg_seq_num"]
        if key in combined.columns
    ]
    if sort_keys:
        combined = combined.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    return combined

"""Analyze multi-day, multi-stock trade/order/snapshot data.

This script performs per-(symbol, trading_day) analysis and saves tables/figures.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SYMBOL_COLS = ["symbol", "skey"]
DAY_COLS = ["trading_day", "date", "yyyymmdd"]
TIME_COLS = ["local_clock_at_arrival", "hhmmss_nano"]


@dataclass
class Config:
    trade_path: Path
    order_path: Path
    snapshot_path: Path
    out_dir: Path
    output_format: str
    aggressive_window_ns: int
    immediate_window_ns: int


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in SYMBOL_COLS:
        if col in df.columns:
            rename_map[col] = "symbol"
            break
    for col in DAY_COLS:
        if col in df.columns:
            rename_map[col] = "trading_day"
            break
    for col in TIME_COLS:
        if col in df.columns:
            rename_map[col] = "local_clock_at_arrival"
            break
    if "cum_trade_amount" in df.columns:
        rename_map["cum_trade_amount"] = "cum_trade_tvl"
    if "trade_amt" in df.columns:
        rename_map["trade_amt"] = "notional"
    if "trade_tpx" in df.columns:
        rename_map["trade_tpx"] = "price"
    if "trade_tvl" in df.columns:
        rename_map["trade_tvl"] = "qty"
    if "order_opx" in df.columns:
        rename_map["order_opx"] = "price"
    if "order_qty" in df.columns:
        rename_map["order_qty"] = "qty"
    if "order_side" in df.columns:
        rename_map["order_side"] = "side"
    if "order_flag" not in df.columns and "msg_order_flag" in df.columns:
        rename_map["msg_order_flag"] = "order_flag"
    if "cum_trade_tvl" not in df.columns and "cum_trade_qty" in df.columns:
        rename_map["cum_trade_qty"] = "cum_trade_tvl"
    if "best_bid" not in df.columns and "bid_1_opx" in df.columns:
        rename_map["bid_1_opx"] = "best_bid"
    if "best_ask" not in df.columns and "ask_1_opx" in df.columns:
        rename_map["ask_1_opx"] = "best_ask"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_and_normalize(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return _normalize_columns(df)


def split_by_symbol_day(df: pd.DataFrame) -> Iterable[Tuple[Tuple[str, str], pd.DataFrame]]:
    keys = ["symbol", "trading_day"]
    if not all(key in df.columns for key in keys):
        return []
    return df.groupby(keys, sort=False)


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "local_clock_at_arrival" not in df.columns:
        return df
    return df.sort_values("local_clock_at_arrival", kind="mergesort")


def _bucket_notional(total_notional: float) -> str:
    if total_notional < 1e8:
        return "lt_1e8"
    if total_notional <= 1e9:
        return "1e8_1e9"
    return "gt_1e9"


def daily_notional_buckets(trade: pd.DataFrame) -> pd.DataFrame:
    if trade.empty:
        return pd.DataFrame(columns=["symbol", "trading_day", "daily_notional", "notional_bucket"])
    grouped = trade.groupby(["symbol", "trading_day"], sort=False)
    rows = []
    for (symbol, day), group in grouped:
        total_notional = float(group.get("notional", group.get("qty", 0)).sum())
        rows.append(
            {
                "symbol": symbol,
                "trading_day": day,
                "daily_notional": total_notional,
                "notional_bucket": _bucket_notional(total_notional),
            }
        )
    return pd.DataFrame(rows)


def _describe_series(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def part1_trade_density(trade: pd.DataFrame, snapshot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stats_rows = []
    for (symbol, day), snap_group in split_by_symbol_day(snapshot):
        trade_group = trade[(trade["symbol"] == symbol) & (trade["trading_day"] == day)]
        if snap_group.empty or trade_group.empty:
            continue
        snap_sorted = _ensure_sorted(snap_group)
        trade_sorted = _ensure_sorted(trade_group)
        snap_ts = snap_sorted["local_clock_at_arrival"].to_numpy()
        trade_ts = trade_sorted["local_clock_at_arrival"].to_numpy()
        left_idx = np.searchsorted(trade_ts, snap_ts[:-1], side="right")
        right_idx = np.searchsorted(trade_ts, snap_ts[1:], side="left")
        counts = right_idx - left_idx
        daily_notional = float(trade_sorted.get("notional", trade_sorted.get("qty", 0)).sum())
        bucket = _bucket_notional(daily_notional)
        for idx, count in enumerate(counts):
            rows.append(
                {
                    "symbol": symbol,
                    "trading_day": day,
                    "snap_ts_left": int(snap_ts[idx]),
                    "snap_ts_right": int(snap_ts[idx + 1]),
                    "trade_count_between": int(count),
                    "notional_bucket": bucket,
                }
            )
        stats = _describe_series(counts.astype(float))
        stats.update({"symbol": symbol, "trading_day": day, "bucket": bucket})
        stats_rows.append(stats)
    return pd.DataFrame(rows), pd.DataFrame(stats_rows)


def part1_trade_snapshot_delay(trade: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, day), trade_group in split_by_symbol_day(trade):
        snap_group = snapshot[(snapshot["symbol"] == symbol) & (snapshot["trading_day"] == day)]
        if trade_group.empty or snap_group.empty:
            continue
        snap_sorted = snap_group.sort_values(["cum_trade_tvl", "local_clock_at_arrival"], kind="mergesort")
        snap_first = snap_sorted.groupby("cum_trade_tvl", sort=False)["local_clock_at_arrival"].first().reset_index()
        trade_merge = trade_group.merge(snap_first, on="cum_trade_tvl", how="left", suffixes=("", "_snap"))
        trade_merge = trade_merge.dropna(subset=["local_clock_at_arrival_snap"])
        dt_ns = trade_merge["local_clock_at_arrival_snap"] - trade_merge["local_clock_at_arrival"]
        for row in trade_merge.itertuples(index=False):
            rows.append(
                {
                    "symbol": symbol,
                    "trading_day": day,
                    "trade_time": int(row.local_clock_at_arrival),
                    "trade_cum_tvl": row.cum_trade_tvl,
                    "snap_time_first_same_tvl": int(row.local_clock_at_arrival_snap),
                    "dt_ns": int(row.local_clock_at_arrival_snap - row.local_clock_at_arrival),
                }
            )
    return pd.DataFrame(rows)


def classify_snapshot_speed(snapshot: pd.DataFrame, raw_order: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, day), snap_group in split_by_symbol_day(snapshot):
        raw_group = raw_order[(raw_order["symbol"] == symbol) & (raw_order["trading_day"] == day)]
        if snap_group.empty or raw_group.empty:
            continue
        snap_sorted = _ensure_sorted(snap_group)
        raw_sorted = _ensure_sorted(raw_group)
        merged = pd.merge_asof(
            snap_sorted,
            raw_sorted[["local_clock_at_arrival", "cum_trade_tvl"]].rename(
                columns={"cum_trade_tvl": "raw_cum_trade_tvl"}
            ),
            on="local_clock_at_arrival",
            direction="backward",
        )
        faster = merged["cum_trade_tvl"] > merged["raw_cum_trade_tvl"]
        slower = merged["cum_trade_tvl"] < merged["raw_cum_trade_tvl"]
        normal = ~(faster | slower)
        total = len(merged)
        rows.append(
            {
                "symbol": symbol,
                "trading_day": day,
                "faster_pct": float(faster.mean()) * 100 if total else 0.0,
                "slower_pct": float(slower.mean()) * 100 if total else 0.0,
                "normal_pct": float(normal.mean()) * 100 if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def classify_trade_speed(trade: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, day), trade_group in split_by_symbol_day(trade):
        snap_group = snapshot[(snapshot["symbol"] == symbol) & (snapshot["trading_day"] == day)]
        if trade_group.empty or snap_group.empty:
            continue
        trade_sorted = _ensure_sorted(trade_group)
        snap_sorted = _ensure_sorted(snap_group)
        merged = pd.merge_asof(
            trade_sorted,
            snap_sorted[["local_clock_at_arrival", "cum_trade_tvl"]].rename(
                columns={"cum_trade_tvl": "snap_cum_trade_tvl"}
            ),
            on="local_clock_at_arrival",
            direction="backward",
        )
        faster = merged["cum_trade_tvl"] > merged["snap_cum_trade_tvl"]
        slower = ~faster
        total = len(merged)
        rows.append(
            {
                "symbol": symbol,
                "trading_day": day,
                "faster_pct": float(faster.mean()) * 100 if total else 0.0,
                "slower_pct": float(slower.mean()) * 100 if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def detect_aggressive_orders(
    orders: pd.DataFrame,
    trade: pd.DataFrame,
    aggressive_window_ns: int,
    immediate_window_ns: int,
) -> pd.DataFrame:
    rows = []
    for (symbol, day), order_group in split_by_symbol_day(orders):
        trade_group = trade[(trade["symbol"] == symbol) & (trade["trading_day"] == day)]
        if order_group.empty or trade_group.empty:
            continue
        order_sorted = _ensure_sorted(order_group)
        trade_sorted = _ensure_sorted(trade_group)
        if "cum_trade_tvl" not in trade_sorted.columns:
            continue
        trade_times = trade_sorted["local_clock_at_arrival"].to_numpy()
        trade_cum = trade_sorted["cum_trade_tvl"].to_numpy()
        for row in order_sorted.itertuples(index=False):
            if row.order_flag != 2:
                continue
            order_time = row.local_clock_at_arrival
            idx = np.searchsorted(trade_times, order_time, side="right") - 1
            last_cum = trade_cum[idx] if idx >= 0 else 0
            future_idx = np.searchsorted(trade_times, order_time + aggressive_window_ns, side="right") - 1
            future_cum = trade_cum[future_idx] if future_idx >= 0 else last_cum
            if future_cum <= last_cum:
                continue
            first_trade_idx = np.searchsorted(trade_times, order_time, side="left")
            if first_trade_idx < len(trade_times):
                first_trade_cum = trade_cum[first_trade_idx]
            else:
                first_trade_cum = future_cum
            rows.append(
                {
                    "symbol": symbol,
                    "trading_day": day,
                    "order_time": int(order_time),
                    "order_price": getattr(row, "price", np.nan),
                    "order_qty": getattr(row, "qty", np.nan),
                    "order_notional": getattr(row, "notional", np.nan),
                    "first_trade_cum_after_order": first_trade_cum,
                    "immediate_window_ns": immediate_window_ns,
                }
            )
    return pd.DataFrame(rows)


def _time_to_hhmmss(local_ns: int) -> int:
    total_seconds = int(local_ns // 1_000_000_000)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return hh * 10000 + mm * 100 + ss


def _session_from_time(local_ns: int) -> str:
    hhmmss = _time_to_hhmmss(local_ns)
    if 93000 <= hhmmss < 113000:
        return "morning"
    if 130000 <= hhmmss <= 150000:
        return "afternoon"
    return "other"


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.nan
    return float(np.sum(values * weights) / total_weight)


def _mid_price_from_snapshot(snapshot: pd.DataFrame) -> pd.Series:
    if "mid_price" in snapshot.columns:
        return snapshot["mid_price"]
    if "best_bid" in snapshot.columns and "best_ask" in snapshot.columns:
        return (snapshot["best_bid"] + snapshot["best_ask"]) / 2
    return pd.Series(np.nan, index=snapshot.index)


def compute_forward_metrics(
    aggressive_orders: pd.DataFrame,
    snapshot: pd.DataFrame,
    trade: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if aggressive_orders.empty:
        return pd.DataFrame(), pd.DataFrame()

    detailed_rows = []
    agg_rows = []
    for (symbol, day), orders_group in split_by_symbol_day(aggressive_orders):
        snap_group = snapshot[(snapshot["symbol"] == symbol) & (snapshot["trading_day"] == day)]
        trade_group = trade[(trade["symbol"] == symbol) & (trade["trading_day"] == day)]
        if orders_group.empty or snap_group.empty or trade_group.empty:
            continue
        snap_sorted = _ensure_sorted(snap_group)
        snap_sorted = snap_sorted.assign(mid_price=_mid_price_from_snapshot(snap_sorted))
        trade_sorted = _ensure_sorted(trade_group)
        trade_times = trade_sorted["local_clock_at_arrival"].to_numpy()
        trade_notional = trade_sorted.get("notional", trade_sorted.get("qty", pd.Series(0))).to_numpy()
        trade_prices = trade_sorted.get("price", pd.Series(np.nan)).to_numpy()
        group_rows = []
        for row in orders_group.itertuples(index=False):
            order_time = row.order_time
            order_price = row.order_price
            order_notional = row.order_notional
            if pd.isna(order_notional) or order_notional == 0:
                continue
            fwd_snap = pd.merge_asof(
                pd.DataFrame({"local_clock_at_arrival": [order_time + 15_000_000_000]}),
                snap_sorted[["local_clock_at_arrival", "mid_price"]],
                on="local_clock_at_arrival",
                direction="backward",
            )
            mid_price = fwd_snap["mid_price"].iloc[0]
            order_ret = (mid_price - order_price) / order_price if order_price and not pd.isna(mid_price) else np.nan
            immediate_start = order_time - row.immediate_window_ns
            immediate_end = order_time
            after_3s = order_time + 3_000_000_000
            after_6s = order_time + 6_000_000_000
            after_15s = order_time + 15_000_000_000

            immediate_mask = (trade_times >= immediate_start) & (trade_times <= immediate_end)
            bucket_0_3 = (trade_times > order_time) & (trade_times <= after_3s)
            bucket_3_6 = (trade_times > after_3s) & (trade_times <= after_6s)
            bucket_rest = (trade_times > after_6s) & (trade_times <= after_15s)

            def bucket_notional(mask: np.ndarray) -> float:
                return float(np.sum(trade_notional[mask])) if mask.any() else 0.0

            def bucket_vwap(mask: np.ndarray) -> float:
                if not mask.any():
                    return np.nan
                weights = trade_notional[mask]
                if weights.sum() == 0:
                    return np.nan
                return float(np.average(trade_prices[mask], weights=weights))

            immediate_notional = bucket_notional(immediate_mask)
            notional_0_3 = bucket_notional(bucket_0_3)
            notional_3_6 = bucket_notional(bucket_3_6)
            notional_rest = bucket_notional(bucket_rest)

            remaining = order_notional
            fill_immediate = min(immediate_notional, remaining)
            remaining -= fill_immediate
            fill_0_3 = min(notional_0_3, remaining)
            remaining -= fill_0_3
            fill_3_6 = min(notional_3_6, remaining)
            remaining -= fill_3_6
            fill_rest = min(notional_rest, remaining)

            session = _session_from_time(order_time)
            entry = {
                "symbol": symbol,
                "trading_day": day,
                "order_time": order_time,
                "order_price": order_price,
                "order_notional": order_notional,
                "f15s_mid_price": mid_price,
                "f15s_order_return": order_ret,
                "session": session,
                "fill_notional_immediate": fill_immediate,
                "fill_notional_0_3s": fill_0_3,
                "fill_notional_3_6s": fill_3_6,
                "fill_notional_rest": fill_rest,
                "vwap_immediate": bucket_vwap(immediate_mask),
                "vwap_0_3s": bucket_vwap(bucket_0_3),
                "vwap_3_6s": bucket_vwap(bucket_3_6),
                "vwap_rest": bucket_vwap(bucket_rest),
            }
            group_rows.append(entry)
            detailed_rows.append(entry)
        detailed = pd.DataFrame(group_rows)
        if detailed.empty:
            continue
        detailed["notional_group"] = np.where(
            detailed["order_notional"] > 1_000_000,
            ">1e6",
            np.where(detailed["order_notional"] < 5_000, "<5k", "other"),
        )
        bucket_map = {
            "immediate": "fill_notional_immediate",
            "0_3s": "fill_notional_0_3s",
            "3_6s": "fill_notional_3_6s",
            "rest": "fill_notional_rest",
        }
        agg_parts = []
        for bucket, col in bucket_map.items():
            bucket_df = detailed.copy()
            bucket_df["bucket"] = bucket
            bucket_df["fill_rate"] = bucket_df[col] / bucket_df["order_notional"]
            bucket_df["trade_return"] = (
                bucket_df[f"vwap_{bucket}"] - bucket_df["order_price"]
            ) / bucket_df["order_price"]
            agg_parts.append(bucket_df)
        agg_data = pd.concat(agg_parts, ignore_index=True)
        agg = (
            agg_data.groupby(["symbol", "trading_day", "notional_group", "bucket", "session"], sort=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "weighted_fill_rate": weighted_stats(
                            x["fill_rate"].to_numpy(), x["order_notional"].to_numpy()
                        ),
                        "weighted_trade_return": weighted_stats(
                            x["trade_return"].to_numpy(), x["order_notional"].to_numpy()
                        ),
                        "weighted_order_return": weighted_stats(
                            x["f15s_order_return"].to_numpy(), x["order_notional"].to_numpy()
                        ),
                    }
                )
            )
            .reset_index()
        )
        agg_rows.append(agg)
    return pd.DataFrame(detailed_rows), pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()


def save_tables_and_plots(
    out_dir: Path,
    tables: Dict[str, pd.DataFrame],
    figs: List[Tuple[str, plt.Figure]],
    output_format: str,
) -> None:
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        path = tables_dir / f"{name}.{output_format}"
        if output_format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

    for name, fig in figs:
        fig_path = figs_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_histogram(values: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.hist(values, bins=50)
    ax.set_title(title)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trade/order/snapshot datasets.")
    parser.add_argument("--trade", required=True, help="Path to trade parquet/csv")
    parser.add_argument("--order", required=True, help="Path to order parquet/csv")
    parser.add_argument("--snapshot", required=True, help="Path to snapshot parquet/csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="Output format")
    parser.add_argument("--aggr_window_ms", type=float, default=50.0, help="Aggressive window in ms")
    parser.add_argument("--immediate_window_us", type=float, default=1000.0, help="Immediate window in us")
    args = parser.parse_args()

    config = Config(
        trade_path=Path(args.trade),
        order_path=Path(args.order),
        snapshot_path=Path(args.snapshot),
        out_dir=Path(args.out_dir),
        output_format=args.format,
        aggressive_window_ns=int(args.aggr_window_ms * 1_000_000),
        immediate_window_ns=int(args.immediate_window_us * 1_000),
    )

    trade = load_and_normalize(config.trade_path)
    orders = load_and_normalize(config.order_path)
    snapshot = load_and_normalize(config.snapshot_path)

    tables: Dict[str, pd.DataFrame] = {}
    figures: List[Tuple[str, plt.Figure]] = []

    trade_between, trade_between_stats = part1_trade_density(trade, snapshot)
    tables["part1_trade_between_snapshots"] = trade_between
    tables["part1_trade_between_stats"] = trade_between_stats

    daily_buckets = daily_notional_buckets(trade)
    if not trade_between.empty:
        for bucket, group in trade_between.groupby("notional_bucket", sort=False):
            values = group["trade_count_between"].to_numpy()
            figures.append(
                (f"part1_trade_between_hist_{bucket}", _plot_histogram(values, f"Trade count between snapshots ({bucket})"))
            )
            stats = _describe_series(values.astype(float))
            print(f"Trade count stats {bucket}: {stats}")

    trade_delay = part1_trade_snapshot_delay(trade, snapshot)
    if not trade_delay.empty and not daily_buckets.empty:
        trade_delay = trade_delay.merge(daily_buckets, on=["symbol", "trading_day"], how="left")
    tables["part1_trade_to_snapshot_delay"] = trade_delay
    if not trade_delay.empty:
        for bucket, group in trade_delay.groupby("notional_bucket", sort=False):
            values = group["dt_ns"].to_numpy()
            figures.append(
                (f"part1_trade_delay_hist_{bucket}", _plot_histogram(values, f"Trade to snapshot delay ({bucket})"))
            )
            stats = _describe_series(values.astype(float))
            print(f"Trade delay stats {bucket}: {stats}")

    snapshot_speed = classify_snapshot_speed(snapshot, orders)
    trade_speed = classify_trade_speed(trade, snapshot)
    tables["part3_snapshot_speed_summary"] = snapshot_speed
    tables["part3_trade_speed_summary"] = trade_speed
    if not snapshot_speed.empty:
        fig, ax = plt.subplots()
        avg_vals = snapshot_speed[["faster_pct", "slower_pct", "normal_pct"]].mean()
        ax.bar(avg_vals.index, avg_vals.values)
        ax.set_title("Snapshot speed class proportions")
        figures.append(("part3_snapshot_speed_bar", fig))
    if not trade_speed.empty:
        fig, ax = plt.subplots()
        avg_vals = trade_speed[["faster_pct", "slower_pct"]].mean()
        ax.bar(avg_vals.index, avg_vals.values)
        ax.set_title("Trade speed class proportions")
        figures.append(("part3_trade_speed_bar", fig))

    aggressive_orders = detect_aggressive_orders(
        orders,
        trade,
        aggressive_window_ns=config.aggressive_window_ns,
        immediate_window_ns=config.immediate_window_ns,
    )
    detailed, agg = compute_forward_metrics(aggressive_orders, snapshot, trade)
    tables["part3_aggressive_order_metrics_detailed"] = detailed
    tables["part3_aggressive_order_metrics_agg"] = agg
    if not detailed.empty:
        for group, group_df in detailed.groupby("notional_group", sort=False):
            values = group_df["f15s_order_return"].dropna().to_numpy()
            if values.size:
                figures.append(
                    (f"part3_f15s_order_return_hist_{group}", _plot_histogram(values, f"F15s order return ({group})"))
                )
    if not agg.empty:
        for group, group_df in agg.groupby("notional_group", sort=False):
            fig, ax = plt.subplots()
            bucket_order = ["immediate", "0_3s", "3_6s", "rest"]
            bucket_vals = []
            for bucket in bucket_order:
                subset = group_df[group_df["bucket"] == bucket]
                bucket_vals.append(float(subset["weighted_fill_rate"].mean()) if not subset.empty else 0.0)
            ax.bar(bucket_order, bucket_vals)
            ax.set_title(f"Weighted fill rate by bucket ({group})")
            figures.append((f"part3_fill_rate_bar_{group}", fig))
            trade_vals = group_df["weighted_trade_return"].dropna().to_numpy()
            if trade_vals.size:
                figures.append(
                    (f"part3_trade_return_hist_{group}", _plot_histogram(trade_vals, f"Trade return ({group})"))
                )

    save_tables_and_plots(config.out_dir, tables, figures, config.output_format)

    print("Summary")
    print(f"Trade-between rows: {len(trade_between)}")
    print(f"Trade-delay rows: {len(trade_delay)}")
    print(f"Snapshot speed rows: {len(snapshot_speed)}")
    print(f"Trade speed rows: {len(trade_speed)}")
    print(f"Aggressive orders rows: {len(aggressive_orders)}")


if __name__ == "__main__":
    main()

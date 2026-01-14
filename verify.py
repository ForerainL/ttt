"""Utilities to validate reconstructed snapshots against official snapshots.

Example:
    >>> import pandas as pd
    >>> from verify import verify_official_is_subset
    >>> result = verify_official_is_subset(gen_snap, official_snap)
    >>> print(result["ok"], result["summary"])
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


DEFAULT_KEY_COLS = ["skey", "yyyymmdd", "hhmmss_nano", "obe_seq_num"]
DEFAULT_STATE_COLS: List[str] = [
    "last",
    "cum_trade_tvl",
    "cum_trade_amt",
]
DEFAULT_STATE_COLS.extend([f"ask_{idx}_opx" for idx in range(1, 11)])
DEFAULT_STATE_COLS.extend([f"ask_{idx}_qty" for idx in range(1, 11)])
DEFAULT_STATE_COLS.extend([f"bid_{idx}_opx" for idx in range(1, 11)])
DEFAULT_STATE_COLS.extend([f"bid_{idx}_qty" for idx in range(1, 11)])


def _intersection(cols: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [col for col in cols if col in df.columns]


def _build_time_key(df: pd.DataFrame) -> pd.Series:
    if "yyyymmdd" not in df.columns or "hhmmss_nano" not in df.columns:
        raise ValueError("yyyymmdd and hhmmss_nano are required for time-based matching")
    yyyymmdd = pd.to_numeric(df["yyyymmdd"], errors="coerce")
    hhmmss_nano = pd.to_numeric(df["hhmmss_nano"], errors="coerce")
    if yyyymmdd.isna().any() or hhmmss_nano.isna().any():
        raise ValueError("Invalid time columns: yyyymmdd or hhmmss_nano contains NaN")
    return yyyymmdd.astype(np.int64) * 1_000_000_000 + hhmmss_nano.astype(np.int64)


def _numeric_tolerance_match(series_off: pd.Series, series_gen: pd.Series, tolerance: float) -> pd.Series:
    diff = (series_off - series_gen).abs()
    return diff.le(tolerance) | (series_off.isna() & series_gen.isna())


def _exact_match(series_off: pd.Series, series_gen: pd.Series) -> pd.Series:
    return series_off.eq(series_gen) | (series_off.isna() & series_gen.isna())


def _top_differences(row: pd.Series, state_cols: List[str]) -> List[str]:
    diffs: List[tuple[str, float]] = []
    for col in state_cols:
        off = row.get(f"{col}_off")
        gen = row.get(f"{col}_gen")
        if pd.isna(off) and pd.isna(gen):
            continue
        if off == gen:
            continue
        if isinstance(off, (int, float, np.number)) and isinstance(gen, (int, float, np.number)):
            diffs.append((col, float(abs(off - gen))))
        else:
            diffs.append((col, float("inf")))
    diffs.sort(key=lambda item: item[1], reverse=True)
    return [col for col, _ in diffs[:3]]


def verify_official_is_subset(
    gen_snap: pd.DataFrame,
    official_snap: pd.DataFrame,
    *,
    key_cols: list[str] | None = None,
    state_cols: list[str] | None = None,
    tolerance: float = 0.0,
    require_monotonic: bool = True,
    time_match: str = "le",
) -> dict:
    """Verify official snapshots are a subsequence of generated snapshots.

    Matching logic:
    - Use key columns when available for exact key matching.
    - If no key columns are available, match by time using equality or latest <= time.
    - Compare state columns exactly or within numeric tolerance.
    - Optionally require strictly increasing matched gen indices.
    """

    key_cols = _intersection(key_cols or DEFAULT_KEY_COLS, official_snap)
    key_cols = [col for col in key_cols if col in gen_snap.columns]
    state_cols = _intersection(state_cols or DEFAULT_STATE_COLS, official_snap)
    state_cols = [col for col in state_cols if col in gen_snap.columns]

    if time_match not in {"le", "eq"}:
        raise ValueError("time_match must be 'le' or 'eq'")

    gen_data = gen_snap.copy(deep=False).reset_index(drop=True)
    off_data = official_snap.copy(deep=False).reset_index(drop=True)
    gen_data = gen_data.assign(gen_idx=np.arange(len(gen_data)))
    off_data = off_data.assign(off_idx=np.arange(len(off_data)))

    if key_cols:
        gen_keys = gen_data[key_cols + ["gen_idx"]].drop_duplicates(subset=key_cols, keep="first")
        gen_states = gen_data[state_cols].add_suffix("_gen")
        gen_merged = pd.concat([gen_keys.reset_index(drop=True), gen_states], axis=1)

        off_keys = off_data[key_cols + ["off_idx"]]
        off_states = off_data[state_cols].add_suffix("_off")
        off_merged = pd.concat([off_keys, off_states], axis=1)

        merged = off_merged.merge(gen_merged, on=key_cols, how="left", sort=False)
        merged = merged.sort_values("off_idx", kind="mergesort")
    else:
        gen_data = gen_data.assign(time_key=_build_time_key(gen_data))
        off_data = off_data.assign(time_key=_build_time_key(off_data))

        gen_states = gen_data[state_cols].add_suffix("_gen")
        off_states = off_data[state_cols].add_suffix("_off")
        gen_merge_base = pd.concat([gen_data[["gen_idx", "time_key"]], gen_states], axis=1)
        off_merge_base = pd.concat([off_data[["off_idx", "time_key"]], off_states], axis=1)

        gen_merge_base = gen_merge_base.sort_values("time_key", kind="mergesort")
        off_merge_base = off_merge_base.sort_values("time_key", kind="mergesort")

        if time_match == "eq":
            merged = off_merge_base.merge(gen_merge_base, on="time_key", how="left", sort=False)
        else:
            merged = pd.merge_asof(
                off_merge_base,
                gen_merge_base,
                on="time_key",
                direction="backward",
            )
        merged = merged.sort_values("off_idx", kind="mergesort")

    candidate_mask = merged["gen_idx"].notna()

    state_match = pd.Series(True, index=merged.index)
    for col in state_cols:
        off_col = merged[f"{col}_off"]
        gen_col = merged[f"{col}_gen"]
        if tolerance > 0 and pd.api.types.is_numeric_dtype(off_col):
            state_match &= _numeric_tolerance_match(off_col, gen_col, tolerance)
        else:
            state_match &= _exact_match(off_col, gen_col)

    matched_mask = candidate_mask & state_match

    monotonic_ok = pd.Series(True, index=merged.index)
    if require_monotonic:
        last_idx = -1
        for idx, row in merged.iterrows():
            if not matched_mask.at[idx]:
                continue
            gen_idx = int(row["gen_idx"])
            if gen_idx <= last_idx:
                monotonic_ok.at[idx] = False
            else:
                last_idx = gen_idx

    ok_mask = matched_mask & monotonic_ok
    n_official = len(off_data)
    n_matched = int(ok_mask.sum())
    n_mismatch = n_official - n_matched

    mismatch_rows = []

    if not ok_mask.all():
        gen_time_sorted = None
        if "time_key" in gen_data.columns:
            gen_time_sorted = gen_data.sort_values("time_key", kind="mergesort")["time_key"].to_numpy()

        for idx, row in merged.iterrows():
            if ok_mask.at[idx]:
                continue
            reason = "state_mismatch"
            if not candidate_mask.at[idx]:
                reason = "no_candidate"
            elif require_monotonic and not monotonic_ok.at[idx]:
                reason = "non_monotonic"
            mismatch = {col: row.get(col) for col in key_cols if col in row}
            mismatch["reason"] = reason
            if reason != "no_candidate":
                mismatch["top_diff_cols"] = _top_differences(row, state_cols)
                mismatch["candidate_time_key"] = row.get("time_key")
            elif gen_time_sorted is not None:
                off_time = row.get("time_key")
                if off_time is not None:
                    pos = np.searchsorted(gen_time_sorted, off_time)
                    closest_idx = min(max(pos, 0), len(gen_time_sorted) - 1)
                    mismatch["closest_time_key"] = gen_time_sorted[closest_idx]
            mismatch_rows.append(mismatch)

    summary = (
        f"Matched {n_matched}/{n_official} official snapshots; "
        f"mismatches={n_mismatch}."
    )

    return {
        "ok": n_mismatch == 0,
        "n_official": n_official,
        "n_matched": n_matched,
        "n_mismatch": n_mismatch,
        "mismatch_rows": mismatch_rows,
        "summary": summary,
    }

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .utils import (
        build_expected_index,
        clip,
        prepare_dataframe,
        resolve_frequency,
        rolling_window_size,
        safe_div,
        safe_float,
    )
except ImportError:
    from utils import (
        build_expected_index,
        clip,
        prepare_dataframe,
        resolve_frequency,
        rolling_window_size,
        safe_div,
        safe_float,
    )


def analyze_integrity(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    freq: str | None = None,
) -> dict:
    clean = prepare_dataframe(df=df, date_col=date_col, target_col=target_col)
    total_rows = len(clean)
    duplicate_count = int(clean[date_col].duplicated().sum())
    duplicate_ratio = safe_div(duplicate_count, total_rows, default=0.0)

    series = clean.groupby(date_col)[target_col].sum().sort_index().astype(float)
    freq_alias, freq_seconds = resolve_frequency(series.index, freq=freq)
    expected_index = build_expected_index(series.index, freq_alias=freq_alias, freq_seconds=freq_seconds)

    missing_count = max(len(expected_index) - len(series), 0)
    missing_timestamp_ratio = safe_div(missing_count, max(len(expected_index), 1), default=0.0)

    if len(series) > 1:
        deltas = series.index.to_series().diff().dropna().dt.total_seconds()
        modal_delta = safe_float(deltas.mode().iloc[0] if not deltas.mode().empty else deltas.median())
        tolerance = max(1.0, 0.1 * abs(modal_delta))
        irregular = (np.abs(deltas - modal_delta) > tolerance).astype(float)
        gap_ratio = safe_float(irregular.mean(), default=0.0)
    else:
        gap_ratio = 0.0

    values = series.values
    zero_ratio = safe_float(np.mean(values == 0), default=0.0)
    negative_ratio = safe_float(np.mean(values < 0), default=0.0)

    if len(values) >= 4:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_ratio_iqr = safe_float(np.mean((values < lower) | (values > upper)), default=0.0)
    else:
        outlier_ratio_iqr = 0.0

    median = np.median(values) if len(values) else 0.0
    mad = np.median(np.abs(values - median)) if len(values) else 0.0
    if mad > 0:
        robust_z = 0.6745 * (values - median) / (mad + 1e-9)
        outlier_ratio_mad = safe_float(np.mean(np.abs(robust_z) > 3.5), default=0.0)
    else:
        outlier_ratio_mad = 0.0

    window = rolling_window_size(len(series), min_size=4, frac=0.12, max_size=48)
    roll_mean = series.rolling(window=window).mean().dropna()
    mean_shift = roll_mean.diff().abs().dropna()
    if len(mean_shift) > 3:
        threshold = safe_float(mean_shift.median() + (2.0 * mean_shift.std(ddof=0)))
        break_count = int((mean_shift > threshold).sum())
        shift_scale = abs(series.mean()) + 1e-9
        rolling_mean_shift_max = safe_float(mean_shift.max() / shift_scale, default=0.0)
    else:
        break_count = 0
        rolling_mean_shift_max = 0.0

    if len(series) > 1:
        coverage_duration_days = safe_float(
            (series.index.max() - series.index.min()).total_seconds() / 86400.0,
            default=0.0,
        )
    else:
        coverage_duration_days = 0.0

    penalty = (
        (0.30 * missing_timestamp_ratio)
        + (0.18 * duplicate_ratio)
        + (0.12 * gap_ratio)
        + (0.16 * outlier_ratio_iqr)
        + (0.10 * outlier_ratio_mad)
        + (0.08 * min(1.0, safe_div(break_count, max(len(series), 1), default=0.0) * 20))
        + (0.06 * negative_ratio)
    )
    integrity_score = clip(100.0 * (1.0 - penalty), low=0.0, high=100.0)

    if integrity_score >= 85:
        integrity_classification = "Healthy"
    elif integrity_score >= 65:
        integrity_classification = "Monitor"
    else:
        integrity_classification = "Critical"

    return {
        "row_count": float(total_rows),
        "observation_count": float(len(series)),
        "missing_timestamp_ratio": float(missing_timestamp_ratio),
        "duplicate_timestamp_ratio": float(duplicate_ratio),
        "gap_ratio": float(gap_ratio),
        "zero_ratio": float(zero_ratio),
        "negative_ratio": float(negative_ratio),
        "outlier_ratio_iqr": float(outlier_ratio_iqr),
        "outlier_ratio_mad": float(outlier_ratio_mad),
        "rolling_mean_shift_max": float(rolling_mean_shift_max),
        "structural_break_count": float(break_count),
        "coverage_duration_days": float(coverage_duration_days),
        "frequency_seconds": float(freq_seconds),
        "integrity_score": float(integrity_score),
        "integrity_classification": integrity_classification,
    }

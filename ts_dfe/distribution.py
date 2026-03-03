from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .utils import clip, clip01, safe_div, safe_float
except ImportError:
    from utils import clip, clip01, safe_div, safe_float


def analyze_distribution(series: pd.Series) -> dict:
    s = pd.Series(series, dtype=float).dropna()
    if len(s) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "p99_median_ratio": np.nan,
            "tail_heaviness_index": np.nan,
            "spike_frequency_95": np.nan,
            "symmetry_score": np.nan,
            "zero_inflation_ratio": np.nan,
            "distribution_risk_score": np.nan,
            "is_gaussian_like": 0.0,
            "is_heavy_tailed": 0.0,
            "is_zero_inflated": 0.0,
            "is_skewed_transactional": 0.0,
            "distribution_classification": "Insufficient data",
        }

    mean_v = safe_float(s.mean())
    std_v = safe_float(s.std(ddof=0))
    cv_v = safe_div(std_v, abs(mean_v), default=np.nan)
    skew_v = safe_float(s.skew(), default=0.0)
    kurt_v = safe_float(s.kurt(), default=0.0)

    p50 = safe_float(s.quantile(0.50))
    p75 = safe_float(s.quantile(0.75))
    p95 = safe_float(s.quantile(0.95))
    p99 = safe_float(s.quantile(0.99))

    p99_median_ratio = safe_div(abs(p99), abs(p50), default=np.nan)
    upper_tail = p99 - p95
    core_span = p75 - p50
    tail_heaviness_index = safe_div(upper_tail, core_span, default=np.nan)
    spike_frequency = safe_float(np.mean(s.values > p95), default=0.0)
    zero_inflation_ratio = safe_float(np.mean(s.values == 0), default=0.0)
    symmetry_score = clip01(1.0 - (abs(skew_v) / 2.0))

    heavy_tail_flag = float((kurt_v > 3.0) or (p99_median_ratio > 8.0) or (tail_heaviness_index > 3.0))
    zero_inflated_flag = float(zero_inflation_ratio > 0.30)
    skewed_flag = float(abs(skew_v) > 1.0)
    gaussian_like_flag = float((heavy_tail_flag == 0.0) and (zero_inflated_flag == 0.0) and (skewed_flag == 0.0))

    if zero_inflated_flag == 1.0:
        classification = "Zero-inflated"
    elif heavy_tail_flag == 1.0:
        classification = "Heavy-tailed"
    elif skewed_flag == 1.0:
        classification = "Skewed transactional"
    else:
        classification = "Gaussian-like"

    risk_score = 100.0 * (
        (0.32 * clip01(abs(skew_v) / 3.0))
        + (0.30 * clip01(max(0.0, kurt_v) / 10.0))
        + (0.18 * clip01(safe_div(p99_median_ratio, 12.0, default=0.0)))
        + (0.20 * clip01(zero_inflation_ratio / 0.50))
    )
    risk_score = clip(risk_score, low=0.0, high=100.0)

    return {
        "mean": float(mean_v),
        "std": float(std_v),
        "cv": float(cv_v),
        "skewness": float(skew_v),
        "kurtosis": float(kurt_v),
        "p99_median_ratio": float(p99_median_ratio),
        "tail_heaviness_index": float(tail_heaviness_index),
        "spike_frequency_95": float(spike_frequency),
        "symmetry_score": float(symmetry_score),
        "zero_inflation_ratio": float(zero_inflation_ratio),
        "distribution_risk_score": float(risk_score),
        "is_gaussian_like": float(gaussian_like_flag),
        "is_heavy_tailed": float(heavy_tail_flag),
        "is_zero_inflated": float(zero_inflated_flag),
        "is_skewed_transactional": float(skewed_flag),
        "distribution_classification": classification,
    }

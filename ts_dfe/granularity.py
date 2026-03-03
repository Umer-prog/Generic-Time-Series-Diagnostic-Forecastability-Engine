from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .utils import (
        ar_mae,
        clip,
        clip01,
        manual_acf,
        mean_forecast_mae,
        naive_mae,
        safe_div,
        safe_float,
    )
except ImportError:
    from utils import (
        ar_mae,
        clip,
        clip01,
        manual_acf,
        mean_forecast_mae,
        naive_mae,
        safe_div,
        safe_float,
    )


def _metrics_for_series(series: pd.Series) -> dict:
    s = pd.Series(series, dtype=float).dropna()
    if len(s) < 6:
        return {
            "count": float(len(s)),
            "cv": np.nan,
            "acf1": np.nan,
            "naive_mae": np.nan,
            "ar1_mae": np.nan,
            "mean_forecast_mae": np.nan,
            "model_improvement_ratio": np.nan,
            "granularity_score": np.nan,
        }

    mean_v = safe_float(s.mean())
    std_v = safe_float(s.std(ddof=0))
    cv_v = safe_div(std_v, abs(mean_v), default=np.nan)
    acf1 = safe_float(manual_acf(s, max_lag=1).get(1, np.nan), default=np.nan)
    naive_err = naive_mae(s)
    ar1_err = ar_mae(s, lags=1)
    mean_err = mean_forecast_mae(s)
    model_improvement = safe_div(naive_err - ar1_err, naive_err, default=np.nan)

    granularity_score = 100.0 * (
        (0.35 * clip01(max(acf1, 0.0)))
        + (0.30 * clip01(1.0 / (1.0 + abs(safe_float(cv_v, default=3.0)))))
        + (0.35 * clip01(max(model_improvement, 0.0)))
    )
    granularity_score = clip(granularity_score, low=0.0, high=100.0)

    return {
        "count": float(len(s)),
        "cv": float(cv_v),
        "acf1": float(acf1),
        "naive_mae": float(naive_err),
        "ar1_mae": float(ar1_err),
        "mean_forecast_mae": float(mean_err),
        "model_improvement_ratio": float(model_improvement),
        "granularity_score": float(granularity_score),
    }


def analyze_granularity(regular_series: pd.Series) -> dict:
    base = pd.Series(regular_series, dtype=float)
    base = base.sort_index()
    weekly = base.resample("W").sum(min_count=1)
    monthly = base.resample("ME").sum(min_count=1)

    original_metrics = _metrics_for_series(base)
    weekly_metrics = _metrics_for_series(weekly)
    monthly_metrics = _metrics_for_series(monthly)

    candidates = {
        "original": original_metrics,
        "weekly": weekly_metrics,
        "monthly": monthly_metrics,
    }
    best_granularity = "original"
    best_score = -np.inf
    for name, metrics in candidates.items():
        score = safe_float(metrics.get("granularity_score", np.nan), default=-np.inf)
        if score > best_score:
            best_score = score
            best_granularity = name

    best_metrics = candidates[best_granularity]
    signal_gain_ratio = safe_div(
        abs(safe_float(best_metrics.get("acf1", np.nan), default=np.nan)),
        abs(safe_float(original_metrics.get("acf1", np.nan), default=np.nan)),
        default=np.nan,
    )
    noise_reduction_ratio = safe_div(
        safe_float(original_metrics.get("cv", np.nan), default=np.nan),
        safe_float(best_metrics.get("cv", np.nan), default=np.nan),
        default=np.nan,
    )
    model_improvement_ratio = safe_float(best_metrics.get("model_improvement_ratio", np.nan), default=np.nan)

    return {
        "original": original_metrics,
        "weekly": weekly_metrics,
        "monthly": monthly_metrics,
        "signal_gain_ratio": float(signal_gain_ratio),
        "noise_reduction_ratio": float(noise_reduction_ratio),
        "model_improvement_ratio": float(model_improvement_ratio),
        "optimal_granularity": best_granularity,
        "optimal_granularity_score": float(best_score),
    }

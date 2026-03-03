from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import adfuller, kpss

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    InterpolationWarning = Warning

try:
    from .utils import clip, clip01, rolling_window_size, safe_float, trend_r2
except ImportError:
    from utils import clip, clip01, rolling_window_size, safe_float, trend_r2


def analyze_stationarity(series: pd.Series) -> dict:
    s = pd.Series(series, dtype=float).dropna()
    n = len(s)
    if n < 8:
        return {
            "observation_count": float(n),
            "adf_pvalue": np.nan,
            "kpss_pvalue": np.nan,
            "rolling_mean_drift": np.nan,
            "rolling_variance_drift": np.nan,
            "cusum_max_abs": np.nan,
            "trend_strength_r2": np.nan,
            "stability_score": np.nan,
            "stationarity_classification": "Insufficient data",
        }

    adf_pvalue = np.nan
    kpss_pvalue = np.nan

    if HAS_STATSMODELS and n >= 20:
        try:
            adf_pvalue = safe_float(adfuller(s.values, autolag="AIC")[1], default=np.nan)
        except Exception:
            adf_pvalue = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InterpolationWarning)
                kpss_pvalue = safe_float(
                    kpss(s.values, regression="c", nlags="auto")[1],
                    default=np.nan,
                )
        except Exception:
            kpss_pvalue = np.nan

    window = rolling_window_size(n, min_size=5, frac=0.20, max_size=52)
    roll_mean = s.rolling(window=window).mean().dropna()
    roll_var = s.rolling(window=window).var().dropna()

    if len(roll_mean) > 1:
        rolling_mean_drift = safe_float(abs(roll_mean.iloc[-1] - roll_mean.iloc[0]) / (abs(s.mean()) + 1e-9))
    else:
        rolling_mean_drift = np.nan

    if len(roll_var) > 1:
        rolling_variance_drift = safe_float(abs(roll_var.iloc[-1] - roll_var.iloc[0]) / (roll_var.mean() + 1e-9))
    else:
        rolling_variance_drift = np.nan

    z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    cusum = np.cumsum(z.values)
    cusum_max_abs = safe_float(np.max(np.abs(cusum)) / np.sqrt(max(n, 1)))

    trend_strength = trend_r2(s)

    adf_penalty = 0.0 if (np.isfinite(adf_pvalue) and adf_pvalue < 0.05) else 0.20
    kpss_penalty = 0.0 if (np.isfinite(kpss_pvalue) and kpss_pvalue > 0.05) else 0.20
    drift_penalty = 0.24 * clip01(safe_float(rolling_mean_drift, default=1.0))
    variance_penalty = 0.16 * clip01(safe_float(rolling_variance_drift, default=1.0))
    cusum_penalty = 0.10 * clip01(safe_float(cusum_max_abs, default=2.0) / 2.5)
    trend_penalty = 0.10 * clip01(safe_float(trend_strength, default=1.0))

    stability_score = 100.0 * (
        1.0
        - adf_penalty
        - kpss_penalty
        - drift_penalty
        - variance_penalty
        - cusum_penalty
        - trend_penalty
    )
    stability_score = clip(stability_score, low=0.0, high=100.0)

    stationary_rule = (
        np.isfinite(adf_pvalue)
        and (adf_pvalue < 0.05)
        and np.isfinite(kpss_pvalue)
        and (kpss_pvalue > 0.05)
        and safe_float(trend_strength, default=1.0) < 0.20
        and safe_float(cusum_max_abs, default=5.0) < 1.50
    )
    trend_rule = safe_float(trend_strength, default=0.0) >= 0.40
    regime_rule = (
        safe_float(cusum_max_abs, default=0.0) >= 1.80
        or safe_float(rolling_mean_drift, default=0.0) >= 0.45
    )

    if stationary_rule:
        stationarity_classification = "Stationary"
    elif trend_rule:
        stationarity_classification = "Trend-dominated"
    elif regime_rule:
        stationarity_classification = "Regime-shifting"
    else:
        stationarity_classification = "Drift-prone"

    return {
        "observation_count": float(n),
        "adf_pvalue": float(adf_pvalue),
        "kpss_pvalue": float(kpss_pvalue),
        "rolling_mean_drift": float(rolling_mean_drift),
        "rolling_variance_drift": float(rolling_variance_drift),
        "cusum_max_abs": float(cusum_max_abs),
        "trend_strength_r2": float(trend_strength),
        "stability_score": float(stability_score),
        "stationarity_classification": stationarity_classification,
    }

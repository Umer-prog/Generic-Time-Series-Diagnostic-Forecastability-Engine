from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.diagnostic import het_arch

    HAS_ARCH_TEST = True
except Exception:
    HAS_ARCH_TEST = False

try:
    from .utils import clip, clip01, manual_acf, rolling_window_size, safe_float
except ImportError:
    from utils import clip, clip01, manual_acf, rolling_window_size, safe_float


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    frame = pd.concat([sx, sy], axis=1).dropna()
    if len(frame) < 3:
        return np.nan
    xv = frame.iloc[:, 0].values
    yv = frame.iloc[:, 1].values
    if np.std(xv, ddof=0) < 1e-12 or np.std(yv, ddof=0) < 1e-12:
        return np.nan
    with np.errstate(all="ignore"):
        corr = np.corrcoef(xv, yv)[0, 1]
    if not np.isfinite(corr):
        return np.nan
    return float(corr)


def analyze_volatility(series: pd.Series) -> dict:
    s = pd.Series(series, dtype=float).dropna()
    n = len(s)
    if n < 8:
        return {
            "observation_count": float(n),
            "rolling_std_mean_corr": np.nan,
            "arch_test_pvalue": np.nan,
            "volatility_clustering_index": np.nan,
            "event_volatility_index": np.nan,
            "volatility_risk_score": np.nan,
            "volatility_classification": "Insufficient data",
        }

    changes = s.diff().dropna()
    window = rolling_window_size(len(changes), min_size=5, frac=0.18, max_size=40)

    roll_std = changes.rolling(window=window).std()
    roll_mean_abs = changes.abs().rolling(window=window).mean()
    rolling_std_mean_corr = safe_float(_safe_corr(roll_std, roll_mean_abs), default=np.nan)

    if HAS_ARCH_TEST and len(changes) >= 20:
        try:
            # LM p-value is at index 1.
            arch_test_pvalue = safe_float(het_arch(changes.values, nlags=min(10, len(changes) // 4))[1], default=np.nan)
        except Exception:
            arch_test_pvalue = np.nan
    else:
        arch_test_pvalue = np.nan

    squared = changes**2
    acf_sq = manual_acf(squared, max_lag=1)
    volatility_clustering_index = safe_float(acf_sq.get(1, np.nan), default=np.nan)

    abs_changes = np.abs(changes.values)
    if len(abs_changes) >= 5:
        p95 = np.percentile(abs_changes, 95)
        p50 = np.percentile(abs_changes, 50)
        event_volatility_index = safe_float(p95 / (p50 + 1e-9))
    else:
        event_volatility_index = np.nan

    if np.isfinite(event_volatility_index) and event_volatility_index > 4.0:
        classification = "Event volatility"
    elif np.isfinite(arch_test_pvalue) and (arch_test_pvalue < 0.05) and (safe_float(volatility_clustering_index, 0.0) > 0.25):
        classification = "Clustered volatility"
    elif (
        (np.isfinite(arch_test_pvalue) and arch_test_pvalue < 0.05)
        or (np.isfinite(rolling_std_mean_corr) and abs(rolling_std_mean_corr) > 0.35)
    ):
        classification = "Heteroskedastic"
    else:
        classification = "Homoskedastic"

    arch_effect_flag = 1.0 if (np.isfinite(arch_test_pvalue) and arch_test_pvalue < 0.05) else 0.0
    volatility_risk_score = 100.0 * (
        (0.30 * clip01(abs(safe_float(rolling_std_mean_corr, default=0.0))))
        + (0.28 * clip01(max(safe_float(volatility_clustering_index, default=0.0), 0.0)))
        + (0.22 * arch_effect_flag)
        + (0.20 * clip01((safe_float(event_volatility_index, default=1.0) - 1.0) / 4.0))
    )
    volatility_risk_score = clip(volatility_risk_score, low=0.0, high=100.0)

    return {
        "observation_count": float(n),
        "rolling_std_mean_corr": float(rolling_std_mean_corr),
        "arch_test_pvalue": float(arch_test_pvalue),
        "volatility_clustering_index": float(volatility_clustering_index),
        "event_volatility_index": float(event_volatility_index),
        "volatility_risk_score": float(volatility_risk_score),
        "volatility_classification": classification,
    }

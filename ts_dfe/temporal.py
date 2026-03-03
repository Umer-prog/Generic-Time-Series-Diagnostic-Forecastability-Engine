from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf as sm_acf
    from statsmodels.tsa.stattools import pacf as sm_pacf

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from .utils import (
        ar_mae,
        ar_r2,
        clip,
        clip01,
        detect_seasonal_lag,
        manual_acf,
        naive_mae,
        safe_div,
        safe_float,
    )
except ImportError:
    from utils import (
        ar_mae,
        ar_r2,
        clip,
        clip01,
        detect_seasonal_lag,
        manual_acf,
        naive_mae,
        safe_div,
        safe_float,
    )


def _acf_with_fallback(series: pd.Series, max_lag: int = 10) -> dict[int, float]:
    s = pd.Series(series, dtype=float).dropna()
    if len(s) < 3:
        return {lag: np.nan for lag in range(max_lag + 1)}

    if HAS_STATSMODELS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    vals = sm_acf(s.values, nlags=max_lag, fft=False)
            out = {lag: np.nan for lag in range(max_lag + 1)}
            for lag, val in enumerate(vals):
                out[lag] = safe_float(val)
            return out
        except Exception:
            pass

    return manual_acf(s, max_lag=max_lag)


def _pacf_with_fallback(series: pd.Series, max_lag: int = 10) -> dict[int, float]:
    s = pd.Series(series, dtype=float).dropna()
    out = {lag: np.nan for lag in range(1, max_lag + 1)}
    if len(s) < 10:
        return out

    if HAS_STATSMODELS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    vals = sm_pacf(s.values, nlags=max_lag, method="ywm")
            for lag in range(1, min(len(vals), max_lag + 1)):
                out[lag] = safe_float(vals[lag])
            return out
        except Exception:
            pass

    # Fallback approximation: use ACF values if PACF is unavailable.
    acf_vals = manual_acf(s, max_lag=max_lag)
    for lag in range(1, max_lag + 1):
        out[lag] = safe_float(acf_vals.get(lag, np.nan))
    return out


def analyze_temporal(series: pd.Series, freq_seconds: float) -> dict:
    s = pd.Series(series, dtype=float).dropna()
    n = len(s)
    if n < 8:
        return {
            "observation_count": float(n),
            **{f"acf_lag_{i}": np.nan for i in range(1, 11)},
            **{f"pacf_lag_{i}": np.nan for i in range(1, 11)},
            "ljung_box_pvalue_lag10": np.nan,
            "ar1_r2": np.nan,
            "ar5_r2": np.nan,
            "seasonal_lag": 0.0,
            "seasonal_acf": np.nan,
            "seasonality_strength_stl": np.nan,
            "naive_mae": np.nan,
            "ar5_mae": np.nan,
            "autoregressive_improvement_over_naive": np.nan,
            "temporal_signal_strength_score": np.nan,
        }

    acf_vals = _acf_with_fallback(s, max_lag=10)
    pacf_vals = _pacf_with_fallback(s, max_lag=10)
    acf1 = safe_float(acf_vals.get(1, np.nan))

    if HAS_STATSMODELS and n > 15:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    lb = acorr_ljungbox(s.values, lags=[min(10, n - 1)], return_df=True)
            ljung_p = safe_float(lb["lb_pvalue"].iloc[0], default=np.nan)
        except Exception:
            ljung_p = np.nan
    else:
        ljung_p = np.nan

    ar1_r2 = ar_r2(s, lags=1)
    ar5_r2 = ar_r2(s, lags=5)

    seasonal_lag = detect_seasonal_lag(freq_seconds)
    seasonal_acf = np.nan
    if seasonal_lag > 1 and seasonal_lag < n:
        season_acf_vals = manual_acf(s, max_lag=seasonal_lag)
        seasonal_acf = safe_float(season_acf_vals.get(seasonal_lag, np.nan))

    if HAS_STATSMODELS and seasonal_lag >= 2 and n >= (3 * seasonal_lag):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    stl = STL(s, period=seasonal_lag, robust=True).fit()
            var_resid = np.var(stl.resid)
            var_combined = np.var(stl.seasonal + stl.resid)
            seasonality_strength = clip01(1.0 - safe_div(var_resid, var_combined, default=1.0))
        except Exception:
            seasonality_strength = np.nan
    else:
        seasonality_strength = np.nan

    naive_err = naive_mae(s)
    ar5_err = ar_mae(s, lags=5)
    ar_improvement = safe_div(naive_err - ar5_err, naive_err, default=np.nan)

    temporal_signal_strength_score = 100.0 * (
        (0.30 * clip01(abs(acf1)))
        + (0.24 * clip01(max(ar1_r2, ar5_r2, 0.0)))
        + (0.20 * clip01(max(seasonality_strength, 0.0) if np.isfinite(seasonality_strength) else 0.0))
        + (0.26 * clip01(max(ar_improvement, 0.0)))
    )
    temporal_signal_strength_score = clip(temporal_signal_strength_score, low=0.0, high=100.0)

    out = {
        "observation_count": float(n),
        "ljung_box_pvalue_lag10": float(ljung_p),
        "ar1_r2": float(ar1_r2),
        "ar5_r2": float(ar5_r2),
        "seasonal_lag": float(seasonal_lag),
        "seasonal_acf": float(seasonal_acf),
        "seasonality_strength_stl": float(seasonality_strength),
        "naive_mae": float(naive_err),
        "ar5_mae": float(ar5_err),
        "autoregressive_improvement_over_naive": float(ar_improvement),
        "temporal_signal_strength_score": float(temporal_signal_strength_score),
    }
    for i in range(1, 11):
        out[f"acf_lag_{i}"] = float(acf_vals.get(i, np.nan))
        out[f"pacf_lag_{i}"] = float(pacf_vals.get(i, np.nan))
    return out

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass
class SeriesBundle:
    series: pd.Series
    freq_alias: Optional[str]
    freq_seconds: float


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None:
            return float(default)
        val = float(value)
        if np.isfinite(val):
            return val
        return float(default)
    except Exception:
        return float(default)


def safe_div(numerator: float, denominator: float, default: float = np.nan) -> float:
    n = safe_float(numerator, default=default)
    d = safe_float(denominator, default=default)
    if not np.isfinite(n) or not np.isfinite(d) or abs(d) < EPS:
        return float(default)
    return float(n / d)


def clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    v = safe_float(value, default=low)
    return float(np.clip(v, low, high))


def clip01(value: float) -> float:
    return clip(value, low=0.0, high=1.0)


def prepare_dataframe(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise KeyError(f"Missing date column: {date_col}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[date_col, target_col])
    out = out.sort_values(date_col)
    return out


def seconds_to_alias(freq_seconds: float) -> Optional[str]:
    if not np.isfinite(freq_seconds) or freq_seconds <= 0:
        return None
    s = int(round(freq_seconds))
    if s % 86400 == 0:
        days = s // 86400
        return f"{days}D"
    if s % 3600 == 0:
        hours = s // 3600
        return f"{hours}h"
    if s % 60 == 0:
        minutes = s // 60
        return f"{minutes}min"
    return f"{s}s"


def normalize_freq_alias(freq_alias: Optional[str]) -> Optional[str]:
    if freq_alias is None:
        return None
    alias = str(freq_alias).strip()
    if not alias:
        return None

    m = re.match(r"^(\d+)?([A-Za-z]+)(-.+)?$", alias)
    if not m:
        return alias

    num, unit, suffix = m.groups()
    if suffix:
        return alias

    replacements = {
        "H": "h",
        "T": "min",
        "S": "s",
        "L": "ms",
        "U": "us",
        "N": "ns",
    }
    if unit in replacements:
        return f"{num or ''}{replacements[unit]}"
    return alias


def infer_frequency(index: pd.DatetimeIndex) -> Tuple[Optional[str], float]:
    if len(index) < 2:
        return None, np.nan

    idx = pd.DatetimeIndex(index).sort_values().unique()
    inferred = pd.infer_freq(idx)
    if inferred:
        try:
            normalized = normalize_freq_alias(inferred)
            offset = pd.tseries.frequencies.to_offset(normalized)
            seconds = _offset_to_seconds(offset)
            return normalized, safe_float(seconds)
        except Exception:
            pass

    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None, np.nan
    median_seconds = safe_float(deltas.median())
    return seconds_to_alias(median_seconds), median_seconds


def resolve_frequency(index: pd.DatetimeIndex, freq: Optional[str]) -> Tuple[Optional[str], float]:
    if freq:
        try:
            normalized = normalize_freq_alias(freq)
            offset = pd.tseries.frequencies.to_offset(normalized)
            seconds = _offset_to_seconds(offset)
            return normalized, safe_float(seconds)
        except Exception:
            pass
    return infer_frequency(index)


def _offset_to_seconds(offset: pd.tseries.offsets.BaseOffset) -> float:
    try:
        return safe_float(pd.Timedelta(offset).total_seconds())
    except Exception:
        name = str(getattr(offset, "name", "")).upper()
        mult = safe_float(getattr(offset, "n", 1), default=1.0)
        if name.startswith("W"):
            return 7.0 * 86400.0 * mult
        if name.startswith("M"):
            return 30.0 * 86400.0 * mult
        if name.startswith("Q"):
            return 91.0 * 86400.0 * mult
        if name.startswith("A") or name.startswith("Y"):
            return 365.0 * 86400.0 * mult
        if name.startswith("B"):
            return 86400.0 * mult
        return np.nan


def build_expected_index(
    index: pd.DatetimeIndex,
    freq_alias: Optional[str],
    freq_seconds: float,
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) == 0:
        return idx

    start, end = idx.min(), idx.max()
    if freq_alias:
        try:
            return pd.date_range(start=start, end=end, freq=freq_alias)
        except Exception:
            pass

    fallback_alias = seconds_to_alias(freq_seconds)
    if fallback_alias:
        try:
            return pd.date_range(start=start, end=end, freq=fallback_alias)
        except Exception:
            pass

    return idx


def build_regular_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    freq: Optional[str] = None,
    agg: str = "sum",
) -> SeriesBundle:
    clean = prepare_dataframe(df=df, date_col=date_col, target_col=target_col)
    grouped = clean.groupby(date_col)[target_col].agg(agg).sort_index().astype(float)
    freq_alias, freq_seconds = resolve_frequency(grouped.index, freq=freq)
    expected = build_expected_index(grouped.index, freq_alias=freq_alias, freq_seconds=freq_seconds)
    regular = grouped.reindex(expected)
    regular.index.name = date_col
    return SeriesBundle(series=regular.astype(float), freq_alias=freq_alias, freq_seconds=freq_seconds)


def rolling_window_size(
    n: int,
    min_size: int = 5,
    frac: float = 0.15,
    max_size: int = 60,
) -> int:
    if n <= 0:
        return min_size
    size = int(round(n * frac))
    return int(np.clip(size, min_size, max_size))


def lagged_design(series: pd.Series, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    y = pd.Series(series, dtype=float).dropna().values
    if len(y) <= lags + 2:
        return np.empty((0, lags)), np.array([])

    rows = []
    targets = []
    for i in range(lags, len(y)):
        rows.append(y[i - lags : i][::-1])
        targets.append(y[i])
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


def ols_predictions(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    if X.size == 0 or y.size == 0:
        return np.array([]), np.nan
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    return pred, safe_float(beta[0])


def ar_r2(series: pd.Series, lags: int) -> float:
    X, y = lagged_design(series, lags=lags)
    if len(y) < 8:
        return np.nan
    pred, _ = ols_predictions(X, y)
    if pred.size == 0:
        return np.nan
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return safe_div(ss_tot - ss_res, ss_tot, default=np.nan)


def ar_mae(series: pd.Series, lags: int) -> float:
    X, y = lagged_design(series, lags=lags)
    if len(y) < 8:
        return np.nan
    pred, _ = ols_predictions(X, y)
    if pred.size == 0:
        return np.nan
    return safe_float(np.mean(np.abs(y - pred)))


def naive_mae(series: pd.Series) -> float:
    s = pd.Series(series, dtype=float).dropna()
    if len(s) < 3:
        return np.nan
    return safe_float(np.mean(np.abs(s.iloc[1:].values - s.iloc[:-1].values)))


def mean_forecast_mae(series: pd.Series) -> float:
    s = pd.Series(series, dtype=float).dropna()
    if len(s) < 3:
        return np.nan
    actual = s.iloc[1:].values
    pred = s.iloc[:-1].expanding().mean().values
    return safe_float(np.mean(np.abs(actual - pred)))


def manual_acf(series: pd.Series, max_lag: int = 10) -> dict[int, float]:
    s = pd.Series(series, dtype=float).dropna().values
    if len(s) < 3:
        return {lag: np.nan for lag in range(max_lag + 1)}

    centered = s - np.mean(s)
    denom = np.sum(centered**2) + EPS
    acf_vals: dict[int, float] = {0: 1.0}
    for lag in range(1, max_lag + 1):
        if lag >= len(s):
            acf_vals[lag] = np.nan
            continue
        num = np.sum(centered[:-lag] * centered[lag:])
        acf_vals[lag] = safe_div(num, denom, default=np.nan)
    return acf_vals


def detect_seasonal_lag(freq_seconds: float) -> int:
    fs = safe_float(freq_seconds)
    if not np.isfinite(fs) or fs <= 0:
        return 0

    day = 86400.0
    week = 7 * day
    month = 30 * day

    if fs <= 3600:  # intra-day
        lag = int(round(day / fs))
        return max(lag, 2)
    if fs <= day + 60:
        return 7
    if fs <= week + day:
        return 52
    if fs <= month + week:
        return 12
    return 0


def trend_r2(series: pd.Series) -> float:
    s = pd.Series(series, dtype=float).dropna()
    if len(s) < 3:
        return np.nan
    x = np.arange(len(s), dtype=float)
    x = (x - x.mean()) / (x.std(ddof=0) + EPS)
    y = s.values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return safe_div(ss_tot - ss_res, ss_tot, default=np.nan)

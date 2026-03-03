from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .utils import clip, clip01, prepare_dataframe, safe_div, safe_float
except ImportError:
    from utils import clip, clip01, prepare_dataframe, safe_div, safe_float


def _gini(values: np.ndarray) -> float:
    x = np.array(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if len(x) == 0 or np.sum(x) <= 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    cum_weighted = np.sum((np.arange(1, n + 1)) * x)
    gini = (2.0 * cum_weighted) / (n * np.sum(x)) - (n + 1.0) / n
    return safe_float(gini)


def _normalized_entropy(prob: np.ndarray) -> float:
    p = np.array(prob, dtype=float)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return safe_float(entropy / np.log(len(p)))


def analyze_structure(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    structural_cols: list[str] | None = None,
) -> dict:
    structural_cols = structural_cols or []
    available_cols = [c for c in structural_cols if c in df.columns]
    if len(available_cols) == 0:
        return {
            "structure_enabled": 0.0,
            "entity_count": 0.0,
            "top1_revenue_share_pct": 0.0,
            "top5_revenue_share_pct": 0.0,
            "top10_revenue_share_pct": 0.0,
            "gini_coefficient": np.nan,
            "herfindahl_index": np.nan,
            "entropy_normalized": np.nan,
            "volatility_contribution_top_decile": np.nan,
            "customer_churn_volatility": np.nan,
            "concentration_score": 0.0,
            "structure_classification": "Not provided",
        }

    work = prepare_dataframe(df=df, date_col=date_col, target_col=target_col)
    for col in available_cols:
        work[col] = work[col].astype(str)

    entity_key = work[available_cols].agg("||".join, axis=1)
    work = work.assign(_entity_key=entity_key)

    totals = work.groupby("_entity_key")[target_col].sum().sort_values(ascending=False)
    total_sum = safe_float(totals.sum(), default=0.0)
    shares = (totals / (total_sum + 1e-9)).astype(float)

    top1_share = safe_float(shares.head(1).sum(), default=0.0)
    top5_share = safe_float(shares.head(5).sum(), default=0.0)
    top10_share = safe_float(shares.head(10).sum(), default=0.0)
    gini = _gini(shares.values)
    herfindahl = safe_float(np.sum((shares.values * 100.0) ** 2), default=np.nan)
    entropy_norm = _normalized_entropy(shares.values)

    # Volatility contribution by top decile entities.
    by_time_entity = (
        work.groupby([date_col, "_entity_key"])[target_col]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    entity_count = len(by_time_entity.columns)

    if entity_count > 0 and len(by_time_entity) > 3:
        top_decile_n = max(1, int(np.ceil(0.10 * entity_count)))
        top_entities = shares.head(top_decile_n).index.tolist()
        total_ts = by_time_entity.sum(axis=1)
        top_ts = by_time_entity[top_entities].sum(axis=1)
        total_var = np.var(total_ts.diff().dropna().values)
        top_var = np.var(top_ts.diff().dropna().values)
        volatility_contribution = safe_div(top_var, total_var, default=np.nan)
    else:
        volatility_contribution = np.nan

    # Customer/entity churn volatility per time step.
    if len(by_time_entity) > 2:
        active = by_time_entity > 0
        prev_active = active.shift(1, fill_value=False).astype(bool)
        entered = (~prev_active & active).sum(axis=1)
        exited = (prev_active & ~active).sum(axis=1)
        prev_count = prev_active.sum(axis=1).astype(float)
        prev_count[prev_count == 0.0] = np.nan
        churn_rate = (entered + exited) / prev_count
        churn_rate = churn_rate.replace([np.inf, -np.inf], np.nan).dropna()
        churn_volatility = safe_float(churn_rate.std(ddof=0), default=np.nan)
    else:
        churn_volatility = np.nan

    topk_pressure = top10_share if entity_count > 10 else top5_share
    concentration_score = 100.0 * (
        (0.24 * clip01(top1_share))
        + (0.24 * clip01(top5_share))
        + (0.20 * clip01(topk_pressure))
        + (0.18 * clip01(safe_div(herfindahl - 1000.0, 3000.0, default=0.0)))
        + (0.14 * clip01(1.0 - safe_float(entropy_norm, default=0.5)))
    )
    concentration_score = clip(concentration_score, low=0.0, high=100.0)

    dominated_rule = (
        (entity_count > 10 and top10_share >= 0.80)
        or (top1_share >= 0.50)
        or (safe_float(herfindahl, default=0.0) >= 2500.0)
    )
    concentrated_rule = (
        (entity_count > 10 and top10_share >= 0.60)
        or (top5_share >= 0.75)
        or (safe_float(gini, default=0.0) >= 0.55)
    )

    if dominated_rule:
        classification = "Dominated by few entities"
    elif concentrated_rule:
        classification = "Concentrated"
    else:
        classification = "Diversified"

    return {
        "structure_enabled": 1.0,
        "entity_count": float(entity_count),
        "top1_revenue_share_pct": float(top1_share * 100.0),
        "top5_revenue_share_pct": float(top5_share * 100.0),
        "top10_revenue_share_pct": float(top10_share * 100.0),
        "gini_coefficient": float(gini),
        "herfindahl_index": float(herfindahl),
        "entropy_normalized": float(entropy_norm),
        "volatility_contribution_top_decile": float(volatility_contribution),
        "customer_churn_volatility": float(churn_volatility),
        "concentration_score": float(concentration_score),
        "structure_classification": classification,
    }

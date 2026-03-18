from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from multivariate_decision import MultivariateDiagnostic
except Exception:
    MultivariateDiagnostic = None

try:
    from .classification import synthesize
    from .distribution import analyze_distribution
    from .granularity import analyze_granularity
    from .integrity import analyze_integrity
    from .readable import (
        TSDFEReport,
        build_summary_report,
        build_technical_report,
    )
    from .stationarity import analyze_stationarity
    from .structure import analyze_structure
    from .temporal import analyze_temporal
    from .utils import build_regular_series
    from .volatility import analyze_volatility
except ImportError:
    from classification import synthesize
    from distribution import analyze_distribution
    from granularity import analyze_granularity
    from integrity import analyze_integrity
    from readable import (
        TSDFEReport,
        build_summary_report,
        build_technical_report,
    )
    from stationarity import analyze_stationarity
    from structure import analyze_structure
    from temporal import analyze_temporal
    from utils import build_regular_series
    from volatility import analyze_volatility


DEFAULT_GRANULARITY_LEVELS = ("original", "W", "ME")


def _normalize_targets(target_col: str | None, target_cols: list[str] | None) -> list[str]:
    if target_cols is None:
        return [target_col] if target_col else []
    out = [c for c in target_cols if c]
    if target_col and target_col not in out:
        out.insert(0, target_col)
    return out


def _normalize_existing_columns(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    if not cols:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _resolve_mode(mode: str, target_cols: list[str], feature_cols: list[str]) -> str:
    m = (mode or "auto").strip().lower()
    if m not in {"auto", "univariate", "multivariate"}:
        raise ValueError("mode must be one of: auto, univariate, multivariate")
    if m == "auto":
        return "multivariate" if (len(target_cols) > 1 or len(feature_cols) > 0) else "univariate"
    return m


def _resolve_report_mode(report_mode: str) -> str:
    m = (report_mode or "technical").strip().lower()
    if m not in {"technical", "summary"}:
        raise ValueError("report_mode must be one of: technical, summary")
    return m


def _default_multivariate_signal(ar_r2: float = np.nan) -> dict:
    ar = float(ar_r2) if np.isfinite(ar_r2) else 0.0
    return {
        "cross_lag_effect": "weak",
        "residual_dependency": 0.0,
        "cv_improvement_multivariate": 0.0,
        "feature_utility_score": 0.0,
        "add_features_decision": "avoid_additional_features",
        "decision_confidence": 0.0,
        "feature_count": 0.0,
        "exogenous_r2": 0.0,
        "ar_r2": float(np.clip(ar, 0.0, 1.0)),
        "exogenous_dominance_ratio": 0.0,
        "exogenous_signal_classification": "Autoregressive Dominated",
        "recommendation": "univariate",
    }


def _infer_frequency_label(index: pd.DatetimeIndex) -> str:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) < 2:
        return "unknown"

    inferred = pd.infer_freq(idx)
    if inferred:
        alias = str(inferred).upper()
        if alias.endswith("D"):
            return "daily"
        if alias.startswith("W"):
            return "weekly"
        if alias in {"M", "ME"}:
            return "monthly"
        if alias in {"Q", "QE"}:
            return "quarterly"
        if alias in {"A", "Y", "YE"}:
            return "yearly"
        if alias.endswith("H"):
            return "hourly"
        return alias.lower()

    deltas = idx.to_series().diff().dropna().dt.total_seconds().values
    if len(deltas) == 0:
        return "unknown"
    med = float(np.median(deltas))
    if not np.isfinite(med):
        return "unknown"
    if abs(med - 86400.0) < 1.0:
        return "daily"
    if abs(med - (7.0 * 86400.0)) < 1.0:
        return "weekly"
    if 28.0 * 86400.0 <= med <= 31.0 * 86400.0:
        return "monthly"
    if abs(med - 3600.0) < 1.0:
        return "hourly"
    return f"{int(round(med))}s"


def _prepare_multivariate_timeseries(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]
    # Preserve order while removing duplicates.
    seen_features: set[str] = set()
    feature_cols = [c for c in feature_cols if not (c in seen_features or seen_features.add(c))]

    keep_cols = [date_col, target_col] + feature_cols
    seen_keep: set[str] = set()
    keep_cols = [c for c in keep_cols if c in df.columns and not (c in seen_keep or seen_keep.add(c))]
    work = df[keep_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    if target_col in work.columns:
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    for col in feature_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # Base-frequency preparation shared across engine and standalone diagnostics.
    agg_map: dict[str, Any] = {target_col: "sum"}
    for col in feature_cols:
        if col in work.columns:
            agg_map[col] = "mean"

    ts = work.groupby(date_col).agg(agg_map).sort_index()
    ts.index = pd.to_datetime(ts.index, errors="coerce")
    ts = ts[~ts.index.isna()]
    if target_col in ts.columns:
        ts = ts.dropna(subset=[target_col])

    try:
        monthly_effective = int(
            ts[target_col]
            .resample("ME")
            .sum(min_count=1)
            .dropna()
            .shape[0]
        )
    except Exception:
        monthly_effective = 0

    context = {
        "raw_rows": int(len(df)),
        "rows": int(len(ts)),
        "frequency": _infer_frequency_label(ts.index),
        "aggregation": "date groupby",
        "target": target_col,
        "features": [c for c in feature_cols if c in ts.columns],
        "effective_observations_monthly": monthly_effective,
    }
    return ts, context


def _grain_key(grain_cols: list[str], group_name: Any) -> str:
    if not grain_cols:
        return "all"
    if len(grain_cols) == 1:
        return f"{grain_cols[0]}={group_name}"
    if not isinstance(group_name, tuple):
        group_name = (group_name,)
    return "|".join(f"{c}={v}" for c, v in zip(grain_cols, group_name))


def _aggregate_for_granularity(
    df: pd.DataFrame,
    date_col: str,
    target_cols: list[str],
    feature_cols: list[str],
    grain_cols: list[str],
    level: str,
) -> pd.DataFrame:
    if level == "original":
        return df.copy()

    keep_cols = [date_col] + grain_cols + target_cols + feature_cols
    seen_keep: set[str] = set()
    keep_cols = [c for c in keep_cols if c in df.columns and not (c in seen_keep or seen_keep.add(c))]
    work = df[keep_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    for c in target_cols + feature_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    agg_map = {c: "sum" for c in target_cols if c in work.columns}
    agg_map.update({c: "mean" for c in feature_cols if c in work.columns and c not in target_cols})

    groupers: list[Any] = [pd.Grouper(key=date_col, freq=level)]
    if grain_cols:
        groupers = grain_cols + groupers

    out = work.groupby(groupers, dropna=False).agg(agg_map).reset_index()
    return out


def _run_univariate_pipeline(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    structural_cols: list[str] | None = None,
    freq: str | None = None,
    report_mode: str = "technical",
    multivariate_signal: dict | None = None,
) -> TSDFEReport:
    structural_cols = structural_cols or []

    integrity = analyze_integrity(
        df=df,
        date_col=date_col,
        target_col=target_col,
        freq=freq,
    )

    bundle = build_regular_series(
        df=df,
        date_col=date_col,
        target_col=target_col,
        freq=freq,
        agg="sum",
    )
    regular = bundle.series
    model_ready_series = regular.interpolate(method="time").ffill().bfill()

    distribution = analyze_distribution(regular.dropna())
    temporal = analyze_temporal(model_ready_series, freq_seconds=bundle.freq_seconds)
    stationarity = analyze_stationarity(model_ready_series)
    volatility = analyze_volatility(model_ready_series)
    structure = analyze_structure(
        df=df,
        date_col=date_col,
        target_col=target_col,
        structural_cols=structural_cols,
    )
    granularity = analyze_granularity(model_ready_series)

    modules = {
        "integrity": integrity,
        "distribution": distribution,
        "temporal": temporal,
        "stationarity": stationarity,
        "volatility": volatility,
        "structure": structure,
        "granularity": granularity,
    }

    synthesis = synthesize(modules)
    signal = multivariate_signal or _default_multivariate_signal(ar_r2=temporal.get("ar5_r2", np.nan))

    result: dict[str, Any] = {
        "mode": "univariate",
        "report_mode": report_mode,
        "integrity": modules["integrity"],
        "distribution": modules["distribution"],
        "temporal": modules["temporal"],
        "stationarity": modules["stationarity"],
        "volatility": modules["volatility"],
        "structure": modules["structure"],
        "granularity": modules["granularity"],
        "multivariate_signal": signal,
        "classification": synthesis["classification"],
        "forecastability_score": synthesis["forecastability_score"],
        "modeling_recommendation": synthesis["modeling_recommendation"],
        "risk_flags": synthesis["risk_flags"],
        "executive_summary": synthesis["executive_summary"],
        "engineering_decision_recommendation": synthesis["engineering_decision_recommendation"],
    }
    if report_mode == "summary":
        result["human_readable_report"] = build_summary_report(result)
    else:
        result["human_readable_report"] = build_technical_report(result)
    return TSDFEReport(result)


def _run_multivariate_for_target(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: list[str],
    freq: str | None = None,
) -> dict:
    ts, data_context = _prepare_multivariate_timeseries(
        df=df,
        date_col=date_col,
        target_col=target_col,
        feature_cols=exog_cols,
    )

    if target_col not in ts.columns:
        out = _default_multivariate_signal()
        out["feature_count"] = float(len(exog_cols))
        out["data_context"] = data_context
        out["error"] = f"Target column '{target_col}' unavailable after aggregation."
        return out

    exog_present = [c for c in exog_cols if c in ts.columns]
    data_context["features"] = exog_present

    if not exog_present:
        out = _default_multivariate_signal()
        out["ar_r2"] = float(0.0)
        out["data_context"] = data_context
        return out

    if MultivariateDiagnostic is None:
        out = _default_multivariate_signal()
        out["recommendation"] = "univariate_with_exogenous"
        out["feature_count"] = float(len(exog_present))
        out["data_context"] = data_context
        out["error"] = "MultivariateDiagnostic module unavailable."
        return out

    diagnostic = MultivariateDiagnostic()
    out = diagnostic.diagnose(
        target=ts[target_col],
        exog=ts[exog_present],
        data_context=data_context,
    )
    if "feature_count" not in out:
        out["feature_count"] = float(len(exog_present))
    if "data_context" not in out:
        out["data_context"] = data_context
    return out


def _build_expanded_summary(result: dict) -> str:
    targets = list(result.get("overall_univariate", {}).keys())
    mode = result.get("mode", "unknown")
    grains = len(result.get("by_grain", {}))
    lines = [
        f"TS-DFE expanded run mode={mode}",
        f"targets={targets}",
        f"grain_groups={grains}",
    ]

    recs = result.get("recommended_approach_by_target", {})
    if recs:
        lines.append("recommended_approach_by_target=" + str(recs))
    return " | ".join(lines)


def run_ts_dfe(
    df: pd.DataFrame,
    date_col: str,
    target_col: str | None = None,
    structural_cols: list[str] | None = None,
    freq: str | None = None,
    *,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    grain_cols: list[str] | None = None,
    granularity_levels: list[str] | None = None,
    mode: str = "auto",
    report_mode: str = "technical",
    min_points_per_group: int = 40,
    max_grain_groups: int | None = 50,
) -> dict:
    target_cols_resolved = _normalize_targets(target_col=target_col, target_cols=target_cols)
    if not target_cols_resolved:
        raise ValueError("Provide target_col or target_cols.")

    structural_cols = _normalize_existing_columns(df, structural_cols or [])
    feature_cols = _normalize_existing_columns(df, feature_cols or [])
    grain_cols = _normalize_existing_columns(df, grain_cols or [])
    target_cols_resolved = _normalize_existing_columns(df, target_cols_resolved)

    if not target_cols_resolved:
        raise ValueError("None of the provided target columns exist in df.")

    granularity_levels = granularity_levels or list(DEFAULT_GRANULARITY_LEVELS)
    granularity_levels = [g for g in granularity_levels if g]
    resolved_mode = _resolve_mode(mode=mode, target_cols=target_cols_resolved, feature_cols=feature_cols)
    resolved_report_mode = _resolve_report_mode(report_mode=report_mode)

    # Backward-compatible single-target path.
    legacy_single_target = (
        len(target_cols_resolved) == 1
        and len(grain_cols) == 0
        and resolved_mode == "univariate"
        and len(feature_cols) == 0
    )
    if legacy_single_target:
        return _run_univariate_pipeline(
            df=df,
            date_col=date_col,
            target_col=target_cols_resolved[0],
            structural_cols=structural_cols,
            freq=freq,
            report_mode=resolved_report_mode,
        )

    result: dict[str, Any] = {
        "mode": resolved_mode,
        "report_mode": resolved_report_mode,
        "config": {
            "date_col": date_col,
            "target_cols": target_cols_resolved,
            "feature_cols": feature_cols,
            "grain_cols": grain_cols,
            "structural_cols": structural_cols,
            "granularity_levels": granularity_levels,
            "min_points_per_group": int(min_points_per_group),
            "max_grain_groups": max_grain_groups,
            "freq": freq,
            "report_mode": resolved_report_mode,
        },
        "overall_univariate": {},
        "overall_multivariate": {},
        "overall_by_granularity": {},
        "best_granularity_by_target": {},
        "recommended_approach_by_target": {},
        "by_grain": {},
    }

    # Overall univariate and multivariate diagnostics.
    for target in target_cols_resolved:
        exog_cols = []
        for col in target_cols_resolved + feature_cols:
            if col != target and col in df.columns and col not in exog_cols:
                exog_cols.append(col)

        if resolved_mode == "multivariate":
            multi_report = _run_multivariate_for_target(
                df=df,
                date_col=date_col,
                target_col=target,
                exog_cols=exog_cols,
                freq=freq,
            )
        else:
            multi_report = _default_multivariate_signal()

        uni_report = _run_univariate_pipeline(
            df=df,
            date_col=date_col,
            target_col=target,
            structural_cols=structural_cols,
            freq=freq,
            report_mode=resolved_report_mode,
            multivariate_signal=multi_report,
        )
        result["overall_univariate"][target] = dict(uni_report)
        result["best_granularity_by_target"][target] = uni_report["granularity"]["optimal_granularity"]

        result["overall_multivariate"][target] = multi_report
        result["recommended_approach_by_target"][target] = multi_report.get("recommendation", "univariate")

    # Overall diagnostics per requested granularity level.
    for level in granularity_levels:
        level_df = _aggregate_for_granularity(
            df=df,
            date_col=date_col,
            target_cols=target_cols_resolved,
            feature_cols=feature_cols,
            grain_cols=[],
            level=level,
        )
        level_entry: dict[str, Any] = {}
        for target in target_cols_resolved:
            if target not in level_df.columns:
                continue
            level_report = _run_univariate_pipeline(
                df=level_df[[date_col, target]].dropna(subset=[target]),
                date_col=date_col,
                target_col=target,
                structural_cols=[],
                freq=(None if level == "original" else level),
                report_mode=resolved_report_mode,
            )
            level_entry[target] = {
                "classification": level_report["classification"],
                "forecastability_score": level_report["forecastability_score"],
                "optimal_granularity": level_report["granularity"]["optimal_granularity"],
            }
        result["overall_by_granularity"][level] = level_entry

    # Grain-wise diagnostics.
    if grain_cols:
        grouped = list(df.groupby(grain_cols, dropna=False, sort=False))
        if max_grain_groups is not None and len(grouped) > max_grain_groups:
            grouped = sorted(grouped, key=lambda item: len(item[1]), reverse=True)[:max_grain_groups]

        for gname, gdf in grouped:
            key = _grain_key(grain_cols, gname)
            if len(gdf) < min_points_per_group:
                result["by_grain"][key] = {
                    "row_count": int(len(gdf)),
                    "status": "skipped_insufficient_points",
                }
                continue

            grain_entry: dict[str, Any] = {
                "row_count": int(len(gdf)),
                "univariate": {},
                "multivariate": {},
                "best_granularity_by_target": {},
                "recommended_approach_by_target": {},
                "by_granularity": {},
            }

            for target in target_cols_resolved:
                exog_cols = []
                for col in target_cols_resolved + feature_cols:
                    if col != target and col in gdf.columns and col not in exog_cols:
                        exog_cols.append(col)
                if resolved_mode == "multivariate":
                    multi_report = _run_multivariate_for_target(
                        df=gdf,
                        date_col=date_col,
                        target_col=target,
                        exog_cols=exog_cols,
                        freq=freq,
                    )
                else:
                    multi_report = _default_multivariate_signal()

                uni_report = _run_univariate_pipeline(
                    df=gdf,
                    date_col=date_col,
                    target_col=target,
                    structural_cols=structural_cols,
                    freq=freq,
                    report_mode=resolved_report_mode,
                    multivariate_signal=multi_report,
                )
                grain_entry["univariate"][target] = dict(uni_report)
                grain_entry["best_granularity_by_target"][target] = uni_report["granularity"]["optimal_granularity"]
                grain_entry["multivariate"][target] = multi_report
                grain_entry["recommended_approach_by_target"][target] = multi_report.get("recommendation", "univariate")

            for level in granularity_levels:
                g_level_df = _aggregate_for_granularity(
                    df=gdf,
                    date_col=date_col,
                    target_cols=target_cols_resolved,
                    feature_cols=feature_cols,
                    grain_cols=[],
                    level=level,
                )
                level_entry: dict[str, Any] = {}
                for target in target_cols_resolved:
                    if target not in g_level_df.columns:
                        continue
                    level_report = _run_univariate_pipeline(
                        df=g_level_df[[date_col, target]].dropna(subset=[target]),
                        date_col=date_col,
                        target_col=target,
                        structural_cols=[],
                        freq=(None if level == "original" else level),
                        report_mode=resolved_report_mode,
                    )
                    level_entry[target] = {
                        "classification": level_report["classification"],
                        "forecastability_score": level_report["forecastability_score"],
                        "optimal_granularity": level_report["granularity"]["optimal_granularity"],
                    }
                grain_entry["by_granularity"][level] = level_entry

            result["by_grain"][key] = grain_entry

    result["summary"] = _build_expanded_summary(result)
    if resolved_report_mode == "summary":
        result["human_readable_report"] = build_summary_report(result)
    else:
        result["human_readable_report"] = build_technical_report(result)
    return TSDFEReport(result)

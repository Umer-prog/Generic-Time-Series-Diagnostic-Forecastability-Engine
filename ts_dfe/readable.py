from __future__ import annotations

from typing import Any

import numpy as np


CORE_MODULES = [
    "integrity",
    "distribution",
    "temporal",
    "stationarity",
    "volatility",
    "structure",
    "granularity",
]

MULTIVARIATE_SIGNAL_KEYS = [
    "cross_lag_effect",
    "residual_dependency",
    "cv_improvement_multivariate",
    "feature_utility_score",
    "decision_confidence",
    "exogenous_r2",
    "ar_r2",
    "exogenous_dominance_ratio",
    "exogenous_signal_classification",
    "recommendation",
]


def _is_expanded_report(report: dict) -> bool:
    return isinstance(report, dict) and "overall_univariate" in report and "overall_multivariate" in report


class TSDFEReport(dict):
    def __str__(self) -> str:
        mode = str(self.get("report_mode", "technical")).strip().lower()
        if mode == "summary":
            return build_summary_report(self)
        return build_technical_report(self)

    def __repr__(self) -> str:
        return self.__str__()


def _fmt_num(value: Any, decimals: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if np.isnan(x):
        return "nan"
    if abs(x) >= 1000:
        return f"{x:,.{decimals}f}"
    return f"{x:.{decimals}f}"


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for key, val in d.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(val, dict):
                out.update(_flatten(val, next_prefix))
            else:
                out[next_prefix] = val
    else:
        out[prefix] = d
    return out


def _signal_label(score: float) -> str:
    if np.isnan(score):
        return "Unknown"
    if score < 35:
        return "Weak"
    if score < 65:
        return "Moderate"
    return "Strong"


def _seasonality_label(strength: float) -> str:
    if np.isnan(strength):
        return "Unknown"
    if strength < 0.20:
        return "Weak"
    if strength < 0.50:
        return "Moderate"
    return "Strong"


def _split_recommendation_to_bullets(recommendation: str) -> list[str]:
    if not recommendation:
        return []

    raw = recommendation.strip().rstrip(".")
    if " because " in raw:
        action, reason = raw.split(" because ", 1)
        return [
            action.strip(),
            f"Reason: {reason.strip()}",
        ]
    if ";" in raw:
        return [x.strip() for x in raw.split(";") if x.strip()]
    return [raw]


def _resolve_driver_signal(report: dict, multi: dict | None = None) -> dict:
    m = multi if isinstance(multi, dict) else report.get("multivariate_signal", {})
    if not isinstance(m, dict):
        m = {}

    ar_r2 = float(m.get("ar_r2", report.get("temporal", {}).get("ar5_r2", np.nan)))
    if not np.isfinite(ar_r2):
        ar_r2 = 0.0
    exogenous_r2 = float(m.get("exogenous_r2", 0.0))
    if not np.isfinite(exogenous_r2):
        exogenous_r2 = 0.0
    ratio = float(m.get("exogenous_dominance_ratio", exogenous_r2 / (ar_r2 + 1e-6)))
    if not np.isfinite(ratio):
        ratio = 0.0

    signal_class = m.get("exogenous_signal_classification", None)
    if not isinstance(signal_class, str) or not signal_class:
        if ratio >= 1.5:
            signal_class = "Exogenous Dominated"
        elif ratio >= 0.7:
            signal_class = "Mixed Drivers"
        else:
            signal_class = "Autoregressive Dominated"

    return {
        "exogenous_r2": float(exogenous_r2),
        "ar_r2": float(ar_r2),
        "exogenous_dominance_ratio": float(ratio),
        "exogenous_signal_classification": signal_class,
    }


def _build_reason_block(report: dict, multi: dict | None = None) -> list[str]:
    distribution = report.get("distribution", {})
    volatility = report.get("volatility", {})
    granularity = report.get("granularity", {})
    driver = _resolve_driver_signal(report, multi=multi)

    reason_items = [
        f"volatility_classification={volatility.get('volatility_classification', 'Unknown')}",
        f"distribution_classification={distribution.get('distribution_classification', 'Unknown')}",
        f"model_improvement_ratio={_fmt_num(granularity.get('model_improvement_ratio', np.nan), 3)}",
        f"exogenous_signal_classification={driver['exogenous_signal_classification']}",
    ]
    return reason_items


def _noise_signal_ratio(mean_value: float, naive_mae_value: float) -> float:
    m = float(mean_value)
    n = float(naive_mae_value)
    if not np.isfinite(m) or not np.isfinite(n):
        return np.nan
    denom = abs(m)
    if denom <= 1e-12:
        return np.nan
    return float(n / denom)


def _predictability_label(noise_signal_ratio: float) -> str:
    if not np.isfinite(noise_signal_ratio):
        return "Unknown"
    if noise_signal_ratio > 1.0:
        return "Very noisy"
    if noise_signal_ratio >= 0.5:
        return "Moderate noise"
    return "Strong signal"


def _structure_summary_text(structure: dict) -> str:
    enabled = float(structure.get("structure_enabled", 0.0))
    if not np.isfinite(enabled) or enabled <= 0.5:
        return "Structure diagnostics disabled (no entity columns provided)"
    return (
        f"{structure.get('structure_classification', 'Not provided')} "
        f"(top5_revenue_share_pct={_fmt_num(structure.get('top5_revenue_share_pct', np.nan), 2)}%)"
    )


def _driver_signal_note(driver_classification: str) -> str:
    if driver_classification == "Exogenous Dominated":
        return "Exogenous drivers are primary predictive signal"
    if driver_classification == "Mixed Drivers":
        return "Both autoregressive and exogenous effects are relevant"
    return "Exogenous drivers provide limited predictive power"


def _driver_consistent_recommendation(
    driver_classification: str,
    temporal_signal_score: float,
    volatility_classification: str,
) -> str:
    vol = str(volatility_classification)
    if driver_classification == "Exogenous Dominated":
        if vol == "Event volatility":
            return "Use event-driven regression models with external drivers"
        return "Use regression / feature-based models"
    if driver_classification == "Mixed Drivers":
        return "Use hybrid autoregressive + feature models"

    # Autoregressive Dominated
    if np.isfinite(temporal_signal_score) and temporal_signal_score >= 35.0:
        return "Use autoregressive models with robust baseline fallback"
    return "Use robust baseline models with event detection"


def _model_starting_point(
    driver_classification: str,
    temporal_signal_score: float,
    volatility_classification: str,
    optimal_granularity: str,
    seasonality_strength: float,
) -> dict[str, str]:
    vol = str(volatility_classification)
    gran = str(optimal_granularity or "original")
    gran_title = gran.capitalize()

    if driver_classification == "Exogenous Dominated":
        primary_model = "Event-driven regression" if vol == "Event volatility" else "Feature-based regression"
        feature_usage = "Required (high driver signal)"
    elif driver_classification == "Mixed Drivers":
        primary_model = "Hybrid autoregressive + regression model"
        feature_usage = "Recommended (mixed driver signal)"
    else:
        if np.isfinite(temporal_signal_score) and temporal_signal_score >= 35.0:
            primary_model = "Autoregressive model (ARIMA/ETS)"
        else:
            primary_model = "Robust regression or baseline model"
        feature_usage = "Optional (low driver signal)"

    if np.isfinite(seasonality_strength) and seasonality_strength >= 0.2:
        baseline_model = "Seasonal naive"
    else:
        baseline_model = "Naive"

    if vol in {"Event volatility", "Heteroskedastic", "Clustered volatility"}:
        loss_function = "Quantile or Huber"
    elif vol == "Homoskedastic":
        loss_function = "MAE"
    else:
        loss_function = "Quantile or Huber"

    return {
        "primary_model": primary_model,
        "baseline_model": baseline_model,
        "training_granularity": gran_title,
        "loss_function": loss_function,
        "feature_usage": feature_usage,
    }


def _append_final_output_fields(
    lines: list[str],
    report: dict,
    risk_flags: list[str],
    modeling_recommendation_override: str | None = None,
) -> None:
    lines.append("[FINAL OUTPUT FIELDS]")
    lines.append(f"- classification: {report.get('classification', 'Unknown')}")
    lines.append(f"- forecastability_score: {_fmt_num(report.get('forecastability_score', np.nan), 3)}")
    model_reco = modeling_recommendation_override
    if model_reco is None:
        model_reco = str(report.get("modeling_recommendation", ""))
    lines.append(f"- modeling_recommendation: {model_reco}")
    lines.append(f"- risk_flags_count: {len(risk_flags)}")
    lines.append(f"- executive_summary: {report.get('executive_summary', '')}")
    lines.append(
        "- engineering_decision_recommendation: "
        f"{report.get('engineering_decision_recommendation', '')}"
    )


def build_univariate_technical_report(report: dict) -> str:
    integrity = report.get("integrity", {})
    distribution = report.get("distribution", {})
    temporal = report.get("temporal", {})
    stationarity = report.get("stationarity", {})
    volatility = report.get("volatility", {})
    structure = report.get("structure", {})
    granularity = report.get("granularity", {})

    acf_vals = [
        abs(float(temporal.get("acf_lag_1", np.nan))),
        abs(float(temporal.get("acf_lag_2", np.nan))),
        abs(float(temporal.get("acf_lag_3", np.nan))),
    ]
    acf_vals = [x for x in acf_vals if np.isfinite(x)]
    avg_acf = float(np.mean(acf_vals)) if acf_vals else np.nan

    temporal_score = float(temporal.get("temporal_signal_strength_score", np.nan))
    signal_label = _signal_label(temporal_score)
    seasonality_strength = float(temporal.get("seasonality_strength_stl", np.nan))
    seasonality_label = _seasonality_label(seasonality_strength)

    dist_class = distribution.get("distribution_classification", "Unknown")
    skewness = float(distribution.get("skewness", np.nan))
    vol_class = volatility.get("volatility_classification", "Unknown")
    top5_pct = float(structure.get("top5_revenue_share_pct", np.nan))

    optimal_granularity = str(granularity.get("optimal_granularity", "original"))
    signal_gain_ratio = float(granularity.get("signal_gain_ratio", np.nan))
    if np.isfinite(signal_gain_ratio):
        signal_gain_pct = (signal_gain_ratio - 1.0) * 100.0
    else:
        signal_gain_pct = np.nan

    recommendation_bullets = _split_recommendation_to_bullets(str(report.get("modeling_recommendation", "")))
    risk_flags = report.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = [str(risk_flags)]

    lines: list[str] = []
    lines.append("DATA CHARACTERIZATION REPORT")
    lines.append("----------------------------------------")
    lines.append(
        f"Signal Strength: {signal_label} "
        f"(temporal_signal_strength_score={_fmt_num(temporal_score, 1)}, avg_acf_lag_1_3={_fmt_num(avg_acf, 3)})"
    )
    lines.append(
        f"Variance: {vol_class} "
        f"(volatility_risk_score={_fmt_num(volatility.get('volatility_risk_score', np.nan), 1)})"
    )
    lines.append(
        f"Distribution: {dist_class} "
        f"(skewness={_fmt_num(skewness, 3)}, kurtosis={_fmt_num(distribution.get('kurtosis', np.nan), 3)})"
    )
    lines.append(
        f"Concentration: {structure.get('structure_classification', 'Not provided')} "
        f"(top5_revenue_share_pct={_fmt_num(top5_pct, 2)}%)"
    )
    lines.append(
        f"Seasonality: {seasonality_label} "
        f"(seasonality_strength_stl={_fmt_num(seasonality_strength, 3)}, seasonal_lag={_fmt_num(temporal.get('seasonal_lag', np.nan), 0)})"
    )
    lines.append(
        f"Granularity: optimal={optimal_granularity} "
        f"(signal_gain={_fmt_num(signal_gain_pct, 1)}%, noise_reduction_ratio={_fmt_num(granularity.get('noise_reduction_ratio', np.nan), 3)})"
    )
    lines.append("")
    lines.append(f"CLASSIFICATION: {report.get('classification', 'Unknown')}")
    lines.append("")
    lines.append(f"FORECASTABILITY SCORE: {_fmt_num(report.get('forecastability_score', np.nan), 0)}/100")
    lines.append("")
    lines.append("RECOMMENDATION:")
    if recommendation_bullets:
        for item in recommendation_bullets:
            lines.append(f"- {item}")
    else:
        lines.append("- No recommendation generated")

    lines.append("")
    lines.append("RISK FLAGS:")
    if risk_flags:
        for item in risk_flags:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("EXECUTIVE SUMMARY:")
    lines.append(str(report.get("executive_summary", "")))
    lines.append("")
    lines.append("ENGINEERING DECISION RECOMMENDATION:")
    lines.append(str(report.get("engineering_decision_recommendation", "")))
    lines.append("")
    lines.append("FULL STRUCTURED METRICS")
    lines.append("----------------------------------------")

    for module_name in CORE_MODULES:
        module_data = report.get(module_name, {})
        lines.append(f"[{module_name.upper()}]")
        flat = _flatten(module_data)
        for key in sorted(flat.keys()):
            value = flat[key]
            if isinstance(value, str):
                rendered = value
            elif isinstance(value, (float, int, np.floating, np.integer)):
                rendered = _fmt_num(value)
            elif isinstance(value, list):
                rendered = ", ".join(str(x) for x in value) if value else "[]"
            else:
                rendered = str(value)
            lines.append(f"- {key}: {rendered}")
        lines.append("")

    _append_final_output_fields(lines, report, risk_flags)
    return "\n".join(lines)


def build_univariate_summary_report(report: dict) -> str:
    distribution = report.get("distribution", {})
    temporal = report.get("temporal", {})
    stationarity = report.get("stationarity", {})
    volatility = report.get("volatility", {})
    structure = report.get("structure", {})
    granularity = report.get("granularity", {})
    risk_flags = report.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = [str(risk_flags)]

    acf_vals = [
        abs(float(temporal.get("acf_lag_1", np.nan))),
        abs(float(temporal.get("acf_lag_2", np.nan))),
        abs(float(temporal.get("acf_lag_3", np.nan))),
    ]
    acf_vals = [x for x in acf_vals if np.isfinite(x)]
    avg_acf = float(np.mean(acf_vals)) if acf_vals else np.nan

    temporal_score = float(temporal.get("temporal_signal_strength_score", np.nan))
    signal_label = _signal_label(temporal_score)
    seasonality_strength = float(temporal.get("seasonality_strength_stl", np.nan))
    seasonality_label = _seasonality_label(seasonality_strength)

    optimal_granularity = str(granularity.get("optimal_granularity", "original"))
    signal_gain_ratio = float(granularity.get("signal_gain_ratio", np.nan))
    signal_gain_pct = (signal_gain_ratio - 1.0) * 100.0 if np.isfinite(signal_gain_ratio) else np.nan

    driver = _resolve_driver_signal(report)
    reason_items = _build_reason_block(report)
    driver_classification = driver["exogenous_signal_classification"]
    driver_note = _driver_signal_note(driver_classification)
    action = _driver_consistent_recommendation(
        driver_classification=driver_classification,
        temporal_signal_score=temporal_score,
        volatility_classification=str(volatility.get("volatility_classification", "Unknown")),
    )
    recommendation_full = f"{action} because {', '.join(reason_items)}."
    noise_signal_ratio = _noise_signal_ratio(
        mean_value=float(distribution.get("mean", np.nan)),
        naive_mae_value=float(temporal.get("naive_mae", np.nan)),
    )
    predictability = _predictability_label(noise_signal_ratio)
    structure_text = _structure_summary_text(structure)
    start = _model_starting_point(
        driver_classification=driver_classification,
        temporal_signal_score=temporal_score,
        volatility_classification=str(volatility.get("volatility_classification", "Unknown")),
        optimal_granularity=optimal_granularity,
        seasonality_strength=seasonality_strength,
    )

    lines: list[str] = []
    lines.append("DATA CHARACTERIZATION REPORT")
    lines.append("----------------------------------------")
    lines.append(
        f"Signal Strength: {signal_label} "
        f"(temporal_signal_strength_score={_fmt_num(temporal_score, 1)}, avg_acf_lag_1_3={_fmt_num(avg_acf, 3)})"
    )
    lines.append(
        f"Predictability: {predictability} (noise_signal_ratio={_fmt_num(noise_signal_ratio, 2)})"
    )
    lines.append(
        f"Variance: {volatility.get('volatility_classification', 'Unknown')} "
        f"(volatility_risk_score={_fmt_num(volatility.get('volatility_risk_score', np.nan), 1)})"
    )
    lines.append(
        f"Distribution: {distribution.get('distribution_classification', 'Unknown')} "
        f"(skewness={_fmt_num(distribution.get('skewness', np.nan), 3)}, "
        f"kurtosis={_fmt_num(distribution.get('kurtosis', np.nan), 3)})"
    )
    lines.append(f"Concentration: {structure_text}")
    lines.append(
        f"Seasonality: {seasonality_label} "
        f"(seasonality_strength_stl={_fmt_num(seasonality_strength, 3)}, "
        f"seasonal_lag={_fmt_num(temporal.get('seasonal_lag', np.nan), 0)})"
    )
    lines.append(
        f"Granularity: optimal={optimal_granularity} "
        f"(signal_gain={_fmt_num(signal_gain_pct, 1)}%, "
        f"noise_reduction_ratio={_fmt_num(granularity.get('noise_reduction_ratio', np.nan), 3)})"
    )
    lines.append(
        f"Driver Signal: {driver_classification} "
        f"(exogenous_dominance_ratio={_fmt_num(driver['exogenous_dominance_ratio'], 2)}, "
        f"exogenous_r2={_fmt_num(driver['exogenous_r2'], 2)}, "
        f"ar_r2={_fmt_num(driver['ar_r2'], 2)})"
    )
    lines.append(f"-> {driver_note}")
    lines.append("")
    lines.append(f"CLASSIFICATION: {report.get('classification', 'Unknown')}")
    lines.append("")
    lines.append(f"FORECASTABILITY SCORE: {_fmt_num(report.get('forecastability_score', np.nan), 0)}/100")
    lines.append("RECOMMENDATION:")
    lines.append(f"- {action if action else 'No recommendation generated'}")
    lines.append("")
    lines.append("Reason:")
    lines.append(",\n".join(reason_items))
    lines.append("")
    lines.append("MODEL STARTING POINT")
    lines.append("----------------------------------------")
    lines.append(f"Primary model: {start['primary_model']}")
    lines.append(f"Baseline model: {start['baseline_model']}")
    lines.append(f"Training granularity: {start['training_granularity']}")
    lines.append(f"Loss function: {start['loss_function']}")
    lines.append(f"Feature usage: {start['feature_usage']}")
    lines.append("")
    lines.append("RISK FLAGS:")
    if risk_flags:
        for item in risk_flags:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("EXECUTIVE SUMMARY:")
    lines.append(
        f"{report.get('classification', 'Unknown')} with forecastability_score="
        f"{_fmt_num(report.get('forecastability_score', np.nan), 2)}."
    )
    lines.append("")
    lines.append("Key metrics:")
    lines.append(
        f"temporal_signal_strength_score={_fmt_num(temporal_score, 3)},\n"
        f"stability_score={_fmt_num(stationarity.get('stability_score', np.nan), 3)},\n"
        f"model_improvement_ratio={_fmt_num(granularity.get('model_improvement_ratio', np.nan), 3)},\n"
        f"exogenous_dominance_ratio={_fmt_num(driver['exogenous_dominance_ratio'], 2)},\n"
        f"optimal_granularity={optimal_granularity}."
    )
    lines.append("")
    lines.append("ENGINEERING DECISION RECOMMENDATION:")
    lines.append(str(report.get("engineering_decision_recommendation", "")))
    lines.append("")
    _append_final_output_fields(
        lines,
        report,
        risk_flags,
        modeling_recommendation_override=recommendation_full,
    )
    return "\n".join(lines)


def _render_module(lines: list[str], module_name: str, module_data: dict) -> None:
    lines.append(f"[{module_name.upper()}]")
    flat = _flatten(module_data)
    for key in sorted(flat.keys()):
        value = flat[key]
        if isinstance(value, str):
            rendered = value
        elif isinstance(value, (float, int, np.floating, np.integer)):
            rendered = _fmt_num(value)
        elif isinstance(value, list):
            rendered = ", ".join(str(x) for x in value) if value else "[]"
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    lines.append("")


def build_expanded_technical_report(report: dict) -> str:
    mode = report.get("mode", "unknown")
    config = report.get("config", {}) if isinstance(report.get("config", {}), dict) else {}
    targets = config.get("target_cols", [])
    features = config.get("feature_cols", [])
    grains = config.get("grain_cols", [])
    levels = config.get("granularity_levels", [])
    overall_uni = report.get("overall_univariate", {})
    overall_multi = report.get("overall_multivariate", {})
    overall_by_level = report.get("overall_by_granularity", {})
    by_grain = report.get("by_grain", {})
    best_grain = report.get("best_granularity_by_target", {})

    lines: list[str] = []
    lines.append("TS-DFE EXPANDED REPORT")
    lines.append("=== TECHNICAL DIAGNOSTIC REPORT ===")
    lines.append("----------------------------------------")
    lines.append(f"Mode: {mode}")
    lines.append(f"Targets: {targets}")
    lines.append(f"Features: {features if features else 'None'}")
    lines.append(f"Grain Columns: {grains if grains else 'None'}")
    lines.append(f"Granularity Levels: {levels if levels else 'None'}")
    lines.append("")

    for target in targets:
        uni = overall_uni.get(target, {})
        multi = overall_multi.get(target, {})
        risk_flags = uni.get("risk_flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = [str(risk_flags)]

        lines.append(f"TARGET: {target}")
        lines.append("----------------------------------------")
        for module_name in CORE_MODULES:
            module_data = uni.get(module_name, {})
            _render_module(lines, module_name, module_data)

        lines.append("[MULTIVARIATE SIGNAL]")
        mv = dict(multi) if isinstance(multi, dict) else {}
        if "recommended_approach" not in mv:
            mv["recommended_approach"] = mv.get("recommendation", "univariate")
        for key in MULTIVARIATE_SIGNAL_KEYS + ["recommended_approach"]:
            if key not in mv:
                continue
            val = mv[key]
            if isinstance(val, (float, int, np.floating, np.integer)):
                lines.append(f"- {key}: {_fmt_num(val)}")
            else:
                lines.append(f"- {key}: {val}")
        lines.append("")

        _append_final_output_fields(lines, uni, risk_flags)
        lines.append("")

    lines.append("OVERALL BY GRANULARITY")
    for level, data in overall_by_level.items():
        if not isinstance(data, dict) or not data:
            lines.append(f"- {level}: no target output")
            continue
        for target, tdata in data.items():
            lines.append(
                f"- {level}/{target}: class={tdata.get('classification', 'Unknown')}, "
                f"score={_fmt_num(tdata.get('forecastability_score', np.nan), 2)}, "
                f"optimal_granularity={tdata.get('optimal_granularity', 'n/a')}"
            )
    lines.append("")

    lines.append("GRAIN OVERVIEW")
    lines.append(f"- total_groups={len(by_grain)}")
    shown = 0
    for grain_key, grain_data in by_grain.items():
        if shown >= 20:
            lines.append("- ... additional groups omitted for readability")
            break
        shown += 1
        if isinstance(grain_data, dict) and grain_data.get("status") == "skipped_insufficient_points":
            lines.append(f"- {grain_key}: skipped_insufficient_points (row_count={grain_data.get('row_count')})")
            continue
        lines.append(f"- {grain_key}: row_count={grain_data.get('row_count', 'n/a')}")
        rec_map = grain_data.get("recommended_approach_by_target", {})
        for target in targets:
            rec = rec_map.get(target, "n/a") if isinstance(rec_map, dict) else "n/a"
            lines.append(
                f"  target={target}, recommendation={rec}, "
                f"best_granularity={best_grain.get(target, 'n/a')}"
            )
    lines.append("")

    lines.append("SUMMARY")
    lines.append(str(report.get("summary", "")))
    lines.append("")
    lines.append("RAW STRUCTURE KEYS")
    lines.append(
        str(
            [
                "mode",
                "report_mode",
                "config",
                "overall_univariate",
                "overall_multivariate",
                "overall_by_granularity",
                "best_granularity_by_target",
                "recommended_approach_by_target",
                "by_grain",
                "summary",
            ]
        )
    )
    return "\n".join(lines)


def build_expanded_summary_report(report: dict) -> str:
    config = report.get("config", {}) if isinstance(report.get("config", {}), dict) else {}
    targets = config.get("target_cols", [])
    overall_uni = report.get("overall_univariate", {})
    overall_multi = report.get("overall_multivariate", {})

    lines: list[str] = []
    lines.append("DATA CHARACTERIZATION REPORT")
    lines.append("----------------------------------------")
    lines.append(f"Mode: {report.get('mode', 'unknown')}")
    lines.append(f"Targets: {targets}")
    lines.append("")

    for target in targets:
        uni = overall_uni.get(target, {})
        multi = overall_multi.get(target, {})
        if not isinstance(uni, dict):
            continue

        temporal = uni.get("temporal", {})
        distribution = uni.get("distribution", {})
        volatility = uni.get("volatility", {})
        structure = uni.get("structure", {})
        granularity = uni.get("granularity", {})
        stationarity = uni.get("stationarity", {})
        risk_flags = uni.get("risk_flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = [str(risk_flags)]

        acf_vals = [
            abs(float(temporal.get("acf_lag_1", np.nan))),
            abs(float(temporal.get("acf_lag_2", np.nan))),
            abs(float(temporal.get("acf_lag_3", np.nan))),
        ]
        acf_vals = [x for x in acf_vals if np.isfinite(x)]
        avg_acf = float(np.mean(acf_vals)) if acf_vals else np.nan
        signal = _signal_label(float(temporal.get("temporal_signal_strength_score", np.nan)))
        temporal_score = float(temporal.get("temporal_signal_strength_score", np.nan))
        seasonality = _seasonality_label(float(temporal.get("seasonality_strength_stl", np.nan)))
        seasonality_strength = float(temporal.get("seasonality_strength_stl", np.nan))
        driver = _resolve_driver_signal(uni, multi=multi)
        reason_items = _build_reason_block(uni, multi=multi)
        driver_classification = driver["exogenous_signal_classification"]
        driver_note = _driver_signal_note(driver_classification)
        action = _driver_consistent_recommendation(
            driver_classification=driver_classification,
            temporal_signal_score=temporal_score,
            volatility_classification=str(volatility.get("volatility_classification", "Unknown")),
        )
        recommendation_full = f"{action} because {', '.join(reason_items)}."
        noise_signal_ratio = _noise_signal_ratio(
            mean_value=float(distribution.get("mean", np.nan)),
            naive_mae_value=float(temporal.get("naive_mae", np.nan)),
        )
        predictability = _predictability_label(noise_signal_ratio)
        structure_text = _structure_summary_text(structure)

        lines.append(f"Target: {target}")
        lines.append(
            f"Signal Strength: {signal} "
            f"(temporal_signal_strength_score={_fmt_num(temporal.get('temporal_signal_strength_score', np.nan), 1)}, "
            f"avg_acf_lag_1_3={_fmt_num(avg_acf, 3)})"
        )
        lines.append(
            f"Predictability: {predictability} (noise_signal_ratio={_fmt_num(noise_signal_ratio, 2)})"
        )
        lines.append(
            f"Variance: {volatility.get('volatility_classification', 'Unknown')} "
            f"(volatility_risk_score={_fmt_num(volatility.get('volatility_risk_score', np.nan), 1)})"
        )
        lines.append(
            f"Distribution: {distribution.get('distribution_classification', 'Unknown')} "
            f"(skewness={_fmt_num(distribution.get('skewness', np.nan), 3)}, "
            f"kurtosis={_fmt_num(distribution.get('kurtosis', np.nan), 3)})"
        )
        lines.append(f"Concentration: {structure_text}")
        lines.append(
            f"Seasonality: {seasonality} "
            f"(seasonality_strength_stl={_fmt_num(temporal.get('seasonality_strength_stl', np.nan), 3)}, "
            f"seasonal_lag={_fmt_num(temporal.get('seasonal_lag', np.nan), 0)})"
        )
        signal_gain_ratio = float(granularity.get("signal_gain_ratio", np.nan))
        signal_gain_pct = (signal_gain_ratio - 1.0) * 100.0 if np.isfinite(signal_gain_ratio) else np.nan
        lines.append(
            f"Granularity: optimal={granularity.get('optimal_granularity', 'original')} "
            f"(signal_gain={_fmt_num(signal_gain_pct, 1)}%, "
            f"noise_reduction_ratio={_fmt_num(granularity.get('noise_reduction_ratio', np.nan), 3)})"
        )
        lines.append(
            f"Driver Signal: {driver_classification} "
            f"(exogenous_dominance_ratio={_fmt_num(driver['exogenous_dominance_ratio'], 2)}, "
            f"exogenous_r2={_fmt_num(driver['exogenous_r2'], 2)}, "
            f"ar_r2={_fmt_num(driver['ar_r2'], 2)})"
        )
        lines.append(f"-> {driver_note}")
        lines.append(f"CLASSIFICATION: {uni.get('classification', 'Unknown')}")
        lines.append(f"FORECASTABILITY SCORE: {_fmt_num(uni.get('forecastability_score', np.nan), 0)}/100")
        lines.append("RECOMMENDATION:")
        lines.append(f"- {action if action else 'No recommendation generated'}")
        lines.append("Reason:")
        lines.append(",\n".join(reason_items))
        start = _model_starting_point(
            driver_classification=driver_classification,
            temporal_signal_score=temporal_score,
            volatility_classification=str(volatility.get("volatility_classification", "Unknown")),
            optimal_granularity=str(granularity.get("optimal_granularity", "original")),
            seasonality_strength=seasonality_strength,
        )
        lines.append("MODEL STARTING POINT")
        lines.append("----------------------------------------")
        lines.append(f"Primary model: {start['primary_model']}")
        lines.append(f"Baseline model: {start['baseline_model']}")
        lines.append(f"Training granularity: {start['training_granularity']}")
        lines.append(f"Loss function: {start['loss_function']}")
        lines.append(f"Feature usage: {start['feature_usage']}")
        lines.append("RISK FLAGS:")
        if risk_flags:
            for item in risk_flags:
                lines.append(f"- {item}")
        else:
            lines.append("- None")
        lines.append("EXECUTIVE SUMMARY:")
        lines.append(
            f"{uni.get('classification', 'Unknown')} with forecastability_score="
            f"{_fmt_num(uni.get('forecastability_score', np.nan), 2)}."
        )
        lines.append("Key metrics:")
        lines.append(
            f"temporal_signal_strength_score={_fmt_num(temporal.get('temporal_signal_strength_score', np.nan), 3)},\n"
            f"stability_score={_fmt_num(stationarity.get('stability_score', np.nan), 3)},\n"
            f"model_improvement_ratio={_fmt_num(granularity.get('model_improvement_ratio', np.nan), 3)},\n"
            f"exogenous_dominance_ratio={_fmt_num(driver['exogenous_dominance_ratio'], 2)},\n"
            f"optimal_granularity={granularity.get('optimal_granularity', 'original')}."
        )
        lines.append("ENGINEERING DECISION RECOMMENDATION:")
        lines.append(str(uni.get("engineering_decision_recommendation", "")))
        _append_final_output_fields(
            lines,
            uni,
            risk_flags,
            modeling_recommendation_override=recommendation_full,
        )
        lines.append("")

    lines.append("SUMMARY")
    lines.append(str(report.get("summary", "")))
    return "\n".join(lines)


def build_technical_report(report: dict) -> str:
    if _is_expanded_report(report):
        return build_expanded_technical_report(report)
    return build_univariate_technical_report(report)


def build_summary_report(report: dict) -> str:
    if _is_expanded_report(report):
        return build_expanded_summary_report(report)
    return build_univariate_summary_report(report)


def build_human_readable_report(report: dict) -> str:
    return build_univariate_technical_report(report)


def build_expanded_human_readable_report(report: dict) -> str:
    return build_expanded_technical_report(report)

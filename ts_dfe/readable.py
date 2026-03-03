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


class TSDFEReport(dict):
    def __str__(self) -> str:
        readable = self.get("human_readable_report", None)
        if isinstance(readable, str) and readable:
            return readable
        return dict.__str__(self)

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


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if np.isnan(x):
        return "nan"
    return f"{100.0 * x:.{decimals}f}%"


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


def build_human_readable_report(report: dict) -> str:
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

    lines.append("[FINAL OUTPUT FIELDS]")
    lines.append(f"- classification: {report.get('classification', 'Unknown')}")
    lines.append(f"- forecastability_score: {_fmt_num(report.get('forecastability_score', np.nan), 3)}")
    lines.append(f"- modeling_recommendation: {report.get('modeling_recommendation', '')}")
    lines.append(
        f"- risk_flags_count: {len(risk_flags)}"
    )
    lines.append(f"- executive_summary: {report.get('executive_summary', '')}")
    lines.append(
        "- engineering_decision_recommendation: "
        f"{report.get('engineering_decision_recommendation', '')}"
    )

    return "\n".join(lines)


def build_expanded_human_readable_report(report: dict) -> str:
    mode = report.get("mode", "unknown")
    config = report.get("config", {}) if isinstance(report.get("config", {}), dict) else {}
    targets = config.get("target_cols", [])
    features = config.get("feature_cols", [])
    grains = config.get("grain_cols", [])
    levels = config.get("granularity_levels", [])

    overall_uni = report.get("overall_univariate", {})
    overall_multi = report.get("overall_multivariate", {})
    best_grain = report.get("best_granularity_by_target", {})
    by_grain = report.get("by_grain", {})
    overall_by_level = report.get("overall_by_granularity", {})

    lines: list[str] = []
    lines.append("TS-DFE EXPANDED REPORT")
    lines.append("----------------------------------------")
    lines.append(f"Mode: {mode}")
    lines.append(f"Targets: {targets}")
    lines.append(f"Features: {features if features else 'None'}")
    lines.append(f"Grain Columns: {grains if grains else 'None'}")
    lines.append(f"Granularity Levels: {levels if levels else 'None'}")
    lines.append("")

    lines.append("TARGET OVERVIEW")
    for target in targets:
        uni = overall_uni.get(target, {})
        multi = overall_multi.get(target, {})
        lines.append(
            f"- {target}: class={uni.get('classification', 'Unknown')}, "
            f"score={_fmt_num(uni.get('forecastability_score', np.nan), 2)}, "
            f"best_granularity={best_grain.get(target, 'n/a')}, "
            f"recommended_approach={multi.get('recommendation', 'n/a')}"
        )
        uni_temporal = (
            uni.get("temporal", {}) if isinstance(uni.get("temporal", {}), dict) else {}
        )
        uni_stationarity = (
            uni.get("stationarity", {}) if isinstance(uni.get("stationarity", {}), dict) else {}
        )
        uni_volatility = (
            uni.get("volatility", {}) if isinstance(uni.get("volatility", {}), dict) else {}
        )
        uni_distribution = (
            uni.get("distribution", {}) if isinstance(uni.get("distribution", {}), dict) else {}
        )
        risk_flags = uni.get("risk_flags", []) if isinstance(uni, dict) else []
        risk_count = len(risk_flags) if isinstance(risk_flags, list) else 0
        lines.append(
            "  "
            f"temporal_signal={_fmt_num(uni_temporal.get('temporal_signal_strength_score', np.nan), 2)}, "
            f"stability_score={_fmt_num(uni_stationarity.get('stability_score', np.nan), 2)}, "
            f"volatility_risk={_fmt_num(uni_volatility.get('volatility_risk_score', np.nan), 2)}, "
            f"distribution={uni_distribution.get('distribution_classification', 'n/a')}, "
            f"risk_flags={risk_count}"
        )
        lines.append(
            "  "
            f"cross_lag_effect={multi.get('cross_lag_effect', 'n/a')}, "
            f"residual_dependency={_fmt_num(multi.get('residual_dependency', np.nan), 3)}, "
            f"cv_improvement_multivariate={_fmt_num(multi.get('cv_improvement_multivariate', np.nan), 3)}, "
            f"feature_utility_score={_fmt_num(multi.get('feature_utility_score', np.nan), 1)}, "
            f"add_features_decision={multi.get('add_features_decision', 'n/a')}, "
            f"decision_confidence={_fmt_num(multi.get('decision_confidence', np.nan), 3)}"
        )
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
        if shown >= 12:
            lines.append("- ... additional groups omitted for readability")
            break
        shown += 1
        if isinstance(grain_data, dict) and grain_data.get("status") == "skipped_insufficient_points":
            lines.append(f"- {grain_key}: skipped_insufficient_points (row_count={grain_data.get('row_count')})")
            continue

        lines.append(f"- {grain_key}: row_count={grain_data.get('row_count', 'n/a')}")
        rec_map = grain_data.get("recommended_approach_by_target", {}) if isinstance(grain_data, dict) else {}
        best_map = grain_data.get("best_granularity_by_target", {}) if isinstance(grain_data, dict) else {}
        for target in targets:
            rec = rec_map.get(target, "n/a") if isinstance(rec_map, dict) else "n/a"
            bgr = best_map.get(target, "n/a") if isinstance(best_map, dict) else "n/a"
            lines.append(f"  target={target}, recommendation={rec}, best_granularity={bgr}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append(str(report.get("summary", "")))
    lines.append("")
    lines.append("RAW STRUCTURE KEYS")
    lines.append(
        str(
            [
                "mode",
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

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

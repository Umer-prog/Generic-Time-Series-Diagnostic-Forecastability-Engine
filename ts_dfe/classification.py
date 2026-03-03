from __future__ import annotations

import numpy as np

try:
    from .utils import clip, clip01, safe_float
except ImportError:
    from utils import clip, clip01, safe_float


def _fmt(v: float) -> str:
    x = safe_float(v, default=np.nan)
    if np.isfinite(x):
        return f"{x:.3f}"
    return "nan"


def _risk_flags(results: dict) -> list[str]:
    flags: list[str] = []
    integrity = results["integrity"]
    distribution = results["distribution"]
    temporal = results["temporal"]
    stationarity = results["stationarity"]
    volatility = results["volatility"]
    structure = results["structure"]
    granularity = results["granularity"]

    if safe_float(integrity.get("missing_timestamp_ratio", 0.0), 0.0) > 0.05:
        flags.append(
            f"Missing timestamp ratio is high: missing_timestamp_ratio={_fmt(integrity['missing_timestamp_ratio'])}."
        )
    if safe_float(integrity.get("duplicate_timestamp_ratio", 0.0), 0.0) > 0.01:
        flags.append(
            f"Duplicate timestamps present: duplicate_timestamp_ratio={_fmt(integrity['duplicate_timestamp_ratio'])}."
        )
    if safe_float(integrity.get("outlier_ratio_mad", 0.0), 0.0) > 0.08:
        flags.append(f"High robust outlier density: outlier_ratio_mad={_fmt(integrity['outlier_ratio_mad'])}.")
    if safe_float(distribution.get("tail_heaviness_index", 0.0), 0.0) > 3.0:
        flags.append(
            f"Heavy tail behavior detected: tail_heaviness_index={_fmt(distribution['tail_heaviness_index'])}."
        )
    if safe_float(temporal.get("temporal_signal_strength_score", 100.0), 100.0) < 35.0:
        flags.append(
            "Weak temporal signal: "
            f"temporal_signal_strength_score={_fmt(temporal['temporal_signal_strength_score'])}."
        )
    if stationarity.get("stationarity_classification") == "Regime-shifting":
        flags.append(
            f"Regime instability observed: cusum_max_abs={_fmt(stationarity['cusum_max_abs'])}, "
            f"rolling_mean_drift={_fmt(stationarity['rolling_mean_drift'])}."
        )
    if safe_float(volatility.get("volatility_risk_score", 0.0), 0.0) > 60.0:
        flags.append(
            f"Volatility risk is elevated: volatility_risk_score={_fmt(volatility['volatility_risk_score'])}."
        )
    if safe_float(structure.get("structure_enabled", 0.0), 0.0) > 0.5 and safe_float(
        structure.get("concentration_score", 0.0), 0.0
    ) > 70.0:
        flags.append(
            f"Structural concentration is high: concentration_score={_fmt(structure['concentration_score'])}, "
            f"top10_revenue_share_pct={_fmt(structure['top10_revenue_share_pct'])}."
        )
    if granularity.get("optimal_granularity") != "original":
        flags.append(
            "Forecastability improves with aggregation: "
            f"optimal_granularity={granularity.get('optimal_granularity')}, "
            f"signal_gain_ratio={_fmt(granularity.get('signal_gain_ratio', np.nan))}."
        )

    return flags


def _determine_classification(results: dict, forecastability_score: float) -> str:
    distribution = results["distribution"]
    temporal = results["temporal"]
    stationarity = results["stationarity"]
    volatility = results["volatility"]
    structure = results["structure"]

    if forecastability_score < 35.0:
        return "Low Forecastability"

    if (
        safe_float(structure.get("structure_enabled", 0.0), 0.0) > 0.5
        and structure.get("structure_classification") == "Dominated by few entities"
    ):
        return "Structural-Concentration Dominated"

    if stationarity.get("stationarity_classification") == "Regime-shifting":
        return "Regime-Shifting"

    if stationarity.get("stationarity_classification") == "Trend-dominated":
        return "Trend-Dominated"

    if (
        safe_float(temporal.get("seasonality_strength_stl", 0.0), 0.0) >= 0.35
        and safe_float(stationarity.get("stability_score", 0.0), 0.0) >= 60.0
    ):
        return "Seasonal Stable"

    if (
        safe_float(temporal.get("ar5_r2", 0.0), 0.0) >= 0.45
        and volatility.get("volatility_classification") == "Homoskedastic"
    ):
        return "Smooth Autoregressive"

    if (
        distribution.get("distribution_classification") in {"Zero-inflated", "Skewed transactional"}
        or volatility.get("volatility_classification") == "Event volatility"
    ):
        return "Event-Driven Transactional"

    return "Externally Driven"


def _build_modeling_recommendation(final_classification: str, results: dict) -> str:
    temporal = results["temporal"]
    stationarity = results["stationarity"]
    volatility = results["volatility"]
    granularity = results["granularity"]
    structure = results["structure"]

    gran = granularity.get("optimal_granularity")
    signal = _fmt(temporal.get("temporal_signal_strength_score", np.nan))
    stability = _fmt(stationarity.get("stability_score", np.nan))
    model_gain = _fmt(granularity.get("model_improvement_ratio", np.nan))
    seasonality = _fmt(temporal.get("seasonality_strength_stl", np.nan))
    ar5_r2 = _fmt(temporal.get("ar5_r2", np.nan))

    if final_classification == "Smooth Autoregressive":
        return (
            "Use autoregressive family models (ARIMA/ETS) at "
            f"{gran} granularity because ar5_r2={ar5_r2}, "
            f"temporal_signal_strength_score={signal}, model_improvement_ratio={model_gain}."
        )
    if final_classification == "Seasonal Stable":
        return (
            "Use seasonal models (SARIMA/TBATS/Prophet) at "
            f"{gran} granularity because seasonality_strength_stl={seasonality}, "
            f"stability_score={stability}, temporal_signal_strength_score={signal}."
        )
    if final_classification == "Trend-Dominated":
        return (
            "Use trend-aware models (differenced regression or local-trend state space) at "
            f"{gran} granularity because trend_strength_r2={_fmt(stationarity.get('trend_strength_r2', np.nan))} "
            f"and stability_score={stability}."
        )
    if final_classification == "Regime-Shifting":
        return (
            "Use rolling or regime-aware models with change-point monitoring because "
            f"cusum_max_abs={_fmt(stationarity.get('cusum_max_abs', np.nan))}, "
            f"rolling_mean_drift={_fmt(stationarity.get('rolling_mean_drift', np.nan))}, "
            f"stability_score={stability}."
        )
    if final_classification == "Event-Driven Transactional":
        return (
            "Use event/exogenous-feature models with robust loss because "
            f"volatility_classification={volatility.get('volatility_classification')}, "
            f"distribution_classification={results['distribution'].get('distribution_classification')}, "
            f"model_improvement_ratio={model_gain}."
        )
    if final_classification == "Structural-Concentration Dominated":
        return (
            "Model top entities separately and aggregate bottom-up because "
            f"top10_revenue_share_pct={_fmt(structure.get('top10_revenue_share_pct', np.nan))}, "
            f"concentration_score={_fmt(structure.get('concentration_score', np.nan))}."
        )
    if final_classification == "Low Forecastability":
        return (
            "Use conservative baselines with wide intervals and add external drivers because "
            f"temporal_signal_strength_score={signal}, stability_score={stability}, "
            f"model_improvement_ratio={model_gain}."
        )
    return (
        "Use external-regressor or causal models with frequent retraining because "
        f"temporal_signal_strength_score={signal}, volatility_risk_score={_fmt(volatility.get('volatility_risk_score', np.nan))}, "
        f"model_improvement_ratio={model_gain}."
    )


def _engineering_decision_recommendation(final_classification: str, results: dict) -> str:
    granularity = results["granularity"]
    stationarity = results["stationarity"]
    temporal = results["temporal"]
    gran = granularity.get("optimal_granularity")
    stability = _fmt(stationarity.get("stability_score", np.nan))
    signal = _fmt(temporal.get("temporal_signal_strength_score", np.nan))

    if final_classification in {"Regime-Shifting", "Low Forecastability"}:
        return (
            "Deploy a guarded baseline pipeline with rolling retrain checkpoints; "
            f"trigger investigation when stability_score<{stability} or temporal_signal_strength_score<{signal}. "
            f"Train at {gran} granularity."
        )
    return (
        "Proceed with production modeling and periodic diagnostics refresh; "
        f"train at {gran} granularity, monitor stability_score={stability} and "
        f"temporal_signal_strength_score={signal}."
    )


def synthesize(results: dict) -> dict:
    temporal = results["temporal"]
    stationarity = results["stationarity"]
    structure = results["structure"]
    granularity = results["granularity"]

    temporal_component = clip(safe_float(temporal.get("temporal_signal_strength_score", np.nan), default=0.0))
    stability_component = clip(safe_float(stationarity.get("stability_score", np.nan), default=0.0))

    if safe_float(structure.get("structure_enabled", 0.0), 0.0) > 0.5:
        concentration_component = clip(100.0 - safe_float(structure.get("concentration_score", 0.0), default=0.0))
    else:
        concentration_component = 50.0

    best_gran = granularity.get(granularity.get("optimal_granularity", "original"), {})
    noise_component = clip(
        100.0 * clip01(1.0 / (1.0 + abs(safe_float(best_gran.get("cv", np.nan), default=3.0))))
    )
    model_component = clip(100.0 * clip01(max(safe_float(granularity.get("model_improvement_ratio", 0.0), 0.0), 0.0)))

    forecastability_score = (
        (0.30 * temporal_component)
        + (0.25 * stability_component)
        + (0.15 * concentration_component)
        + (0.15 * noise_component)
        + (0.15 * model_component)
    )
    forecastability_score = clip(forecastability_score, low=0.0, high=100.0)

    final_classification = _determine_classification(results, forecastability_score=forecastability_score)
    recommendation = _build_modeling_recommendation(final_classification, results)
    risk_flags = _risk_flags(results)

    executive_summary = (
        f"{final_classification} with forecastability_score={forecastability_score:.2f}. "
        f"Key metrics: temporal_signal_strength_score={_fmt(temporal_component)}, "
        f"stability_score={_fmt(stability_component)}, "
        f"model_improvement_ratio={_fmt(granularity.get('model_improvement_ratio', np.nan))}, "
        f"optimal_granularity={granularity.get('optimal_granularity')}."
    )
    engineering_recommendation = _engineering_decision_recommendation(final_classification, results)

    return {
        "classification": final_classification,
        "forecastability_score": float(forecastability_score),
        "modeling_recommendation": recommendation,
        "risk_flags": risk_flags,
        "executive_summary": executive_summary,
        "engineering_decision_recommendation": engineering_recommendation,
    }

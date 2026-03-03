from __future__ import annotations

import pandas as pd

try:
    from .classification import synthesize
    from .distribution import analyze_distribution
    from .granularity import analyze_granularity
    from .integrity import analyze_integrity
    from .readable import TSDFEReport, build_human_readable_report
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
    from readable import TSDFEReport, build_human_readable_report
    from stationarity import analyze_stationarity
    from structure import analyze_structure
    from temporal import analyze_temporal
    from utils import build_regular_series
    from volatility import analyze_volatility


def run_ts_dfe(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    structural_cols: list[str] | None = None,
    freq: str | None = None,
) -> dict:
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

    result = {
        "integrity": modules["integrity"],
        "distribution": modules["distribution"],
        "temporal": modules["temporal"],
        "stationarity": modules["stationarity"],
        "volatility": modules["volatility"],
        "structure": modules["structure"],
        "granularity": modules["granularity"],
        "classification": synthesis["classification"],
        "forecastability_score": synthesis["forecastability_score"],
        "modeling_recommendation": synthesis["modeling_recommendation"],
        "risk_flags": synthesis["risk_flags"],
        "executive_summary": synthesis["executive_summary"],
        "engineering_decision_recommendation": synthesis["engineering_decision_recommendation"],
    }
    result["human_readable_report"] = build_human_readable_report(result)
    return TSDFEReport(result)

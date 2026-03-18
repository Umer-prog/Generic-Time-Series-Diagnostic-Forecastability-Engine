from __future__ import annotations

import numpy as np
import pandas as pd

from ts_dfe.classification import synthesize
from ts_dfe.distribution import analyze_distribution
from ts_dfe.engine import run_ts_dfe
from ts_dfe.granularity import analyze_granularity
from ts_dfe.integrity import analyze_integrity
from ts_dfe.readable import CORE_MODULES, TSDFEReport, build_human_readable_report
from ts_dfe.stationarity import analyze_stationarity
from ts_dfe.structure import analyze_structure
from ts_dfe.temporal import analyze_temporal
from ts_dfe.volatility import analyze_volatility


def _seasonal_df(n: int = 180, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    trend = 0.08 * np.arange(n)
    seasonal = 9.0 * np.sin((2.0 * np.pi * np.arange(n)) / 7.0)
    noise = rng.normal(0, 2.0, size=n)
    values = 120.0 + trend + seasonal + noise
    customers = rng.choice(["A", "B", "C", "D", "E", "F"], size=n, p=[0.28, 0.22, 0.18, 0.14, 0.1, 0.08])
    return pd.DataFrame(
        {
            "date": idx,
            "sales": values,
            "customer": customers,
            "region": rng.choice(["North", "South"], size=n, p=[0.55, 0.45]),
        }
    )


def _assert_numeric_module_fields(module_data: dict, non_numeric_keys: set[str]) -> None:
    for key, value in module_data.items():
        if key in non_numeric_keys:
            assert isinstance(value, str)
        elif isinstance(value, dict):
            _assert_numeric_module_fields(
                value,
                non_numeric_keys={"optimal_granularity"},
            )
        else:
            assert isinstance(value, (int, float, np.integer, np.floating)), f"{key} is not numeric: {type(value)}"


def test_run_ts_dfe_full_output_schema_and_ranges() -> None:
    df = _seasonal_df()
    report = run_ts_dfe(df, date_col="date", target_col="sales", structural_cols=["customer"])

    expected_top_keys = {
        "integrity",
        "distribution",
        "temporal",
        "stationarity",
        "volatility",
        "structure",
        "granularity",
        "classification",
        "forecastability_score",
        "modeling_recommendation",
        "risk_flags",
        "executive_summary",
        "engineering_decision_recommendation",
        "human_readable_report",
    }
    assert expected_top_keys.issubset(report.keys())
    assert isinstance(report, TSDFEReport)

    assert isinstance(report["classification"], str)
    assert 0.0 <= float(report["forecastability_score"]) <= 100.0
    assert isinstance(report["risk_flags"], list)
    assert isinstance(report["human_readable_report"], str)

    _assert_numeric_module_fields(report["integrity"], {"integrity_classification"})
    _assert_numeric_module_fields(report["distribution"], {"distribution_classification"})
    _assert_numeric_module_fields(report["temporal"], set())
    _assert_numeric_module_fields(report["stationarity"], {"stationarity_classification"})
    _assert_numeric_module_fields(report["volatility"], {"volatility_classification"})
    _assert_numeric_module_fields(report["structure"], {"structure_classification"})
    _assert_numeric_module_fields(report["granularity"], {"optimal_granularity"})


def test_human_readable_report_sections_and_str_representation() -> None:
    df = _seasonal_df(n=120)
    report = run_ts_dfe(df, date_col="date", target_col="sales", structural_cols=["customer"])
    text = str(report)

    assert text == report["human_readable_report"]
    assert "DATA CHARACTERIZATION REPORT" in text
    assert "CLASSIFICATION:" in text
    assert "FORECASTABILITY SCORE:" in text
    assert "RECOMMENDATION:" in text
    assert "RISK FLAGS:" in text
    assert "FULL STRUCTURED METRICS" in text
    for section in CORE_MODULES:
        assert f"[{section.upper()}]" in text


def test_integrity_detects_missing_timestamps_and_duplicates() -> None:
    idx = pd.date_range("2025-01-01", periods=40, freq="D")
    df = pd.DataFrame({"date": idx, "sales": np.linspace(10, 50, 40)})
    df = df.drop(index=[5, 9, 17]).reset_index(drop=True)
    df = pd.concat([df, df.iloc[[4]]], ignore_index=True)

    out = analyze_integrity(df, date_col="date", target_col="sales")
    assert out["missing_timestamp_ratio"] > 0
    assert out["duplicate_timestamp_ratio"] > 0
    assert out["integrity_classification"] in {"Healthy", "Monitor", "Critical"}


def test_distribution_zero_inflated_rule() -> None:
    series = pd.Series([0.0] * 70 + [1.0, 2.0, 3.0, 4.0] * 5)
    out = analyze_distribution(series)
    assert out["distribution_classification"] == "Zero-inflated"
    assert out["is_zero_inflated"] == 1.0
    assert out["zero_inflation_ratio"] > 0.30


def test_temporal_short_series_returns_nan_metrics() -> None:
    out = analyze_temporal(pd.Series([1, 2, 3, 4, 5]), freq_seconds=86400.0)
    assert np.isnan(out["temporal_signal_strength_score"])
    assert out["seasonal_lag"] == 0.0
    for i in range(1, 4):
        assert np.isnan(out[f"acf_lag_{i}"])
    assert "pacf_lag_1" not in out


def test_stationarity_trend_dominated_classification() -> None:
    x = np.arange(200, dtype=float)
    y = 10.0 + 0.8 * x + np.sin(x / 3.0) * 0.1
    out = analyze_stationarity(pd.Series(y))
    assert out["trend_strength_r2"] >= 0.40
    assert out["stationarity_classification"] == "Trend-dominated"


def test_volatility_event_classification() -> None:
    base = np.ones(120) * 100.0
    spikes = np.zeros(120)
    spikes[[10, 35, 70, 95]] = [130, -120, 140, -125]
    out = analyze_volatility(pd.Series(base + spikes))
    assert out["event_volatility_index"] > 4.0
    assert out["volatility_classification"] == "Event volatility"


def test_structure_optional_and_concentration_modes() -> None:
    df = _seasonal_df(n=120, seed=4)

    none_case = analyze_structure(df, date_col="date", target_col="sales", structural_cols=None)
    assert none_case["structure_enabled"] == 0.0
    assert none_case["structure_classification"] == "Not provided"

    # Force one dominant entity (>50% share) to trigger "Dominated by few entities".
    dom = df.copy()
    dom["customer"] = np.where(np.arange(len(dom)) < 85, "DOMINANT", dom["customer"])
    dom_case = analyze_structure(dom, date_col="date", target_col="sales", structural_cols=["customer"])
    assert dom_case["structure_enabled"] == 1.0
    assert dom_case["top1_revenue_share_pct"] >= 50.0
    assert dom_case["structure_classification"] == "Dominated by few entities"


def test_granularity_returns_all_levels_and_valid_optimal_choice() -> None:
    df = _seasonal_df(n=210)
    s = df.set_index("date")["sales"]
    out = analyze_granularity(s)

    assert {"original", "weekly", "monthly"}.issubset(out.keys())
    assert out["optimal_granularity"] in {"original", "weekly", "monthly"}
    assert isinstance(out["original"]["acf1"], float)
    assert isinstance(out["weekly"]["cv"], float)
    assert isinstance(out["monthly"]["naive_mae"], float)


def test_synthesis_and_readable_report_include_metric_grounded_recommendation() -> None:
    df = _seasonal_df(n=170, seed=22)
    report = run_ts_dfe(df, date_col="date", target_col="sales", structural_cols=["customer"])

    # Ensure synthesis contract remains intact.
    minimal = {
        "integrity": report["integrity"],
        "distribution": report["distribution"],
        "temporal": report["temporal"],
        "stationarity": report["stationarity"],
        "volatility": report["volatility"],
        "structure": report["structure"],
        "granularity": report["granularity"],
    }
    synthesis = synthesize(minimal)
    assert isinstance(synthesis["modeling_recommendation"], str)
    assert "=" in synthesis["modeling_recommendation"]

    readable = build_human_readable_report(report)
    assert "EXECUTIVE SUMMARY:" in readable
    assert "ENGINEERING DECISION RECOMMENDATION:" in readable


def test_summary_report_mode_structure() -> None:
    df = _seasonal_df(n=150, seed=31)
    report = run_ts_dfe(
        df,
        date_col="date",
        target_col="sales",
        structural_cols=["customer"],
        report_mode="summary",
    )
    text = str(report)

    assert report["report_mode"] == "summary"
    assert text == report["human_readable_report"]
    assert "DATA CHARACTERIZATION REPORT" in text
    # Classification is now the headline — appears before other detail lines.
    assert "CLASSIFICATION:" in text
    assert "FORECASTABILITY SCORE:" in text
    assert "Stationarity:" in text
    assert "MODEL STARTING POINT" in text
    assert "RISK FLAGS:" in text
    # Removed from summary mode.
    assert "Driver Signal:" not in text
    assert "[FINAL OUTPUT FIELDS]" not in text
    assert "EXECUTIVE SUMMARY:" not in text


def test_multivariate_exogenous_dominance_fields_and_technical_section() -> None:
    n = 220
    rng = np.random.default_rng(777)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    feat_1 = rng.normal(0.0, 1.0, size=n)
    feat_2 = rng.normal(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 0.25, size=n)
    target = (5.0 * feat_1) + (3.5 * feat_2) + noise

    df = pd.DataFrame(
        {
            "date": idx,
            "sales": target,
            "feat_1": feat_1,
            "feat_2": feat_2,
            "customer": rng.choice(["A", "B", "C"], size=n),
        }
    )

    report = run_ts_dfe(
        df,
        date_col="date",
        target_cols=["sales"],
        feature_cols=["feat_1", "feat_2"],
        structural_cols=["customer"],
        mode="multivariate",
        report_mode="technical",
    )

    multi = report["overall_multivariate"]["sales"]
    for key in {
        "exogenous_r2",
        "ar_r2",
        "exogenous_dominance_ratio",
        "exogenous_signal_classification",
    }:
        assert key in multi
    assert multi["exogenous_signal_classification"] in {
        "Exogenous Dominated",
        "Mixed Drivers",
        "Autoregressive Dominated",
    }

    text = str(report)
    assert "=== TECHNICAL DIAGNOSTIC REPORT ===" in text
    assert "[MULTIVARIATE SIGNAL]" in text
    assert "exogenous_dominance_ratio" in text

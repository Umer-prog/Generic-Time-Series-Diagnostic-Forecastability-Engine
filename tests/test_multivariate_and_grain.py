from __future__ import annotations

import numpy as np
import pandas as pd

from multivariate_decision import MultivariateDiagnostic
from ts_dfe.engine import run_ts_dfe


def _multivariate_df(n: int = 260, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    marketing = 20 + 2.0 * np.sin((2 * np.pi * np.arange(n)) / 30) + rng.normal(0, 1.2, n)
    temperature = 25 + 5.0 * np.sin((2 * np.pi * np.arange(n)) / 365) + rng.normal(0, 0.8, n)

    demand = 80 + 0.5 * marketing + 0.2 * temperature + rng.normal(0, 2.0, n)
    sales = 120 + 0.65 * demand + 0.4 * np.roll(marketing, 1) + rng.normal(0, 3.0, n)

    region = np.where(np.arange(n) % 2 == 0, "North", "South")
    return pd.DataFrame(
        {
            "date": idx,
            "sales": sales,
            "demand": demand,
            "marketing": marketing,
            "temperature": temperature,
            "region": region,
        }
    )


def test_multivariate_diagnostic_output_contract() -> None:
    df = _multivariate_df()
    diag = MultivariateDiagnostic(max_lag=4, cv_folds=3)
    out = diag.diagnose(
        target=df["sales"],
        exog=df[["demand", "marketing", "temperature"]],
    )

    assert {"cross_lag_effect", "residual_dependency", "cv_improvement_multivariate", "recommendation"}.issubset(out)
    assert {"feature_utility_score", "add_features_decision", "decision_confidence"}.issubset(out)
    assert out["cross_lag_effect"] in {"weak", "moderate", "strong"}
    assert isinstance(out["residual_dependency"], float)
    assert isinstance(out["cv_improvement_multivariate"], float)
    assert 0.0 <= out["feature_utility_score"] <= 100.0
    assert out["add_features_decision"] in {
        "add_features_high_impact",
        "add_features_selectively",
        "test_selected_features",
        "avoid_additional_features",
    }
    assert 0.0 <= out["decision_confidence"] <= 1.0
    assert out["recommendation"] in {"univariate", "univariate_with_exogenous", "multivariate"}
    assert "human_readable_report" in out
    assert "MULTIVARIATE DIAGNOSTIC REPORT" in str(out)


def test_engine_multivariate_mode_expanded_output() -> None:
    df = _multivariate_df()
    out = run_ts_dfe(
        df,
        date_col="date",
        target_cols=["sales", "demand"],
        feature_cols=["marketing", "temperature"],
        mode="multivariate",
        grain_cols=[],
    )

    assert out["mode"] == "multivariate"
    assert {"overall_univariate", "overall_multivariate", "overall_by_granularity", "recommended_approach_by_target"}.issubset(
        out.keys()
    )
    assert "human_readable_report" in out
    assert "TS-DFE EXPANDED REPORT" in str(out)
    assert {"sales", "demand"}.issubset(out["overall_univariate"].keys())
    assert {"sales", "demand"}.issubset(out["overall_multivariate"].keys())

    for target in ["sales", "demand"]:
        multi = out["overall_multivariate"][target]
        assert {"cross_lag_effect", "residual_dependency", "cv_improvement_multivariate", "recommendation"}.issubset(
            multi.keys()
        )
        assert multi["recommendation"] in {"univariate", "univariate_with_exogenous", "multivariate"}

    assert {"original", "W", "ME"}.issubset(out["overall_by_granularity"].keys())


def test_engine_grain_wise_output_contract() -> None:
    df = _multivariate_df(n=220)
    out = run_ts_dfe(
        df,
        date_col="date",
        target_cols=["sales", "demand"],
        feature_cols=["marketing"],
        grain_cols=["region"],
        mode="multivariate",
        min_points_per_group=40,
    )

    assert "by_grain" in out
    assert len(out["by_grain"]) >= 2
    assert "GRAIN OVERVIEW" in out["human_readable_report"]

    found_full_entry = False
    for _, grain_entry in out["by_grain"].items():
        if grain_entry.get("status") == "skipped_insufficient_points":
            continue
        found_full_entry = True
        assert {"univariate", "multivariate", "by_granularity", "recommended_approach_by_target"}.issubset(
            grain_entry.keys()
        )
        assert {"original", "W", "ME"}.issubset(grain_entry["by_granularity"].keys())
    assert found_full_entry

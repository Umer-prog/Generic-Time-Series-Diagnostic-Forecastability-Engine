import sys
from pathlib import Path

import pandas as pd

# Add project root (not ts_dfe folder)
sys.path.insert(0, r"D:/OneDrive - Global Data 365/Analytics Engine templates/tests/data tests")

from multivariate_decision import MultivariateDiagnostic
from ts_dfe.engine import run_ts_dfe


DATA_PATH = Path("preprocessed.xlsx")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} not found in current working directory.")

df = pd.read_excel(DATA_PATH)

date_candidates = ["Date", "Document Date", "date"]
target_candidates = ["Sales", "Weekly_Sales", "sales"]
feature_candidates = ["Quantity", "Unit Price", "Unit_Price", "Price"]
grain_candidates = ["Region", "Segment", "Brand"]
structural_candidates = ["Customer", "Material"]

date_col = next((c for c in date_candidates if c in df.columns), None)
target_col = next((c for c in target_candidates if c in df.columns), None)
features = [c for c in feature_candidates if c in df.columns and c != target_col]
grains = [c for c in grain_candidates if c in df.columns]  # keep one grain column for readable output
structural_cols = [c for c in structural_candidates if c in df.columns]

if not date_col or not target_col:
    raise ValueError(
        f"Could not resolve required columns. Found columns: {list(df.columns)}"
    )

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[date_col, target_col]).copy()

print("Dataset:", DATA_PATH.name, "| rows:", len(df))
print("date_col:", date_col, "| target_col:", target_col)
print("features:", features if features else "None")
print("grain_cols:", grains if grains else "None")
print("structural_cols:", structural_cols if structural_cols else "None")


# 1) Univariate baseline (legacy-compatible)
uni_report = run_ts_dfe(
    df,
    date_col=date_col,
    target_col=target_col,
    structural_cols=structural_cols
)
# print("\n=== UNIVARIATE BASELINE (HUMAN-READABLE) ===")
# print(uni_report)  # TSDFEReport string rendering

# # 2) Single-target + drivers (multivariate decision style)
# driver_report = run_ts_dfe(
#     df,
#     date_col=date_col,
#     target_cols=[target_col],
#     feature_cols=features,   # resolved existing columns, not candidate list
#     mode="multivariate",
# )
# print("\n=== MULTIVARIATE ENGINE (HUMAN-READABLE) ===")
# print(driver_report)
# print("\n=== OVERALL MULTIVARIATE DECISION (HUMAN-READABLE) ===")
# overall_multi = driver_report["overall_multivariate"][target_col]
# print(overall_multi)
# print("\n=== OVERALL MULTIVARIATE DECISION METRICS ===")
# print("cross_lag_effect:", overall_multi.get("cross_lag_effect"))
# print("residual_dependency:", overall_multi.get("residual_dependency"))
# print("cv_improvement_multivariate:", overall_multi.get("cv_improvement_multivariate"))
# print("feature_utility_score:", overall_multi.get("feature_utility_score"))
# print("add_features_decision:", overall_multi.get("add_features_decision"))
# print("decision_confidence:", overall_multi.get("decision_confidence"))
# print("recommendation:", overall_multi.get("recommendation"))


# 2) Multivariate + grain-wise engine run
multi_grain_report = run_ts_dfe(
    df,
    date_col=date_col,
    target_cols=[target_col],
    feature_cols=features,
    #grain_cols=grains,
    structural_cols=structural_cols,
    mode="multivariate",
    min_points_per_group=30,
    max_grain_groups=12,
)

print("\n=== MULTIVARIATE + GRAIN-WISE ENGINE (HUMAN-READABLE) ===")
print(multi_grain_report)
print("\n=== MULTIVARIATE + GRAIN-WISE ENGINE (RAW SNAPSHOT) ===")
print("mode:", multi_grain_report["mode"])
print("summary:", multi_grain_report["summary"])
print("recommended_approach_by_target:", multi_grain_report["recommended_approach_by_target"])
print("best_granularity_by_target:", multi_grain_report["best_granularity_by_target"])
print("overall multivariate decision:", multi_grain_report["overall_multivariate"][target_col])
print("grain groups:", len(multi_grain_report["by_grain"]))

for grain_key, grain_data in list(multi_grain_report["by_grain"].items())[:5]:
    if grain_data.get("status") == "skipped_insufficient_points":
        print(f"{grain_key} -> skipped_insufficient_points")
        continue
    rec = grain_data["recommended_approach_by_target"][target_col]
    score = grain_data["univariate"][target_col]["forecastability_score"]
    opt_grain = grain_data["best_granularity_by_target"][target_col]
    print(f"{grain_key} -> recommendation={rec}, score={score:.2f}, best_granularity={opt_grain}")


# 3) Standalone MultivariateDiagnostic check on daily aggregated data
print("\n=== STANDALONE MULTIVARIATE DIAGNOSTIC ===")
diag = MultivariateDiagnostic(max_lag=5, cv_folds=3)

agg_map = {target_col: "sum"}
for col in features:
    agg_map[col] = "mean"
daily = df.groupby(date_col).agg(agg_map).sort_index()

decision = diag.diagnose(
    target=daily[target_col],
    exog=daily[features] if features else pd.DataFrame(index=daily.index),
)
print(decision)
print("standalone feature_utility_score:", decision.get("feature_utility_score"))
print("standalone add_features_decision:", decision.get("add_features_decision"))
print("standalone decision_confidence:", decision.get("decision_confidence"))

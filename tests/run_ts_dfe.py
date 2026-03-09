import sys
from pathlib import Path

import pandas as pd

# Add project root (not ts_dfe folder)
sys.path.insert(0, r"D:/OneDrive - Global Data 365/Analytics Engine templates/tests/data tests")

from multivariate_decision import MultivariateDiagnostic
from ts_dfe.engine import _prepare_multivariate_timeseries, run_ts_dfe


DATA_PATH = Path("preprocessed.xlsx")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} not found in current working directory.")

df = pd.read_excel(DATA_PATH)
#df = pd.read_csv(DATA_PATH)

date_candidates = ["Date", "Document Date", "date"]
target_candidates = ["Sales", "Weekly_Sales", "sales"]
feature_candidates = ["Region", "Segment", "Brand", "Brand"] #["Holiday_flag","Temperature", "Fuel_Price", "CPI", "Unemployment"] #["Quantity", "Unit Price", "Unit_Price", "Price", "Sales Type"]
grain_candidates =  ["Inventory Location ID"]#["Region", "Segment", "Brand"]
structural_candidates = ["Customer", "Material"]

date_col = next((c for c in date_candidates if c in df.columns), None)
target_col = next((c for c in target_candidates if c in df.columns), None)
features = list(dict.fromkeys([c for c in feature_candidates if c in df.columns and c != target_col]))
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


def print_multi_signal_metrics(title: str, multi_block: dict) -> None:
    print(f"\n=== {title} ===")
    print("cross_lag_effect:", multi_block.get("cross_lag_effect"))
    print("residual_dependency:", multi_block.get("residual_dependency"))
    print("cv_improvement_multivariate:", multi_block.get("cv_improvement_multivariate"))
    print("feature_utility_score:", multi_block.get("feature_utility_score"))
    print("decision_confidence:", multi_block.get("decision_confidence"))
    print("exogenous_r2:", multi_block.get("exogenous_r2"))
    print("ar_r2:", multi_block.get("ar_r2"))
    print("exogenous_dominance_ratio:", multi_block.get("exogenous_dominance_ratio"))
    print("exogenous_signal_classification:", multi_block.get("exogenous_signal_classification"))
    print("recommendation:", multi_block.get("recommendation"))


# 1) Univariate technical mode
uni_technical = run_ts_dfe(
    df,
    date_col=date_col,
    target_col=target_col,
    structural_cols=structural_cols,
    report_mode="technical",
)
# print("\n=== UNIVARIATE | TECHNICAL MODE ===")
# print(uni_technical)


# 1b) Univariate summary mode
uni_summary = run_ts_dfe(
    df,
    date_col=date_col,
    target_col=target_col,
    structural_cols=structural_cols,
    report_mode="summary",
)
print("\n=== UNIVARIATE | SUMMARY MODE ===")
print(uni_summary)


# 2) Multivariate technical mode (single target + features)
multi_technical = run_ts_dfe(
    df,
    date_col=date_col,
    target_cols=[target_col],  # single target
    feature_cols=features,
    mode="multivariate",
    structural_cols=structural_cols,
    report_mode="technical",
)
# print("\n=== MULTIVARIATE | TECHNICAL MODE ===")
# print(multi_technical)
# multi_technical_signal = multi_technical["overall_multivariate"][target_col]
# print_multi_signal_metrics("MULTIVARIATE SIGNAL METRICS | TECHNICAL", multi_technical_signal)


# 2b) Multivariate summary mode (single target + features)
multi_summary = run_ts_dfe(
    df,
    date_col=date_col,
    target_cols=[target_col],
    feature_cols=features,
    structural_cols=structural_cols,
    mode="multivariate",
    report_mode="summary",
)
print("\n=== MULTIVARIATE | SUMMARY MODE ===")
print(multi_summary)
multi_summary_signal = multi_summary["overall_multivariate"][target_col]
print_multi_signal_metrics("MULTIVARIATE SIGNAL METRICS | SUMMARY", multi_summary_signal)


# 3) Optional: multivariate + grain-wise technical run
# multi_grain_report = run_ts_dfe(
#     df,
#     date_col=date_col,
#     target_cols=[target_col],
#     feature_cols=features,
#     grain_cols=grains,
#     structural_cols=structural_cols,
#     mode="multivariate",
#     report_mode="technical",
#     min_points_per_group=30,
#     max_grain_groups=12,
# )
# print("\n=== MULTIVARIATE + GRAIN-WISE | TECHNICAL MODE ===")
# print(multi_grain_report)
# print("grain groups:", len(multi_grain_report["by_grain"]))


# 4) Standalone MultivariateDiagnostic check
print("\n=== STANDALONE MULTIVARIATE DIAGNOSTIC ===")
diag = MultivariateDiagnostic(max_lag=5, cv_folds=3)

daily, data_context = _prepare_multivariate_timeseries(
    df=df,
    date_col=date_col,
    target_col=target_col,
    feature_cols=features,
)

decision = diag.diagnose(
    target=daily[target_col],
    exog=daily[features] if features else pd.DataFrame(index=daily.index),
    data_context=data_context,
)
print(decision)
print_multi_signal_metrics("STANDALONE MULTIVARIATE SIGNAL METRICS", decision)

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multivariate_decision import MultivariateDiagnostic
from ts_dfe.engine import _prepare_multivariate_timeseries, run_ts_dfe

DEFAULT_CONFIG = Path(__file__).with_name("data_profiles.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TS-DFE using a JSON data profile."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to profile JSON file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="sales_xlsx",
        help="Profile name inside JSON config.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit.",
    )
    return parser.parse_args()


def load_profiles(config_path: Path) -> dict[str, dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "profiles" in raw:
        profiles = raw["profiles"]
    else:
        profiles = raw

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Profile JSON must be a non-empty object of profiles.")
    return profiles


def resolve_data_path(raw_path: str, config_path: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    local = Path.cwd() / p
    if local.exists():
        return local
    return config_path.parent / p


def pick_column(df: pd.DataFrame, fixed: str | None, candidates: list[str]) -> str | None:
    if fixed:
        return fixed if fixed in df.columns else None
    return next((c for c in candidates if c in df.columns), None)


def read_data(
    data_path: Path,
    file_type: str | None,
    sheet_name: str | None,
    read_kwargs: dict[str, Any] | None,
) -> pd.DataFrame:
    ext = data_path.suffix.lower()
    detected_type = file_type or ("csv" if ext == ".csv" else "excel")
    kwargs = dict(read_kwargs or {})
    if detected_type == "csv":
        return pd.read_csv(data_path, **kwargs)
    if sheet_name is None:
        return pd.read_excel(data_path, **kwargs)
    return pd.read_excel(data_path, sheet_name=sheet_name, **kwargs)


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


def run_profile(profile_name: str, profile: dict[str, Any], config_path: Path) -> None:
    data_path_raw = profile.get("data_path")
    if not data_path_raw:
        raise ValueError(f"Profile '{profile_name}' is missing 'data_path'.")

    data_path = resolve_data_path(data_path_raw, config_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found for profile '{profile_name}': {data_path}")

    file_type = profile.get("file_type")
    sheet_name = profile.get("sheet_name")
    read_kwargs = profile.get("read_kwargs")
    if read_kwargs is not None and not isinstance(read_kwargs, dict):
        raise ValueError(f"Profile '{profile_name}' has invalid read_kwargs. It must be an object.")

    df = read_data(
        data_path,
        file_type=file_type,
        sheet_name=sheet_name,
        read_kwargs=read_kwargs,
    )

    date_col = pick_column(df, profile.get("date_col"), profile.get("date_candidates", []))
    target_col = pick_column(df, profile.get("target_col"), profile.get("target_candidates", []))
    feature_candidates = profile.get("feature_cols") or profile.get("feature_candidates", [])
    grain_candidates = profile.get("grain_cols") or profile.get("grain_candidates", [])
    structural_candidates = profile.get("structural_cols") or profile.get("structural_candidates", [])

    features = list(dict.fromkeys([c for c in feature_candidates if c in df.columns and c != target_col]))
    grains = [c for c in grain_candidates if c in df.columns]
    structural_cols = [c for c in structural_candidates if c in df.columns]

    if not date_col or not target_col:
        raise ValueError(
            f"Could not resolve required columns for profile '{profile_name}'. "
            f"date_col={date_col}, target_col={target_col}. "
            f"Columns found: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[date_col, target_col]).copy()
    if df.empty:
        raise ValueError(f"Profile '{profile_name}' produced an empty dataframe after cleaning.")

    print("Profile:", profile_name)
    print("Dataset:", data_path.name, "| rows:", len(df))
    print("date_col:", date_col, "| target_col:", target_col)
    print("features:", features if features else "None")
    print("grain_cols:", grains if grains else "None")
    print("structural_cols:", structural_cols if structural_cols else "None")

    uni_summary = run_ts_dfe(
        df,
        date_col=date_col,
        target_col=target_col,
        structural_cols=structural_cols,
        report_mode="summary",
    )
    print("\n=== UNIVARIATE | SUMMARY MODE ===")
    print(uni_summary)

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


def main() -> int:
    args = parse_args()
    try:
        profiles = load_profiles(args.config)
        if args.list_profiles:
            print("Available profiles:")
            for name in sorted(profiles.keys()):
                print("-", name)
            return 0

        profile = profiles.get(args.profile)
        if not profile:
            raise KeyError(
                f"Profile '{args.profile}' not found in {args.config}. "
                f"Available: {', '.join(sorted(profiles.keys()))}"
            )
        run_profile(args.profile, profile, args.config)
        return 0
    except Exception as exc:
        print("ERROR:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# TS-DFE

Generic Time Series Diagnostic & Forecastability Engine with 8 modular layers:

1. `integrity.py` - Data Integrity & Structural Health
2. `distribution.py` - Distribution & Tail Analysis
3. `temporal.py` - Temporal Dependence Diagnostics
4. `stationarity.py` - Stationarity & Regime Stability
5. `volatility.py` - Volatility & Heteroskedasticity
6. `structure.py` - Structural Concentration (Optional)
7. `granularity.py` - Granularity & Forecastability Benchmarking
8. `classification.py` - Classification & Recommendation Engine

Entry point:

- `ts_dfe/engine.py` -> `run_ts_dfe(...)`
- Detailed architecture: `ts_dfe/FUNCTIONAL_ARCHITECTURE.md`
- External multivariate decision helper: `multivariate_decision/diagnostic.py`

Common usage modes:

1. Backward-compatible univariate mode:
   - `run_ts_dfe(df, date_col="date", target_col="sales", structural_cols=["customer"])`
2. Expanded multivariate mode:
   - `run_ts_dfe(df, date_col="date", target_cols=["sales", "demand"], feature_cols=["marketing"], mode="multivariate")`
3. Grain-wise mode:
   - `run_ts_dfe(df, date_col="date", target_cols=["sales"], grain_cols=["region"], mode="multivariate")`

Expanded output sections (non-legacy mode):

- `overall_univariate`
- `overall_multivariate`
- `overall_by_granularity`
- `by_grain`
- `recommended_approach_by_target`
- `best_granularity_by_target`

Output keys:

- `integrity`
- `distribution`
- `temporal`
- `stationarity`
- `volatility`
- `structure`
- `granularity`
- `classification`
- `forecastability_score`
- `modeling_recommendation`
- `risk_flags`
- `executive_summary`
- `engineering_decision_recommendation`

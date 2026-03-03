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

- `engine.py` -> `run_ts_dfe(df, date_col, target_col, structural_cols=None, freq=None)`
- Detailed function-by-function architecture: `FUNCTIONAL_ARCHITECTURE.md`

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

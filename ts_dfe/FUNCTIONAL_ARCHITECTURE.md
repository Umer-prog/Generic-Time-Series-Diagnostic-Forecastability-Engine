# TS-DFE Functional Architecture (Code-Level)

This document explains the current implementation of `ts_dfe` in functional detail:
- Every file and function
- End-to-end execution flow
- Scoring formulas and decision rules
- Input/output contracts
- Edge-case behavior

The goal is to make it easy to reason about behavior, validate outputs, and extend the code safely.

---

## 0. Expanded Mode

The engine supports two runtime shapes:

1. **Legacy single-target path** (backward-compatible):
   - Triggered when one target is provided and no multivariate/grain features are requested.
   - Returns the same `TSDFEReport` contract as before.
2. **Expanded path**:
   - Supports `target_cols`, `feature_cols`, `grain_cols`, `granularity_levels`, and `mode`.
   - Returns structured sections:
     - `overall_univariate`
     - `overall_multivariate`
     - `overall_by_granularity`
     - `by_grain`
     - `best_granularity_by_target`
     - `recommended_approach_by_target`

External module:
- `multivariate_decision/diagnostic.py`
- Class: `MultivariateDiagnostic`
- Methods: `granger_test`, `residual_corr_test`, `cv_comparison`, `structural_violation_check`, `diagnose`

---

## 1. High-Level Design

`TS-DFE` is a modular diagnostics engine for univariate time series (with optional structural/entity columns).
It computes:
- Module-level numeric diagnostics
- Rule-based classifications
- Composite forecastability score
- Actionable recommendation strings tied to metric evidence
- Human-readable report text

Core pipeline:
1. Validate and regularize time series
2. Compute 7 module diagnostics
3. Synthesize final classification and score
4. Produce human-readable reporting layer

Primary entrypoint: `run_ts_dfe(...)` in `engine.py`
Primary output object: `TSDFEReport` (dict-like with `__str__` rendering)

---

## 2. End-to-End Execution Flow

Call flow in production:

1. `run_ts_dfe(df, date_col, target_col, structural_cols=None, freq=None)`
2. `analyze_integrity(...)` on raw cleaned data
3. `build_regular_series(...)` to create regular indexed series
4. Fill gaps for model modules via `interpolate(method="time").ffill().bfill()`
5. Run:
   - `analyze_distribution(...)`
   - `analyze_temporal(...)`
   - `analyze_stationarity(...)`
   - `analyze_volatility(...)`
   - `analyze_structure(...)`
   - `analyze_granularity(...)`
6. Merge module outputs into `modules` dict
7. `synthesize(modules)` for:
   - `classification`
   - `forecastability_score`
   - `modeling_recommendation`
   - `risk_flags`
   - `executive_summary`
   - `engineering_decision_recommendation`
8. `build_summary_report(result)` or `build_technical_report(result)` depending on `report_mode`
9. Wrap final dict in `TSDFEReport` and return

Notes:
- Integrity uses raw grouped series with missing periods preserved in expected-index comparison.
- Most statistical modules operate on gap-filled regularized series to avoid failures from missing observations.
- Structural module is optional and enabled only when `structural_cols` exists in dataframe.

---

## 3. Output Contract

Top-level returned keys from `run_ts_dfe`:

```
integrity            dict — data quality metrics
distribution         dict — distribution shape metrics
temporal             dict — ACF, AR, seasonality metrics
stationarity         dict — ADF/KPSS, drift, CUSUM, stability score
volatility           dict — ARCH, clustering, event index
structure            dict — entity concentration (disabled if no structural_cols)
granularity          dict — original/weekly/monthly comparison
classification       str  — regime label
forecastability_score float
modeling_recommendation str
risk_flags           list[str]
executive_summary    str
engineering_decision_recommendation str
human_readable_report str
report_mode          str  — "summary" or "technical"
```

Type guarantees:
- All metric fields are numeric (`float`/`int`).
- Classification and recommendation fields are strings.
- `risk_flags` is a list of strings.

---

## 4. File-by-File and Function-by-Function

## 4.1 `engine.py`

### `run_ts_dfe(df, date_col, target_col, structural_cols=None, freq=None, report_mode="technical") -> TSDFEReport`

Purpose: Orchestrates full TS-DFE pipeline.

Inputs:
- `df`: source DataFrame
- `date_col`: datetime-like column name
- `target_col`: numeric target column name
- `structural_cols`: optional entity columns
- `freq`: optional forced frequency alias
- `report_mode`: `"summary"` or `"technical"` (default `"technical"`)

Steps:
1. Normalizes `structural_cols`.
2. Calls `analyze_integrity` (raw structure health).
3. Calls `build_regular_series` from utils (group/sort/reindex frequency).
4. Creates model-ready series via interpolation + forward/backward fill.
5. Calls all module analyzers.
6. Builds `modules` dict.
7. Calls `synthesize(modules)` for global decisions.
8. Assembles final dict.
9. Generates `human_readable_report` via `build_summary_report` or `build_technical_report`.
10. Returns `TSDFEReport(result)`.

Edge behavior:
- Missing/invalid dates and target values are dropped in preprocessing.
- If `structural_cols` are absent or invalid, structure module returns disabled output.

---

## 4.2 `utils.py` (Shared Foundations)

### `SeriesBundle` dataclass
Fields: `series`, `freq_alias`, `freq_seconds`

### `safe_float(value, default=np.nan) -> float`
Safe numeric cast with finite check. Returns `default` for None, errors, or inf/nan.

### `safe_div(numerator, denominator, default=np.nan) -> float`
Safe division with zero/invalid denominator handling.

### `clip(value, low=0.0, high=100.0) -> float`
Numeric clamp used in scoring.

### `clip01(value) -> float`
Convenience clamp into `[0, 1]`.

### `prepare_dataframe(df, date_col, target_col) -> pd.DataFrame`
Canonical cleaning step: column checks, coerce date/target, drop invalids, sort by date.

### `seconds_to_alias(freq_seconds) -> Optional[str]`
Convert seconds interval to pandas alias (`D`, `H`, `T`, `S` forms).

### `infer_frequency(index) -> (alias, seconds)`
Infer frequency from datetime index. Tries `pd.infer_freq`, converts offset, falls back to median delta.

### `resolve_frequency(index, freq) -> (alias, seconds)`
Prefer explicit `freq`, otherwise infer.

### `build_expected_index(index, freq_alias, freq_seconds) -> pd.DatetimeIndex`
Build ideal regular index from min→max datetime bounds.

### `build_regular_series(df, date_col, target_col, freq=None, agg="sum") -> SeriesBundle`
Convert raw data to a regular time series: prepare → group → resolve freq → reindex.

### `rolling_window_size(n, min_size=5, frac=0.15, max_size=60) -> int`
Consistent adaptive rolling window size.

### `lagged_design(series, lags) -> (X, y)`
Build supervised matrix for AR-style regression.

### `ols_predictions(X, y) -> (pred, intercept)`
Fit OLS via `np.linalg.lstsq`, return predictions.

### `ar_r2(series, lags) -> float`
Compute in-sample AR(lags) R².

### `ar_mae(series, lags) -> float`
Compute in-sample AR(lags) MAE.

### `naive_mae(series) -> float`
One-step naive forecast MAE (`y_t-1` as prediction).

### `manual_acf(series, max_lag) -> dict`
ACF fallback when statsmodels unavailable or fails.

### `detect_seasonal_lag(freq_seconds) -> int`
Heuristic seasonal lag from sampling frequency:
- Intra-day: day-length lag
- Daily-ish: `7`
- Weekly-ish: `52`
- Monthly-ish: `12`
- Otherwise: `0`

### `trend_r2(series) -> float`
Linear trend explanatory strength using normalized time index.

---

## 4.3 `integrity.py` (Data Integrity & Structural Health)

### `analyze_integrity(df, date_col, target_col, freq=None) -> dict`

Output metrics:
- `row_count`, `observation_count`
- `missing_timestamp_ratio` — expected vs actual timestamps
- `duplicate_timestamp_ratio`
- `gap_ratio` — irregular spacing from modal delta
- `zero_ratio`, `negative_ratio`
- `outlier_ratio_iqr` — outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
- `outlier_ratio_mad` — robust z-score threshold `|z| > 3.5`
- `rolling_mean_shift_max`
- `structural_break_count` — rolling mean spikes above `median + 2×std`
- `coverage_duration_days`, `frequency_seconds`
- `integrity_score` — `100 × (1 - weighted_penalty)`, clipped to `[0, 100]`
- `integrity_classification` — `Healthy` (≥85), `Monitor` (≥65), else `Critical`

---

## 4.4 `distribution.py` (Distribution & Tail Analysis)

### `analyze_distribution(series) -> dict`

Output metrics:
- `mean`, `std`, `cv`
- `skewness`, `kurtosis`
- `p99_median_ratio`
- `tail_heaviness_index` — `(p99−p95)/(p75−p50)`
- `zero_inflation_ratio`
- Binary flags: `is_gaussian_like`, `is_heavy_tailed`, `is_zero_inflated`, `is_skewed_transactional`
- `distribution_classification`

Classification precedence:
1. Zero-inflated (`zero_ratio > 0.30`)
2. Heavy-tailed (`kurtosis > 3` OR `p99_median_ratio > 8` OR `tail_heaviness_index > 3`)
3. Skewed transactional (`|skew| > 1.0`)
4. Gaussian-like (fallback)

Note: `symmetry_score`, `spike_frequency_95`, and `distribution_risk_score` were removed — they were computed but never used in any decision logic or display.

---

## 4.5 `temporal.py` (Temporal Dependence Diagnostics)

Dependencies: Uses statsmodels if available for ACF/STL; falls back to internal methods.

### `_acf_with_fallback(series, max_lag=3) -> dict`
Returns lagged ACF values for lags 1–3 using statsmodels or manual fallback.

### `analyze_temporal(series, freq_seconds) -> dict`

Output metrics:
- `observation_count`
- `acf_lag_1`, `acf_lag_2`, `acf_lag_3` — only lags 1–3 (lags 4–10 and all PACF removed)
- `ar1_r2`, `ar5_r2`
- `seasonal_lag`, `seasonal_acf`
- `seasonality_strength_stl`
- `naive_mae`, `ar5_mae`
- `autoregressive_improvement_over_naive`
- `temporal_signal_strength_score` — weighted: `|acf1|`, best AR R², STL strength, AR vs naive improvement; scaled to `[0, 100]`

Short-series behavior: if `n < 8`, returns numeric NaNs and `seasonal_lag = 0`.

Removed metrics: `pacf_lag_1..10` (were stored but never used downstream), `ljung_box_pvalue_lag10`, `acf_lag_4..10`.

---

## 4.6 `stationarity.py` (Stationarity & Regime Stability)

### `analyze_stationarity(series) -> dict`

Output metrics:
- `adf_pvalue`, `kpss_pvalue`
- `rolling_mean_drift`, `rolling_variance_drift`
- `cusum_max_abs` — cumulative sum of standardized series
- `trend_strength_r2`
- `stability_score` — starts at 100 with penalties for failing tests, drift, high CUSUM, strong trend
- `stationarity_classification`

Classification rules (in priority order):
1. `Stationary` if ADF/KPSS/stability thresholds all satisfied
2. `Trend-dominated` if `trend_strength_r2 >= 0.40`
3. `Regime-shifting` if CUSUM or drift thresholds breached
4. `Drift-prone` otherwise

---

## 4.7 `volatility.py` (Volatility & Heteroskedasticity)

### `analyze_volatility(series) -> dict`

Output metrics:
- `observation_count`
- `rolling_std_mean_corr`
- `arch_test_pvalue`
- `volatility_clustering_index` — ACF lag1 of squared changes
- `event_volatility_index` — `p95(|delta|)/p50(|delta|)`
- `volatility_risk_score` — weighted from corr, clustering, ARCH flag, event index; `[0, 100]`
- `volatility_classification`

Classification rules (priority order):
1. `Event volatility` if `event_volatility_index > 4`
2. `Clustered volatility` if ARCH p<0.05 and clustering>0.25
3. `Heteroskedastic` if ARCH significant OR strong rolling corr
4. `Homoskedastic` otherwise

Removed metric: `cv_across_time_windows` — computed segment CV but was not used in classification, scoring, or risk flags.

---

## 4.8 `structure.py` (Structural Concentration)

### `_gini(values) -> float`
Gini concentration from non-negative finite shares.

### `_normalized_entropy(prob) -> float`
Entropy normalized to `[0, 1]`.

### `analyze_structure(df, date_col, target_col, structural_cols=None) -> dict`

Disabled behavior: if no valid `structural_cols`, returns `structure_enabled=0` with default metrics.

Enabled output metrics:
- `entity_count` — unique entity count from totals (no longer uses expensive pivot)
- `top1_revenue_share_pct`, `top5_revenue_share_pct`, `top10_revenue_share_pct`
- `gini_coefficient`, `herfindahl_index`, `entropy_normalized`
- `concentration_score`
- `structure_classification` — `Dominated by few entities`, `Concentrated`, or `Diversified`
- `structure_enabled`

Classification rules:
- `Dominated by few entities` if very high top-share/HHI
- `Concentrated` for moderate concentration thresholds
- `Diversified` otherwise

Removed metrics: `volatility_contribution_top_decile`, `customer_churn_volatility` — were never used in scoring, risk flags, or classification. The expensive `by_time_entity` pivot they required was also removed.

---

## 4.9 `granularity.py` (Granularity & Forecastability Benchmarking)

### `_metrics_for_series(series) -> dict`

Output metrics:
- `count`, `cv`, `acf1`, `naive_mae`, `ar1_mae`
- `model_improvement_ratio` — `(naive_mae - ar_mae) / naive_mae`
- `granularity_score` — mix of signal (`acf1`), low-noise proxy (inverse `cv`), AR improvement; `[0, 100]`

Note: `mean_forecast_mae` removed — it was computed and returned but not used in `granularity_score` or any downstream logic.

### `analyze_granularity(regular_series) -> dict`

Flow:
1. Compute base, weekly (`W`), monthly (`ME`) series.
2. Compute metrics for each via `_metrics_for_series`.
3. Select max `granularity_score` as `optimal_granularity`.
4. Compute `signal_gain_ratio`, `noise_reduction_ratio`, `model_improvement_ratio`.

Note: `optimal_granularity_score` removed from return — it was the winner's internal score and was not used after selection.

---

## 4.10 `classification.py` (Final Synthesis Engine)

### `_fmt(v) -> str`
Local numeric formatter for recommendation strings.

### `_risk_flags(results) -> list[str]`
Converts threshold breaches into explicit risk flag messages:
- Missing timestamp ratio > 5%
- Duplicate timestamps > 1%
- Robust outlier density > 8%
- Heavy tail behavior
- Weak temporal signal (< 35)
- Regime instability (Regime-shifting classification)
- Volatility risk > 60
- High concentration (enabled + concentration_score > 70)
- Signal stronger at aggregated granularity (informational, not alarming)

### `_determine_classification(results, forecastability_score) -> str`

Nine class outcomes (priority order, first match wins):

1. `Low Forecastability` — score < 35
2. `Structural-Concentration Dominated` — structure enabled AND `"Dominated by few entities"`
3. `Regime-Shifting` — stationarity_classification == `"Regime-shifting"`
4. `Trend-Dominated` — stationarity_classification == `"Trend-dominated"`
5. `Seasonal Stable` — seasonality_strength ≥ 0.35 AND stability_score ≥ 60
6. `Smooth Autoregressive` — ar5_r2 ≥ 0.45 AND volatility == `"Homoskedastic"`
7. `Intermittent Demand` — Zero-inflated AND NOT Event volatility AND event_volatility_index ≤ 3.0
8. `Event-Driven Transactional` — Skewed transactional OR Event volatility
9. `Externally Driven` — fallback

### `_build_modeling_recommendation(final_classification, results) -> str`
Metric-cited recommendation text for each class. All recommendations include `key=value` evidence.

Intermittent Demand recommendation: `"Use intermittent demand models (Croston, SBA, or ADIDA) at {gran} granularity because zero_inflation_ratio=X, ..."`

### `_engineering_decision_recommendation(final_classification, results) -> str`
Engineering-action framing (production vs guarded baseline posture).

Intermittent Demand: `"Proceed with intermittent demand pipeline; train at {gran} granularity, monitor demand-interval accuracy and zero-ratio stability."`

### `synthesize(results) -> dict`

Forecastability score formula:
```
0.30 × temporal_signal_strength_score
0.25 × stability_score
0.15 × concentration_component  (inverted if structure enabled, else 50)
0.15 × noise_component          (100 × 1/(1+|CV|))
0.15 × model_improvement_ratio  (AR vs naive gain, clipped to [0,1])
```
Clipped to `[0, 100]`.

Returned fields:
- `classification`, `forecastability_score`, `modeling_recommendation`
- `risk_flags`, `executive_summary`, `engineering_decision_recommendation`

---

## 4.11 `readable.py` (Presentation Layer)

### `TSDFEReport(dict)`
Dict-compatible container with human-readable `__str__` and `__repr__`.
Printing the object renders `human_readable_report`.

### Report modes

**`build_summary_report(result) -> str`**
Dispatches to `build_univariate_summary_report` (single target) or `build_expanded_summary_report` (multivariate/grain).

**`build_technical_report(result) -> str`**
Dispatches to `build_univariate_technical_report` or `build_expanded_technical_report`.

**`build_human_readable_report(report) -> str`**
Thin wrapper for `build_univariate_technical_report` — retained for backward compatibility with direct callers.

### Summary mode structure (`build_univariate_summary_report`)

```
DATA CHARACTERIZATION REPORT
[series info]

CLASSIFICATION: {regime}          ← HEADLINE (line 1 of content)
FORECASTABILITY SCORE: X/100
Key drivers: temporal=X, stability=X, model_improvement=X

Stationarity: {label} (stability_score=X)
Signal Strength: {label} (temporal_score=X, avg_acf_1_3=X)
Distribution: {label} (zero_inflation_ratio=X%, skewness=X)
Volatility: {label}
Seasonality: {label} (strength=X, lag=Y)
Granularity: optimal=X (signal_gain=Y%)
Concentration: {text}

RECOMMENDATION: {action}
Reason: {stationarity_class, volatility_class, dist_class, model_improvement}

MODEL STARTING POINT
  Primary model: X
  Training granularity: X
  Loss function: X

RISK FLAGS:
  • ...
```

Items removed from summary mode: Predictability/noise_signal_ratio, Driver Signal section, Effective observations (monthly), baseline_model, feature_usage, EXECUTIVE SUMMARY, Key metrics block, ENGINEERING DECISION RECOMMENDATION, [FINAL OUTPUT FIELDS].

### Technical mode structure (`build_univariate_technical_report`)
Includes everything above plus full structured metrics per module and final output fields.

---

## 4.12 `__init__.py`

Public API exports:
- `run_ts_dfe`
- `build_human_readable_report`
- `TSDFEReport`

---

## 5. Statistical Dependencies and Fallback Strategy

Optional dependency: `statsmodels`

Where used:
- Temporal: ACF/STL
- Stationarity: ADF/KPSS
- Volatility: ARCH LM test

Fallback behavior:
- Manual ACF and internal regressions keep engine operational without statsmodels.
- Some p-values and seasonality metrics become `NaN` when tests are unavailable.

---

## 6. Data and Modeling Assumptions

1. Target is numeric and aggregatable by date.
2. Time series can be mapped to a regular grid for diagnostics.
3. AR diagnostics are in-sample fit quality indicators, not production forecast backtests.
4. Rule-based classification is deterministic and threshold-driven.
5. Structure diagnostics assume non-negative additive contribution logic (shares/concentration).
6. Negative values in the target column are flagged as a data quality signal but not excluded from computation (negatives may be semantically valid, e.g. inventory issues/write-backs).

---

## 7. Extension Guidance (Safe Change Points)

Recommended extension locations:
- Add new metric helpers in `utils.py`.
- Add new module metrics in module analyzer return dicts.
- Update scoring weights in module-specific score formulas and `classification.synthesize` composite.
- Add new risk flags in `_risk_flags`.
- Add new final classes in `_determine_classification` with corresponding entries in `_build_modeling_recommendation` and `_engineering_decision_recommendation`.
- Expand readable report sections in `build_univariate_summary_report` and `build_univariate_technical_report`.

Change discipline:
1. Preserve output key stability where downstream users depend on keys.
2. Keep all metrics numeric.
3. Keep recommendation text metric-grounded (include `key=value` evidence).
4. Add/adjust tests in `tests/test_ts_dfe.py` for every rule change.

---

## 8. Traceability Checklist (Input → Decision)

For a single run:
1. Input row acceptance: `prepare_dataframe`
2. Temporal regularization: `build_regular_series`
3. Quality diagnostics: `integrity`
4. Signal/noise/statistical diagnostics: 5 core modules (distribution, temporal, stationarity, volatility, granularity)
5. Structural concentration diagnostics if entity columns exist: `structure`
6. Granularity comparison: original / weekly / monthly
7. Final score and class: weighted deterministic synthesis in `classification.synthesize`
8. Reporting: `build_summary_report` or `build_technical_report`

This checklist can be used for debugging, audits, and stakeholder walkthroughs.

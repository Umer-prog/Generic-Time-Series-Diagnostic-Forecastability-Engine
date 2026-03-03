# TS-DFE Functional Architecture (Code-Level)

This document explains the current implementation of `ts_dfe` in functional detail:
- Every file and function
- End-to-end execution flow
- Scoring formulas and decision rules
- Input/output contracts
- Edge-case behavior

The goal is to make it easy to reason about behavior, validate outputs, and extend the code safely.

## 0. Expanded Mode Addendum (Latest)

The engine now supports two runtime shapes:

1. Legacy single-target path (backward-compatible):
   - Triggered when one target is provided and no multivariate/grain features are requested.
   - Returns the same `TSDFEReport` contract as before.
2. Expanded path:
   - Supports `target_cols`, `feature_cols`, `grain_cols`, `granularity_levels`, and `mode`.
   - Returns structured sections:
   - `overall_univariate`
   - `overall_multivariate`
   - `overall_by_granularity`
   - `by_grain`
   - `best_granularity_by_target`
   - `recommended_approach_by_target`

External module added:
- `multivariate_decision/diagnostic.py`
- Class: `MultivariateDiagnostic`
- Methods:
  - `granger_test(...)`
  - `residual_corr_test(...)`
  - `cv_comparison(...)`
  - `structural_violation_check(...)`
  - `diagnose(...)`

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

Primary entrypoint:
- `run_ts_dfe(...)` in `engine.py`

Primary output object:
- `TSDFEReport` (dict-like object with pretty `__str__` rendering)

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
8. `build_human_readable_report(result)` generates printable report text
9. Wrap final dict in `TSDFEReport` and return

Notes:
- Integrity uses raw grouped series with missing periods preserved in expected-index comparison.
- Most statistical modules operate on gap-filled regularized series to avoid failures from missing observations.
- Structural module is optional and enabled only when `structural_cols` exists in dataframe.

---

## 3. Output Contract

Top-level returned keys from `run_ts_dfe`:
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
- `human_readable_report`

Type guarantees:
- All metric fields are numeric (`float`/`int`, often stored as `float`).
- Classification and recommendation fields are strings.
- `risk_flags` is a list of strings.

---

## 4. File-by-File and Function-by-Function

## 4.1 `engine.py`

### `run_ts_dfe(df, date_col, target_col, structural_cols=None, freq=None) -> dict`
Purpose:
- Orchestrates full TS-DFE pipeline.

Inputs:
- `df`: source DataFrame
- `date_col`: datetime-like column name
- `target_col`: numeric target column name
- `structural_cols`: optional entity columns
- `freq`: optional forced frequency alias

Steps:
1. Normalizes `structural_cols`.
2. Calls `analyze_integrity` (raw structure health).
3. Calls `build_regular_series` from utils (group/sort/reindex frequency).
4. Creates model-ready series via interpolation + forward/backward fill.
5. Calls all module analyzers.
6. Builds `modules` dict.
7. Calls `synthesize(modules)` for global decisions.
8. Assembles final dict.
9. Generates `human_readable_report`.
10. Returns `TSDFEReport(result)`.

Edge behavior:
- Missing/invalid dates and target values are dropped in preprocessing.
- If `structural_cols` are absent or invalid, structure module returns disabled output.

---

## 4.2 `utils.py` (Shared Foundations)

### `SeriesBundle` dataclass
Fields:
- `series`: regularized `pd.Series`
- `freq_alias`: inferred/forced alias (e.g., `D`, `W`)
- `freq_seconds`: inferred/forced period length in seconds

### `safe_float(value, default=np.nan) -> float`
Purpose:
- Safe numeric cast with finite check.

Behavior:
- Returns `default` for `None`, conversion errors, inf/nan-like invalids.

### `safe_div(numerator, denominator, default=np.nan) -> float`
Purpose:
- Safe division with zero/invalid denominator handling.

### `clip(value, low=0.0, high=100.0) -> float`
Purpose:
- Numeric clamp used in scoring.

### `clip01(value) -> float`
Purpose:
- Convenience clamp into `[0, 1]`.

### `prepare_dataframe(df, date_col, target_col) -> pd.DataFrame`
Purpose:
- Canonical cleaning step for date/target.

Steps:
1. Column existence checks.
2. `to_datetime(..., errors="coerce")`
3. `to_numeric(..., errors="coerce")`
4. Drop invalid rows
5. Sort by date

### `seconds_to_alias(freq_seconds) -> Optional[str]`
Purpose:
- Convert seconds interval to pandas alias (`D`, `H`, `T`, `S` forms).

### `infer_frequency(index) -> (alias, seconds)`
Purpose:
- Infer frequency from datetime index.

Logic:
1. Try `pd.infer_freq`.
2. Convert inferred offset via `_offset_to_seconds`.
3. Fallback to median observed delta.

### `resolve_frequency(index, freq) -> (alias, seconds)`
Purpose:
- Prefer explicit `freq`, otherwise infer.

### `_offset_to_seconds(offset) -> float`
Purpose:
- Robust conversion from pandas offset to seconds.

Logic:
1. Try `pd.Timedelta(offset)`.
2. Fallback for week/month/quarter/year/business day offsets by heuristic day counts.

### `build_expected_index(index, freq_alias, freq_seconds) -> pd.DatetimeIndex`
Purpose:
- Build ideal regular index from min->max datetime bounds.

Logic:
1. Use `freq_alias` if valid.
2. Else derive alias from seconds.
3. Else return unique sorted original index.

### `build_regular_series(df, date_col, target_col, freq=None, agg="sum") -> SeriesBundle`
Purpose:
- Convert raw data to a regular time series.

Steps:
1. `prepare_dataframe`
2. Group by date and aggregate target
3. Resolve frequency
4. Build expected index
5. Reindex grouped series to expected grid

### `rolling_window_size(n, min_size=5, frac=0.15, max_size=60) -> int`
Purpose:
- Consistent adaptive rolling window size.

### `lagged_design(series, lags) -> (X, y)`
Purpose:
- Build supervised matrix for AR-style regression.

### `ols_predictions(X, y) -> (pred, intercept)`
Purpose:
- Fit OLS via `np.linalg.lstsq`, return predictions.

### `ar_r2(series, lags) -> float`
Purpose:
- Compute in-sample AR(lags) R^2.

### `ar_mae(series, lags) -> float`
Purpose:
- Compute in-sample AR(lags) MAE.

### `naive_mae(series) -> float`
Purpose:
- One-step naive forecast MAE (`y_t-1` as prediction).

### `mean_forecast_mae(series) -> float`
Purpose:
- Expanding-mean one-step MAE.

### `manual_acf(series, max_lag=10) -> dict`
Purpose:
- ACF fallback when statsmodels unavailable or fails.

### `detect_seasonal_lag(freq_seconds) -> int`
Purpose:
- Heuristic seasonal lag from sampling frequency.

Rules:
- Intra-day: day-length lag
- Daily-ish: `7`
- Weekly-ish: `52`
- Monthly-ish: `12`
- Otherwise `0`

### `trend_r2(series) -> float`
Purpose:
- Linear trend explanatory strength using normalized time index.

---

## 4.3 `integrity.py` (Data Integrity & Structural Health)

### `analyze_integrity(df, date_col, target_col, freq=None) -> dict`
Purpose:
- Validate data quality assumptions before modeling.

Metrics:
- `row_count`
- `observation_count`
- `missing_timestamp_ratio`
- `duplicate_timestamp_ratio`
- `gap_ratio`
- `zero_ratio`
- `negative_ratio`
- `outlier_ratio_iqr`
- `outlier_ratio_mad`
- `rolling_mean_shift_max`
- `structural_break_count`
- `coverage_duration_days`
- `frequency_seconds`
- `integrity_score`
- `integrity_classification`

Key logic:
- Missing timestamp ratio compares expected regular index against actual grouped timestamps.
- Gap ratio checks irregular interval spacing from modal delta.
- Outliers:
  - IQR method: outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
  - MAD method: robust z-score threshold `|z| > 3.5`
- Structural breaks: rolling mean absolute shift spikes above `median + 2*std`.

Integrity score formula:
- `100 * (1 - weighted_penalty)` with penalties from missing, duplicates, gaps, outliers, breaks, negatives.
- Clipped to `[0, 100]`.

Classification rule:
- `>=85`: `Healthy`
- `>=65`: `Monitor`
- else: `Critical`

---

## 4.4 `distribution.py` (Distribution & Tail Analysis)

### `analyze_distribution(series) -> dict`
Purpose:
- Characterize distribution shape, tails, asymmetry, and zero inflation.

Metrics:
- `mean`, `std`, `cv`
- `skewness`, `kurtosis`
- `p99_median_ratio`
- `tail_heaviness_index` (`(p99-p95)/(p75-p50)`)
- `spike_frequency_95`
- `symmetry_score`
- `zero_inflation_ratio`
- `distribution_risk_score`
- Binary flags:
  - `is_gaussian_like`
  - `is_heavy_tailed`
  - `is_zero_inflated`
  - `is_skewed_transactional`
- `distribution_classification`

Classification precedence:
1. Zero-inflated
2. Heavy-tailed
3. Skewed transactional
4. Gaussian-like

Heavy-tail trigger:
- `kurtosis > 3` OR `p99_median_ratio > 8` OR `tail_heaviness_index > 3`.

---

## 4.5 `temporal.py` (Temporal Dependence Diagnostics)

Dependencies:
- Uses statsmodels if available for ACF/PACF/Ljung-Box/STL.
- Falls back to internal methods if not.

### `_acf_with_fallback(series, max_lag=10) -> dict`
Purpose:
- Return lagged ACF values using statsmodels or manual fallback.

### `_pacf_with_fallback(series, max_lag=10) -> dict`
Purpose:
- Return lagged PACF values using statsmodels or ACF approximation fallback.

### `analyze_temporal(series, freq_seconds) -> dict`
Purpose:
- Quantify memory, seasonality, and predictive gain over naive.

Metrics:
- `acf_lag_1...acf_lag_10`
- `pacf_lag_1...pacf_lag_10`
- `ljung_box_pvalue_lag10`
- `ar1_r2`, `ar5_r2`
- `seasonal_lag`, `seasonal_acf`
- `seasonality_strength_stl`
- `naive_mae`, `ar5_mae`
- `autoregressive_improvement_over_naive`
- `temporal_signal_strength_score`

Temporal score formula:
- Weighted combination of:
  - `|acf1|`
  - best of `ar1_r2/ar5_r2`
  - STL seasonality strength
  - AR vs naive improvement
- Scaled to `[0, 100]`.

Short-series behavior:
- If `n < 8`, returns numeric NaNs and `seasonal_lag = 0`.

---

## 4.6 `stationarity.py` (Stationarity & Regime Stability)

### `analyze_stationarity(series) -> dict`
Purpose:
- Determine whether process is stationary, trending, drifting, or regime-shifting.

Metrics:
- `adf_pvalue`
- `kpss_pvalue`
- `rolling_mean_drift`
- `rolling_variance_drift`
- `cusum_max_abs`
- `trend_strength_r2`
- `stability_score`
- `stationarity_classification`

Key logic:
- ADF/KPSS computed when statsmodels exists and `n >= 20`.
- CUSUM proxy built from standardized series cumulative sum.
- Trend strength from linear `R^2`.

Stability score:
- Starts at 100 with penalties for:
  - failing ADF evidence
  - failing KPSS evidence
  - mean/variance drift
  - high CUSUM
  - strong trend

Classification rules:
1. `Stationary` if ADF/KPSS/stability thresholds all satisfied
2. `Trend-dominated` if `trend_strength_r2 >= 0.40`
3. `Regime-shifting` if CUSUM or drift thresholds breached
4. else `Drift-prone`

---

## 4.7 `volatility.py` (Volatility & Heteroskedasticity)

### `analyze_volatility(series) -> dict`
Purpose:
- Diagnose volatility behavior and conditional variance structure.

Metrics:
- `rolling_std_mean_corr`
- `arch_test_pvalue`
- `volatility_clustering_index` (ACF lag1 of squared changes)
- `cv_across_time_windows`
- `event_volatility_index` (`p95(|delta|)/p50(|delta|)`)
- `volatility_risk_score`
- `volatility_classification`

Classification rules:
1. `Event volatility` if `event_volatility_index > 4`
2. `Clustered volatility` if ARCH p<0.05 and clustering>0.25
3. `Heteroskedastic` if ARCH significant OR strong rolling corr
4. else `Homoskedastic`

Risk score:
- Weighted from rolling correlation, clustering index, ARCH flag, and event index.
- Clipped to `[0, 100]`.

---

## 4.8 `structure.py` (Structural Concentration)

### `_gini(values) -> float`
Purpose:
- Compute Gini concentration from non-negative finite shares.

### `_normalized_entropy(prob) -> float`
Purpose:
- Compute entropy normalized to `[0,1]`.

### `analyze_structure(df, date_col, target_col, structural_cols=None) -> dict`
Purpose:
- Quantify dependence on top entities and structural volatility impacts.

Disabled behavior:
- If no valid `structural_cols`, returns `structure_enabled=0` with default metrics.

Enabled metrics:
- `entity_count`
- `top1_revenue_share_pct`
- `top5_revenue_share_pct`
- `top10_revenue_share_pct`
- `gini_coefficient`
- `herfindahl_index`
- `entropy_normalized`
- `volatility_contribution_top_decile`
- `customer_churn_volatility`
- `concentration_score`
- `structure_classification`

Key logic:
- Entity key formed by concatenating provided structural columns.
- Concentration:
  - top-k shares
  - HHI from percent shares
  - entropy and gini blending
- Volatility contribution:
  - variance ratio of top-decile entity aggregate changes vs total changes
- Churn volatility:
  - entry/exit activity variability across periods

Classification rules:
- `Dominated by few entities` if very high top-share/HHI
- `Concentrated` for moderate concentration thresholds
- else `Diversified`

---

## 4.9 `granularity.py` (Granularity & Forecastability Benchmarking)

### `_metrics_for_series(series) -> dict`
Purpose:
- Compute core comparability metrics for one timescale.

Metrics:
- `count`
- `cv`
- `acf1`
- `naive_mae`
- `ar1_mae`
- `mean_forecast_mae`
- `model_improvement_ratio`
- `granularity_score`

Granularity score:
- Mix of signal (`acf1`), low-noise proxy (`cv` inverse), and AR improvement.
- Scaled to `[0, 100]`.

### `analyze_granularity(regular_series) -> dict`
Purpose:
- Compare original vs weekly vs monthly aggregations.

Flow:
1. Compute base, weekly (`W`), monthly (`ME`) series.
2. Compute metrics for each.
3. Select max `granularity_score` as `optimal_granularity`.
4. Compute:
   - `signal_gain_ratio`
   - `noise_reduction_ratio`
   - `model_improvement_ratio` from best granularity

---

## 4.10 `classification.py` (Final Synthesis Engine)

### `_fmt(v) -> str`
Purpose:
- Local numeric formatter for recommendation strings.

### `_risk_flags(results) -> list[str]`
Purpose:
- Convert threshold breaches into explicit risk flag messages.

Examples:
- high missing timestamps
- duplicate timestamps
- heavy tails
- weak temporal signal
- regime instability
- high volatility risk
- high concentration
- improved signal under aggregation

### `_determine_classification(results, forecastability_score) -> str`
Purpose:
- Rule-based final class selection.

Class outcomes:
- `Low Forecastability`
- `Structural-Concentration Dominated`
- `Regime-Shifting`
- `Trend-Dominated`
- `Seasonal Stable`
- `Smooth Autoregressive`
- `Event-Driven Transactional`
- `Externally Driven`

Priority order:
- Applied top-to-bottom in function, first match wins.

### `_build_modeling_recommendation(final_classification, results) -> str`
Purpose:
- Construct metric-cited recommendation text tailored to final class.

Guarantee:
- Recommendation includes measured evidence (`key=value` style metrics).

### `_engineering_decision_recommendation(final_classification, results) -> str`
Purpose:
- Create engineering-action framing (production vs guarded baseline posture).

### `synthesize(results) -> dict`
Purpose:
- Produce final decision layer and summaries.

Composite components:
- `temporal_component`
- `stability_component`
- `concentration_component` (inverted concentration score when structure enabled)
- `noise_component` (inverse CV proxy)
- `model_component` (improvement ratio)

Forecastability score formula:
- `0.30*temporal + 0.25*stability + 0.15*concentration + 0.15*noise + 0.15*model`
- Clipped to `[0,100]`

Returned fields:
- `classification`
- `forecastability_score`
- `modeling_recommendation`
- `risk_flags`
- `executive_summary`
- `engineering_decision_recommendation`

---

## 4.11 `readable.py` (Presentation Layer)

### `TSDFEReport(dict)`
Purpose:
- Dict-compatible container with human-readable `__str__` and `__repr__`.

Behavior:
- If `human_readable_report` exists, printing the object renders that text.
- Key/value access remains identical to normal dict.

### `_fmt_num(value, decimals=3) -> str`
Purpose:
- Numeric string formatter with NaN handling and comma formatting for large numbers.

### `_fmt_pct(value, decimals=1) -> str`
Purpose:
- Percent formatter (`value * 100`).

Note:
- Currently helper is defined but not used in report generation path.

### `_flatten(d, prefix="") -> dict`
Purpose:
- Flatten nested dictionaries for detailed metric dump section.

### `_signal_label(score) -> str`
Purpose:
- Map signal score to `Weak/Moderate/Strong/Unknown`.

### `_seasonality_label(strength) -> str`
Purpose:
- Map seasonality strength to `Weak/Moderate/Strong/Unknown`.

### `_split_recommendation_to_bullets(recommendation) -> list[str]`
Purpose:
- Convert recommendation text into bullet-friendly lines.

### `build_human_readable_report(report) -> str`
Purpose:
- Build full narrative report text with both summary and complete metrics.

Sections generated:
- `DATA CHARACTERIZATION REPORT`
- Signal/variance/distribution/concentration/seasonality/granularity summary lines
- Final class and score
- Recommendation bullets
- Risk flags
- Executive and engineering summaries
- Full structured metrics per module
- Final output fields snapshot

---

## 4.12 `__init__.py`

Exports:
- `run_ts_dfe`
- `build_human_readable_report`
- `TSDFEReport`

Purpose:
- Public API surface for package imports.

---

## 5. Statistical Dependencies and Fallback Strategy

Optional dependency:
- `statsmodels`

Where used:
- Temporal: ACF/PACF/Ljung-Box/STL
- Stationarity: ADF/KPSS
- Volatility: ARCH LM test

Fallback behavior:
- Manual ACF and internal regressions keep engine operational without statsmodels.
- Some p-values and seasonality metrics become `NaN` when tests are unavailable/inapplicable.

---

## 6. Data and Modeling Assumptions

1. Target is numeric and aggregatable by date.
2. Time series can be mapped to a regular grid for diagnostics.
3. AR diagnostics are in-sample fit quality indicators, not production forecast backtests.
4. Rule-based classification is deterministic and threshold-driven.
5. Structure diagnostics assume non-negative additive contribution logic (shares/concentration).

---

## 7. Extension Guidance (Safe Change Points)

Recommended extension locations:
- Add new metric helpers in `utils.py`.
- Add new module metrics in module analyzer return dicts.
- Update scoring weights in:
  - module-specific score formulas
  - `classification.synthesize` composite
- Add new risk flags in `_risk_flags`.
- Add new final classes in `_determine_classification`.
- Expand readable report sections in `build_human_readable_report`.

Change discipline:
1. Preserve output key stability where downstream users depend on keys.
2. Keep all metrics numeric.
3. Keep recommendation text metric-grounded.
4. Add/adjust tests in `tests/test_ts_dfe.py` for every rule change.

---

## 8. Traceability Checklist (Input -> Decision)

For a single run:
1. Input row acceptance defined by `prepare_dataframe`.
2. Temporal regularization logic in `build_regular_series`.
3. Quality diagnostics in `integrity`.
4. Signal/noise/statistical diagnostics in five core modules.
5. Structural concentration diagnostics if entity columns exist.
6. Granularity comparison across original/weekly/monthly.
7. Final score and class from weighted deterministic synthesis.
8. Human-readable reporting layer preserves all computed outputs.

This checklist can be used for debugging, audits, and stakeholder walkthroughs.

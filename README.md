# TS-DFE — Time Series Diagnostic & Forecastability Engine

Generic engine for diagnosing the structure of a time series and recommending a modeling approach. Runs 7 diagnostic modules, synthesizes a forecastability score, and classifies the series into one of 9 regimes.

---

## Architecture: 7 diagnostic modules + synthesis

| Module | File | Purpose |
|--------|------|---------|
| Integrity | `integrity.py` | Data quality — gaps, duplicates, outliers, structural breaks |
| Distribution | `distribution.py` | Shape — zero-inflation, heavy tails, skewness |
| Temporal | `temporal.py` | Signal — ACF, AR fit, seasonality strength |
| Stationarity | `stationarity.py` | Stability — ADF/KPSS, CUSUM, mean/variance drift |
| Volatility | `volatility.py` | Heteroskedasticity — ARCH, clustering, event spikes |
| Structure | `structure.py` | Concentration — entity shares, HHI, Gini (optional) |
| Granularity | `granularity.py` | Aggregation — best level from original/weekly/monthly |
| Classification | `classification.py` | Synthesis — score, regime label, recommendation |

External multivariate helper: `multivariate_decision/diagnostic.py`

Full architecture reference: `ts_dfe/FUNCTIONAL_ARCHITECTURE.md`
Practical usage guide: `USER_GUIDE.md`

---

## 9 Classification Regimes

| Regime | Primary trigger |
|--------|----------------|
| Low Forecastability | forecastability_score < 35 |
| Structural-Concentration Dominated | structure enabled AND top entities dominate |
| Regime-Shifting | CUSUM ≥ 1.80 or rolling mean drift ≥ 0.45 |
| Trend-Dominated | trend_strength_r2 ≥ 0.40 |
| Seasonal Stable | seasonality_strength ≥ 0.35 AND stability_score ≥ 60 |
| Smooth Autoregressive | ar5_r2 ≥ 0.45 AND Homoskedastic volatility |
| Intermittent Demand | Zero-inflated AND event_volatility_index ≤ 3.0 |
| Event-Driven Transactional | Skewed transactional OR Event volatility |
| Externally Driven | Fallback when no pattern dominates |

---

## Forecastability Score

Weighted composite [0–100]:

```
0.30 × temporal_signal_strength_score
0.25 × stability_score
0.15 × concentration_component  (inverted if structure enabled)
0.15 × noise_component          (1 / (1 + CV))
0.15 × model_improvement_ratio  (AR vs naive gain)
```

Score < 35 → "Low Forecastability" regardless of other signals.

---

## Entry Point

```python
from ts_dfe.engine import run_ts_dfe

report = run_ts_dfe(
    df,
    date_col="date",
    target_col="sales",
    structural_cols=["customer"],   # optional — enables concentration module
    report_mode="summary",          # "summary" | "technical"
)

print(report)                       # prints human_readable_report
print(report["classification"])     # regime label string
print(report["forecastability_score"])
```

---

## Usage Modes

### 1. Univariate (single target, legacy-compatible)
```python
run_ts_dfe(df, date_col="date", target_col="sales")
```

### 2. Multivariate (multiple targets or external features)
```python
run_ts_dfe(
    df,
    date_col="date",
    target_cols=["sales"],
    feature_cols=["marketing", "temperature"],
    mode="multivariate",
)
```

### 3. Grain-wise (run per segment)
```python
run_ts_dfe(
    df,
    date_col="date",
    target_cols=["sales"],
    grain_cols=["region", "channel"],
    mode="multivariate",
)
```

---

## Report Modes

| Mode | Content |
|------|---------|
| `"summary"` | Headline: regime + score. Supporting: 7 evidence lines. Recommendation + model starting point + risk flags. |
| `"technical"` | Everything in summary plus full structured metrics for all modules. |

---

## Output Keys (univariate)

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
```

## Output Keys (expanded multivariate/grain mode)

```
overall_univariate          dict[target → univariate report]
overall_multivariate        dict[target → multivariate signal]
overall_by_granularity      dict[level → dict[target → report]]
best_granularity_by_target  dict[target → "original"|"weekly"|"monthly"]
recommended_approach_by_target dict[target → recommendation]
by_grain                    dict[grain_key → full report block]
summary                     str
```

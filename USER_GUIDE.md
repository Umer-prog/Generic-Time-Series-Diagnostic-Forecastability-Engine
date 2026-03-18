# TS-DFE User Guide

Practical guide for running the Time Series Diagnostic & Forecastability Engine against your data.

---

## What TS-DFE does

Given a time series, TS-DFE answers three questions:

1. **What kind of series is this?** — Assigns one of 9 regime labels (e.g. Seasonal Stable, Intermittent Demand, Trend-Dominated).
2. **How forecastable is it?** — A score from 0–100 driven by temporal signal strength, stability, noise, and AR improvement.
3. **What should I do with it?** — A model family recommendation with metric evidence.

It does **not** build or train a model. It characterizes the data so you can make a better modeling decision.

---

## Quick start

```python
import pandas as pd
from ts_dfe.engine import run_ts_dfe

df = pd.read_excel("Sales.xlsx", sheet_name="Report", header=4, usecols="C:L")

report = run_ts_dfe(
    df,
    date_col="Posting Date",
    target_col="Sales",
    structural_cols=["Salesperson Code", "Sell-to Customer No"],
    report_mode="summary",
)

print(report)
```

`print(report)` renders the human-readable summary. `report["classification"]` returns the regime label string. `report["forecastability_score"]` returns the numeric score.

---

## Call signatures

### Univariate (single target)

```python
run_ts_dfe(
    df,
    date_col="date",
    target_col="sales",
    structural_cols=["customer"],   # optional — enables concentration module
    report_mode="summary",          # "summary" | "technical"
)
```

### Multivariate (target + external features)

```python
run_ts_dfe(
    df,
    date_col="date",
    target_cols=["sales"],
    feature_cols=["marketing_spend", "temperature"],
    structural_cols=["customer"],
    mode="multivariate",
    report_mode="summary",
)
```

### Grain-wise (run per segment)

```python
run_ts_dfe(
    df,
    date_col="date",
    target_cols=["sales"],
    grain_cols=["region", "channel"],
    mode="multivariate",
    report_mode="summary",
)
```

Each unique combination of grain values is diagnosed separately and collected under `report["by_grain"]`.

---

## Report modes

| Mode | Use when |
|------|----------|
| `"summary"` | You want a quick regime verdict and top-line evidence. Default for operational use. |
| `"technical"` | You want all raw module metrics for debugging, model config, or audit. |

Summary mode leads with the **CLASSIFICATION** headline and shows only the metrics that directly support or qualify the regime decision. Technical mode adds a full structured metrics dump for every module.

---

## Reading the summary output

```
DATA CHARACTERIZATION REPORT
Series: sales | 365 observations | 2022-01-01 → 2022-12-31 | freq: D

CLASSIFICATION: Seasonal Stable         ← regime label — act on this
FORECASTABILITY SCORE: 74.3/100
Key drivers: temporal=81.2, stability=68.5, model_improvement=0.72

Stationarity: Stationary (stability_score=68.5)
Signal Strength: Strong (temporal_score=81.2, avg_acf_1_3=0.71)
Distribution: Gaussian-like (zero_inflation_ratio=0.0%, skewness=0.18)
Volatility: Homoskedastic
Seasonality: Strong (strength=0.62, lag=7)
Granularity: optimal=original (signal_gain=1.4%)
Concentration: Concentrated (top5=58.3%)

RECOMMENDATION: Use seasonal models (SARIMA/TBATS/Prophet) at original
  granularity because seasonality_strength_stl=0.623, stability_score=68.5,
  temporal_signal_strength_score=81.2.
Reason: stationarity=Stationary, volatility=Homoskedastic,
  distribution=Gaussian-like, model_improvement=0.72

MODEL STARTING POINT
  Primary model: SARIMA / TBATS / Prophet
  Training granularity: original
  Loss function: MAE / Huber

RISK FLAGS:
  • (none)
```

**What to act on:** CLASSIFICATION + RECOMMENDATION. Everything else is supporting evidence.

**Key drivers** shows the three largest forecastability score components (temporal 30%, stability 25%, model improvement 15%). Low values here explain why the score is low.

**Granularity line:** if `optimal != original`, the engine found that aggregating improves signal. Consider training at the recommended level.

**RISK FLAGS:** each flag names the specific metric and its value. Address Critical flags before modeling.

---

## The 9 classification regimes

| Regime | Primary trigger | Suggested model family |
|--------|----------------|------------------------|
| Low Forecastability | forecastability_score < 35 | Conservative baselines; add external drivers |
| Structural-Concentration Dominated | Top entities dominate share | Model top entities separately; aggregate bottom-up |
| Regime-Shifting | CUSUM ≥ 1.80 or rolling mean drift ≥ 0.45 | Rolling/regime-aware models with change-point monitoring |
| Trend-Dominated | trend_strength_r2 ≥ 0.40 | Differenced regression or local-trend state space |
| Seasonal Stable | seasonality_strength ≥ 0.35 AND stability_score ≥ 60 | SARIMA / TBATS / Prophet |
| Smooth Autoregressive | ar5_r2 ≥ 0.45 AND Homoskedastic | ARIMA / ETS |
| Intermittent Demand | Zero-inflated AND event_volatility_index ≤ 3.0 | Croston / SBA / ADIDA |
| Event-Driven Transactional | Skewed transactional OR Event volatility | Event/exogenous-feature models with robust loss |
| Externally Driven | Fallback — no pattern dominates | Causal/external-regressor models |

**Priority order matters.** Low Forecastability overrides everything. Structural-Concentration comes next. The engine resolves the first matching rule top-to-bottom.

---

## Forecastability score

```
0.30 × temporal_signal_strength_score    (ACF, AR fit, STL seasonality)
0.25 × stability_score                   (ADF/KPSS, drift, CUSUM)
0.15 × concentration_component           (inverted concentration if structure enabled)
0.15 × noise_component                   (1 / (1 + CV))
0.15 × model_improvement_ratio           (AR vs naive MAE gain)
```

Score < 35 → **Low Forecastability** regardless of other signals.

Rough interpretation:
- 70–100: good forecastability, standard modeling pipeline appropriate
- 50–70: moderate signal, monitor stability and review risk flags
- 35–50: weak signal, use simpler models with wider intervals
- < 35: poor forecastability, baselines only, add external data

---

## Domain-specific guidance

### Sales (transactional line items)

- **Aggregated total sales:** expect Seasonal Stable or Trend-Dominated. Check seasonality lag — daily data often shows lag=7 (weekly cycle).
- **SKU-level:** often Intermittent Demand (many zero days). Use Croston/SBA, not SARIMA.
- **Spike-heavy promotions:** expect Event-Driven Transactional. Use robust loss, consider exogenous event flags.
- **structural_cols useful here:** pass customer or salesperson to measure concentration risk. High concentration score means top customers drive disproportionate variance.

### Purchases / procurement

- **Aggregated spend:** typically Smooth Autoregressive or Seasonal Stable at monthly/weekly granularity.
- **PO line items:** often Intermittent Demand. High `zero_inflation_ratio` confirms.
- **Watch the granularity flag:** if `optimal_granularity = monthly`, aggregate before modeling — don't fit daily.

### Inventory

- **Closing stock balance:** non-stationary by nature (cumulative variable). Engine will correctly detect Trend-Dominated or Regime-Shifting. Forecast the **net movement** (receipts - issues), not the level directly.
- **Receipts/issues series:** positive-only = Intermittent Demand typical. Signed movement series may show `negative_ratio > 0` in integrity — this is semantically valid, not an error.
- **Reorder-level shifts:** will trigger Regime-Shifting. The recommendation ("rolling or regime-aware models") is correct even though the cause is policy, not a true structural break.

### Finance

- **P&L accounts (monthly/quarterly):** typically Seasonal Stable or Smooth Autoregressive.
- **Ledger journal entries (daily):** high zero-inflation and skewness → Event-Driven Transactional. The granularity flag to `optimal=monthly` is the intended outcome, not a warning.
- **Mixed frequencies:** the engine tests original, weekly, and monthly. A poor `integrity_score` with many missing timestamps usually means the raw data is at mixed frequencies — aggregate to a consistent level before running.

---

## Understanding risk flags

Each flag includes the metric name and observed value so you can assess severity.

| Flag | What it means | Action |
|------|--------------|--------|
| `missing_timestamp_ratio=X` | X% of expected periods have no data | Fill or investigate gaps before modeling |
| `duplicate_timestamp_ratio=X` | X% of dates appear more than once | Deduplicate; check source aggregation |
| `outlier_ratio_mad=X` | X% of values are robust outliers | Review for data errors vs real events |
| `tail_heaviness_index=X` | Heavy tail behavior | Use robust loss (Huber/MAE), avoid MSE |
| `temporal_signal_strength_score=X` | Weak autocorrelation/AR fit | Consider external drivers; baselines safer |
| `cusum_max_abs=X` | Regime instability | Rolling retrain; change-point monitoring |
| `volatility_risk_score=X` | Elevated volatility | Use wider prediction intervals |
| `concentration_score=X` | Top entities dominate | Model top entities separately |
| `optimal_granularity=weekly/monthly` | Signal improves with aggregation | Train at that level; resample first |

---

## Output structure reference

### Univariate output keys

```
integrity            dict — data quality metrics
distribution         dict — distribution shape metrics
temporal             dict — ACF (lags 1-3), AR R², seasonality
stationarity         dict — ADF/KPSS, CUSUM, drift, stability_score
volatility           dict — ARCH, event_volatility_index, risk_score
structure            dict — entity concentration (disabled if no structural_cols)
granularity          dict — original/weekly/monthly comparison
classification       str  — regime label
forecastability_score float
modeling_recommendation str
risk_flags           list[str]
executive_summary    str
engineering_decision_recommendation str
human_readable_report str
report_mode          str
```

### Expanded (multivariate/grain) output keys

```
overall_univariate          dict[target → univariate report]
overall_multivariate        dict[target → multivariate signal]
overall_by_granularity      dict[level → dict[target → report]]
best_granularity_by_target  dict[target → "original"|"weekly"|"monthly"]
recommended_approach_by_target dict[target → recommendation string]
by_grain                    dict[grain_key → full report block]
summary                     str
```

### Multivariate signal keys (per target in `overall_multivariate`)

```
exogenous_r2                    float — R² from exogenous-only regression
ar_r2                           float — R² from AR-only model
exogenous_dominance_ratio       float — exog_r2 / (exog_r2 + ar_r2)
exogenous_signal_classification str   — "Exogenous Dominated" | "Mixed Drivers" | "Autoregressive Dominated"
```

---

## Common patterns

**"I just want the regime label and score"**
```python
report = run_ts_dfe(df, date_col="date", target_col="sales", report_mode="summary")
label = report["classification"]
score = report["forecastability_score"]
flags = report["risk_flags"]
```

**"I want to diagnose many SKUs in a loop"**
```python
results = {}
for sku in df["sku"].unique():
    subset = df[df["sku"] == sku]
    results[sku] = run_ts_dfe(subset, date_col="date", target_col="qty", report_mode="summary")
```

**"I want to check if features are useful before building a model"**
```python
report = run_ts_dfe(
    df,
    date_col="date",
    target_cols=["sales"],
    feature_cols=["marketing", "temperature"],
    mode="multivariate",
    report_mode="technical",
)
signal = report["overall_multivariate"]["sales"]
print(signal["exogenous_signal_classification"])  # "Exogenous Dominated" → features help
print(signal["exogenous_dominance_ratio"])         # > 0.6 → feature-heavy model warranted
```

**"I want grain-level breakdown by region"**
```python
report = run_ts_dfe(
    df,
    date_col="date",
    target_cols=["sales"],
    grain_cols=["region"],
    mode="multivariate",
    report_mode="summary",
)
for grain_key, grain_report in report["by_grain"].items():
    print(grain_key, grain_report["summary"])
```

---

## Data requirements

- At least 20 observations recommended for reliable results (< 8 skips temporal module entirely).
- Date column must be parseable by `pd.to_datetime`.
- Target column must be numeric.
- Duplicate dates are valid input — the engine aggregates them (sum by default).
- Missing dates within the series range are handled by interpolation before statistical analysis.
- Negative target values are valid (inventory movements, finance adjustments) — flagged in `integrity` but not excluded.

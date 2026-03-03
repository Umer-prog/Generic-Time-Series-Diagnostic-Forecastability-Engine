# Multivariate Decision Helper

This module is intentionally separate from `ts_dfe` core engine.

Primary class:

- `MultivariateDiagnostic`

Implemented methods:

- `granger_test(...)`
- `residual_corr_test(...)`
- `cv_comparison(...)`
- `structural_violation_check(...)`
- `diagnose(...)`

Typical output:

```python
{
  "cross_lag_effect": "weak|moderate|strong",
  "residual_dependency": 0.12,
  "cv_improvement_multivariate": 0.08,
  "recommendation": "univariate|univariate_with_exogenous|multivariate",
  "details": {...}
}
```

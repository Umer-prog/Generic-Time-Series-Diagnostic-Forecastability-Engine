from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import grangercausalitytests

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# Keep runtime output clean for unstable edge cases in statistical internals.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"statsmodels\.regression\.linear_model",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"numpy\.lib\._function_base_impl",
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        val = float(value)
        if np.isfinite(val):
            return val
        return float(default)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return float(np.clip(_safe_float(value, default=0.0), 0.0, 1.0))


def _fmt_num(value: Any, decimals: int = 3) -> str:
    x = _safe_float(value, default=np.nan)
    if np.isnan(x):
        return "nan"
    if abs(x) >= 1000:
        return f"{x:,.{decimals}f}"
    return f"{x:.{decimals}f}"


def _safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return 0.0
    a = a[mask]
    b = b[mask]
    if np.std(a, ddof=0) < 1e-12 or np.std(b, ddof=0) < 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            corr = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return abs(float(corr))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred) or len(y_true) < 3:
        return np.nan
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 3:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12 or not np.isfinite(ss_tot):
        return np.nan
    ss_res = float(np.sum((yt - yp) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)
    return float(np.clip(r2, -1.0, 1.0))


def _effect_score(cross_lag_effect: str) -> float:
    mapping = {
        "weak": 0.20,
        "moderate": 0.60,
        "strong": 1.00,
    }
    return float(mapping.get(str(cross_lag_effect).lower(), 0.20))


class MultivariateDecisionReport(dict):
    def __str__(self) -> str:
        readable = self.get("human_readable_report", None)
        if isinstance(readable, str) and readable:
            return readable
        return dict.__str__(self)

    def __repr__(self) -> str:
        return self.__str__()


class MultivariateDiagnostic:
    def __init__(self, max_lag: int = 5, cv_folds: int = 3) -> None:
        self.max_lag = max(1, int(max_lag))
        self.cv_folds = max(2, int(cv_folds))

    def _align_inputs(self, target: pd.Series, exog: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        y = pd.Series(target).astype(float).rename("target")
        X = pd.DataFrame(exog).copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        frame = pd.concat([y, X], axis=1).dropna()
        y_clean = frame["target"]
        X_clean = frame.drop(columns=["target"])
        return y_clean, X_clean

    def granger_test(
        self,
        target: pd.Series,
        exog: pd.DataFrame,
        max_lag: int | None = None,
        significance: float = 0.05,
    ) -> dict:
        y, X = self._align_inputs(target, exog)
        lag = self.max_lag if max_lag is None else max(1, int(max_lag))
        if X.shape[1] == 0 or len(y) < (lag + 8):
            return {
                "cross_lag_effect": "weak",
                "significant_feature_ratio": 0.0,
                "feature_min_pvalues": {},
            }

        pvals: dict[str, float] = {}
        for col in X.columns:
            x = X[col]
            if HAS_STATSMODELS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning,
                            module=r"statsmodels\.regression\.linear_model",
                        )
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning,
                            module=r"numpy\.lib\._function_base_impl",
                        )
                        data = pd.concat([y, x], axis=1).dropna()
                        with np.errstate(all="ignore"):
                            out = grangercausalitytests(data, maxlag=lag, verbose=False)
                        candidate = []
                        for l in range(1, lag + 1):
                            if l in out:
                                candidate.append(_safe_float(out[l][0]["ssr_ftest"][1], default=np.nan))
                        pvals[col] = float(np.nanmin(candidate)) if candidate else np.nan
                    continue
                except Exception:
                    pass

            # Fallback if statsmodels unavailable or test fails.
            corrs = []
            for l in range(1, lag + 1):
                if len(y) <= l:
                    continue
                corr_abs = _safe_abs_corr(y.iloc[l:].values, x.iloc[:-l].values)
                corrs.append(corr_abs)
            max_corr = max(corrs) if corrs else 0.0
            pvals[col] = float(np.clip(1.0 - max_corr, 0.0, 1.0))

        sig_count = sum(1 for p in pvals.values() if np.isfinite(p) and p < significance)
        sig_ratio = sig_count / max(len(pvals), 1)

        if sig_ratio >= 0.50:
            effect = "strong"
        elif sig_ratio >= 0.20:
            effect = "moderate"
        else:
            effect = "weak"

        return {
            "cross_lag_effect": effect,
            "significant_feature_ratio": float(sig_ratio),
            "feature_min_pvalues": {k: float(v) for k, v in pvals.items()},
        }

    def residual_corr_test(self, target: pd.Series, exog: pd.DataFrame) -> float:
        y, X = self._align_inputs(target, exog)
        if X.shape[1] == 0 or len(y) < 10:
            return 0.0

        Xv = X.values
        yv = y.values
        X1 = np.column_stack([np.ones(len(Xv)), Xv])
        beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
        pred = X1 @ beta
        resid = yv - pred

        corr_vals = []
        for i in range(Xv.shape[1]):
            corr_vals.append(_safe_abs_corr(resid, Xv[:, i]))
        if not corr_vals:
            return 0.0
        return float(np.mean(corr_vals))

    def cv_comparison(self, target: pd.Series, exog: pd.DataFrame, folds: int | None = None) -> float:
        y, X = self._align_inputs(target, exog)
        if X.shape[1] == 0 or len(y) < 40:
            return 0.0

        f = self.cv_folds if folds is None else max(2, int(folds))
        frame = pd.concat([y.rename("target"), X], axis=1)
        frame["lag1"] = frame["target"].shift(1)
        frame = frame.dropna()
        if len(frame) < 30:
            return 0.0

        n = len(frame)
        min_train = max(20, int(0.5 * n))
        if min_train >= (n - 5):
            return 0.0

        split_points = np.linspace(min_train, n - 5, num=f, dtype=int)
        test_window = max(5, (n - min_train) // f)

        improvements = []
        feature_cols = ["lag1"] + [c for c in X.columns if c in frame.columns]
        for split in split_points:
            train = frame.iloc[:split]
            test = frame.iloc[split : split + test_window]
            if len(train) < 20 or len(test) < 5:
                continue

            y_test = test["target"].values

            # Univariate baseline: lag-1 persistence.
            pred_uni = test["lag1"].values
            mae_uni = np.mean(np.abs(y_test - pred_uni))
            if not np.isfinite(mae_uni) or mae_uni <= 0:
                continue

            # Multivariate linear model using lag1 + exogenous features.
            X_train = train[feature_cols].values
            y_train = train["target"].values
            X_test = test[feature_cols].values

            Xtr = np.column_stack([np.ones(len(X_train)), X_train])
            beta, *_ = np.linalg.lstsq(Xtr, y_train, rcond=None)
            Xte = np.column_stack([np.ones(len(X_test)), X_test])
            pred_multi = Xte @ beta
            mae_multi = np.mean(np.abs(y_test - pred_multi))

            improvements.append((mae_uni - mae_multi) / mae_uni)

        if not improvements:
            return 0.0
        return float(np.mean(improvements))

    def structural_violation_check(self, target: pd.Series, exog: pd.DataFrame) -> dict:
        X = pd.DataFrame(exog).copy()
        if X.shape[1] == 0:
            return {
                "violation_score": 0.0,
                "missing_ratio": 0.0,
                "constant_feature_ratio": 0.0,
                "condition_number": 1.0,
                "sample_feature_ratio": np.nan,
                "flags": [],
            }

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        missing_ratio = float(X.isna().mean().mean())
        X_clean = X.dropna()
        n = len(X_clean)
        p = max(X_clean.shape[1], 1)
        sample_feature_ratio = float(n / p) if p > 0 else np.nan

        if n > 0:
            std = X_clean.std(ddof=0).replace(0, np.nan)
            constant_feature_ratio = float(np.mean(std.isna()))
        else:
            constant_feature_ratio = 1.0

        if n > p and p > 0:
            Xv = X_clean.values
            Xv = (Xv - Xv.mean(axis=0, keepdims=True)) / (Xv.std(axis=0, keepdims=True) + 1e-9)
            try:
                condition_number = float(np.linalg.cond(Xv))
            except Exception:
                condition_number = np.nan
        else:
            condition_number = np.nan

        cond_penalty = _clip01((_safe_float(condition_number, default=0.0) - 20.0) / 80.0)
        sample_penalty = _clip01((10.0 - _safe_float(sample_feature_ratio, default=0.0)) / 10.0)
        violation_score = float(
            (0.35 * _clip01(missing_ratio))
            + (0.25 * _clip01(constant_feature_ratio))
            + (0.25 * cond_penalty)
            + (0.15 * sample_penalty)
        )

        flags = []
        if missing_ratio > 0.10:
            flags.append(f"high_missing_features={missing_ratio:.3f}")
        if constant_feature_ratio > 0.20:
            flags.append(f"constant_feature_ratio={constant_feature_ratio:.3f}")
        if np.isfinite(condition_number) and condition_number > 30:
            flags.append(f"high_collinearity_condition_number={condition_number:.2f}")
        if np.isfinite(sample_feature_ratio) and sample_feature_ratio < 10:
            flags.append(f"low_sample_feature_ratio={sample_feature_ratio:.2f}")

        return {
            "violation_score": float(violation_score),
            "missing_ratio": float(missing_ratio),
            "constant_feature_ratio": float(constant_feature_ratio),
            "condition_number": float(condition_number),
            "sample_feature_ratio": float(sample_feature_ratio),
            "flags": flags,
        }

    def exogenous_dominance_signal(self, target: pd.Series, exog: pd.DataFrame, ar_lag: int = 5) -> dict:
        y, X = self._align_inputs(target, exog)
        lag = max(1, int(ar_lag))

        feature_count = int(X.shape[1])
        exogenous_r2 = 0.0
        ar_r2 = 0.0

        if feature_count > 0 and len(y) >= 8:
            Xv = X.values
            yv = y.values
            X1 = np.column_stack([np.ones(len(Xv)), Xv])
            try:
                beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
                pred = X1 @ beta
                exogenous_r2 = _safe_float(_safe_r2(yv, pred), default=0.0)
            except Exception:
                exogenous_r2 = 0.0
        exogenous_r2 = float(np.clip(exogenous_r2, 0.0, 1.0))

        yv = y.values
        if len(yv) > lag + 6:
            rows = []
            yt = []
            for i in range(lag, len(yv)):
                past = yv[i - lag : i][::-1]
                if np.all(np.isfinite(past)) and np.isfinite(yv[i]):
                    rows.append(past)
                    yt.append(yv[i])
            if len(yt) >= 8:
                X_ar = np.asarray(rows, dtype=float)
                y_ar = np.asarray(yt, dtype=float)
                X_ar1 = np.column_stack([np.ones(len(X_ar)), X_ar])
                try:
                    beta_ar, *_ = np.linalg.lstsq(X_ar1, y_ar, rcond=None)
                    pred_ar = X_ar1 @ beta_ar
                    ar_r2 = _safe_float(_safe_r2(y_ar, pred_ar), default=0.0)
                except Exception:
                    ar_r2 = 0.0
        ar_r2 = float(np.clip(ar_r2, 0.0, 1.0))

        ratio = float(exogenous_r2 / (ar_r2 + 1e-6))
        if ratio >= 1.5:
            exogenous_signal_classification = "Exogenous Dominated"
        elif ratio >= 0.7:
            exogenous_signal_classification = "Mixed Drivers"
        else:
            exogenous_signal_classification = "Autoregressive Dominated"

        return {
            "feature_count": float(feature_count),
            "exogenous_r2": float(exogenous_r2),
            "ar_r2": float(ar_r2),
            "exogenous_dominance_ratio": float(ratio),
            "exogenous_signal_classification": exogenous_signal_classification,
        }

    def _recommendation(
        self,
        cross_lag_effect: str,
        residual_dependency: float,
        cv_improvement_multivariate: float,
        violation_score: float,
        exogenous_signal_classification: str,
    ) -> str:
        if violation_score >= 0.65:
            return "univariate_with_exogenous"
        if exogenous_signal_classification == "Exogenous Dominated":
            if residual_dependency <= 0.25:
                return "univariate_with_exogenous"
            return "multivariate"
        if (
            cross_lag_effect in {"moderate", "strong"}
            and cv_improvement_multivariate >= 0.08
            and residual_dependency <= 0.20
        ):
            return "multivariate"
        if cv_improvement_multivariate > 0 or cross_lag_effect in {"moderate", "strong"}:
            return "univariate_with_exogenous"
        return "univariate"

    def _feature_utility_score(
        self,
        cross_lag_effect: str,
        residual_dependency: float,
        cv_improvement_multivariate: float,
        violation_score: float,
    ) -> float:
        effect_component = _effect_score(cross_lag_effect)
        cv_component = _clip01((cv_improvement_multivariate + 0.05) / 0.25)
        residual_component = 1.0 - _clip01(residual_dependency / 0.40)
        structural_component = 1.0 - _clip01(violation_score)

        score = 100.0 * (
            (0.35 * effect_component)
            + (0.35 * cv_component)
            + (0.20 * residual_component)
            + (0.10 * structural_component)
        )
        return float(np.clip(score, 0.0, 100.0))

    def _add_features_decision(
        self,
        recommendation: str,
        feature_utility_score: float,
        cv_improvement_multivariate: float,
    ) -> str:
        if recommendation == "multivariate" and feature_utility_score >= 65.0:
            return "add_features_high_impact"
        if recommendation == "univariate_with_exogenous" and feature_utility_score >= 45.0:
            return "add_features_selectively"
        if recommendation == "univariate_with_exogenous":
            return "test_selected_features"
        if cv_improvement_multivariate > 0.02:
            return "test_selected_features"
        return "avoid_additional_features"

    def _decision_confidence(
        self,
        feature_utility_score: float,
        cross_lag_effect: str,
        violation_score: float,
    ) -> float:
        effect_component = _effect_score(cross_lag_effect)
        utility_component = _clip01(feature_utility_score / 100.0)
        structural_component = 1.0 - _clip01(violation_score)
        conf = (0.40 * utility_component) + (0.35 * effect_component) + (0.25 * structural_component)
        return float(np.clip(conf, 0.0, 1.0))

    def _build_human_readable_report(self, output: dict) -> str:
        cross_lag_effect = output.get("cross_lag_effect", "weak")
        residual_dependency = _safe_float(output.get("residual_dependency", np.nan), default=np.nan)
        cv_gain = _safe_float(output.get("cv_improvement_multivariate", np.nan), default=np.nan)
        recommendation = output.get("recommendation", "univariate")
        feature_utility_score = _safe_float(output.get("feature_utility_score", np.nan), default=np.nan)
        add_features_decision = output.get("add_features_decision", "avoid_additional_features")
        decision_confidence = _safe_float(output.get("decision_confidence", np.nan), default=np.nan)
        exogenous_r2 = _safe_float(output.get("exogenous_r2", np.nan), default=np.nan)
        ar_r2 = _safe_float(output.get("ar_r2", np.nan), default=np.nan)
        dominance_ratio = _safe_float(output.get("exogenous_dominance_ratio", np.nan), default=np.nan)
        signal_class = output.get("exogenous_signal_classification", "Autoregressive Dominated")
        feature_count = int(_safe_float(output.get("feature_count", 0.0), default=0.0))

        details = output.get("details", {})
        granger = details.get("granger", {}) if isinstance(details, dict) else {}
        structural = details.get("structural_violation", {}) if isinstance(details, dict) else {}

        sig_ratio = _safe_float(granger.get("significant_feature_ratio", np.nan), default=np.nan)
        violation_score = _safe_float(structural.get("violation_score", np.nan), default=np.nan)
        flags = structural.get("flags", [])
        if not isinstance(flags, list):
            flags = [str(flags)]

        if residual_dependency < 0.10:
            residual_label = "Low"
        elif residual_dependency < 0.25:
            residual_label = "Moderate"
        else:
            residual_label = "High"

        if cv_gain >= 0.08:
            cv_label = "Strong improvement"
        elif cv_gain > 0:
            cv_label = "Marginal improvement"
        elif cv_gain > -0.05:
            cv_label = "Neutral"
        else:
            cv_label = "Degradation"

        if violation_score < 0.20:
            violation_label = "Low"
        elif violation_score < 0.50:
            violation_label = "Moderate"
        else:
            violation_label = "High"

        lines: list[str] = []
        lines.append("MULTIVARIATE DIAGNOSTIC REPORT")
        lines.append("----------------------------------------")
        lines.append(
            f"Cross-Lag Effect: {cross_lag_effect.title()} "
            f"(significant_feature_ratio={_fmt_num(sig_ratio, 3)})"
        )
        lines.append(
            f"Residual Dependency: {residual_label} "
            f"(residual_dependency={_fmt_num(residual_dependency, 3)})"
        )
        lines.append(
            f"CV Comparison: {cv_label} "
            f"(cv_improvement_multivariate={_fmt_num(cv_gain, 3)})"
        )
        lines.append(
            f"Structural Violations: {violation_label} "
            f"(violation_score={_fmt_num(violation_score, 3)})"
        )
        lines.append(
            "Feature Utility: "
            f"{_fmt_num(feature_utility_score, 1)}/100 "
            f"(decision_confidence={_fmt_num(decision_confidence, 3)})"
        )
        lines.append(
            f"Driver Signal: {signal_class} "
            f"(exogenous_dominance_ratio={_fmt_num(dominance_ratio, 3)}, "
            f"exogenous_r2={_fmt_num(exogenous_r2, 3)}, ar_r2={_fmt_num(ar_r2, 3)}, feature_count={feature_count})"
        )
        lines.append("")
        lines.append(f"RECOMMENDED APPROACH: {recommendation}")
        lines.append(f"ADD-FEATURES DECISION: {add_features_decision}")
        lines.append("")
        lines.append("INTERPRETATION:")
        if recommendation == "multivariate":
            lines.append("- Multivariate signal is strong and stable; proceed with multivariate modeling.")
        elif recommendation == "univariate_with_exogenous":
            lines.append("- Use univariate base model with selected external regressors.")
        else:
            lines.append("- Multivariate lift is not reliable; keep univariate baseline.")

        lines.append("")
        lines.append("STRUCTURAL FLAGS:")
        if flags:
            for flag in flags:
                lines.append(f"- {flag}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("RAW OUTPUT:")
        lines.append(f"- cross_lag_effect: {cross_lag_effect}")
        lines.append(f"- residual_dependency: {_fmt_num(residual_dependency, 6)}")
        lines.append(f"- cv_improvement_multivariate: {_fmt_num(cv_gain, 6)}")
        lines.append(f"- feature_utility_score: {_fmt_num(feature_utility_score, 6)}")
        lines.append(f"- add_features_decision: {add_features_decision}")
        lines.append(f"- decision_confidence: {_fmt_num(decision_confidence, 6)}")
        lines.append(f"- exogenous_r2: {_fmt_num(exogenous_r2, 6)}")
        lines.append(f"- ar_r2: {_fmt_num(ar_r2, 6)}")
        lines.append(f"- exogenous_dominance_ratio: {_fmt_num(dominance_ratio, 6)}")
        lines.append(f"- exogenous_signal_classification: {signal_class}")
        lines.append(f"- feature_count: {feature_count}")
        lines.append(f"- recommendation: {recommendation}")

        return "\n".join(lines)

    def diagnose(
        self,
        target: pd.Series,
        exog: pd.DataFrame,
        max_lag: int | None = None,
        significance: float = 0.05,
    ) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"statsmodels\.regression\.linear_model",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"numpy\.lib\._function_base_impl",
            )
            with np.errstate(all="ignore"):
                granger = self.granger_test(target=target, exog=exog, max_lag=max_lag, significance=significance)
                residual_dependency = self.residual_corr_test(target=target, exog=exog)
                cv_gain = self.cv_comparison(target=target, exog=exog)
                structural = self.structural_violation_check(target=target, exog=exog)
                exogenous_signal = self.exogenous_dominance_signal(target=target, exog=exog, ar_lag=5)

        recommendation = self._recommendation(
            cross_lag_effect=granger["cross_lag_effect"],
            residual_dependency=residual_dependency,
            cv_improvement_multivariate=cv_gain,
            violation_score=structural["violation_score"],
            exogenous_signal_classification=exogenous_signal["exogenous_signal_classification"],
        )
        feature_utility_score = self._feature_utility_score(
            cross_lag_effect=granger["cross_lag_effect"],
            residual_dependency=residual_dependency,
            cv_improvement_multivariate=cv_gain,
            violation_score=structural["violation_score"],
        )
        add_features_decision = self._add_features_decision(
            recommendation=recommendation,
            feature_utility_score=feature_utility_score,
            cv_improvement_multivariate=cv_gain,
        )
        decision_confidence = self._decision_confidence(
            feature_utility_score=feature_utility_score,
            cross_lag_effect=granger["cross_lag_effect"],
            violation_score=structural["violation_score"],
        )

        output = {
            "cross_lag_effect": granger["cross_lag_effect"],
            "residual_dependency": float(_safe_float(residual_dependency, default=np.nan)),
            "cv_improvement_multivariate": float(_safe_float(cv_gain, default=np.nan)),
            "feature_utility_score": float(_safe_float(feature_utility_score, default=np.nan)),
            "add_features_decision": add_features_decision,
            "decision_confidence": float(_safe_float(decision_confidence, default=np.nan)),
            "feature_count": float(_safe_float(exogenous_signal.get("feature_count", 0.0), default=0.0)),
            "exogenous_r2": float(_safe_float(exogenous_signal.get("exogenous_r2", np.nan), default=np.nan)),
            "ar_r2": float(_safe_float(exogenous_signal.get("ar_r2", np.nan), default=np.nan)),
            "exogenous_dominance_ratio": float(
                _safe_float(exogenous_signal.get("exogenous_dominance_ratio", np.nan), default=np.nan)
            ),
            "exogenous_signal_classification": exogenous_signal.get(
                "exogenous_signal_classification", "Autoregressive Dominated"
            ),
            "recommendation": recommendation,
            "details": {
                "granger": granger,
                "structural_violation": structural,
            },
        }
        output["human_readable_report"] = self._build_human_readable_report(output)
        return MultivariateDecisionReport(output)

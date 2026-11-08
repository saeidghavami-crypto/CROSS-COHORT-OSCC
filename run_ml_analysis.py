#!/usr/bin/env python3
"""Leakage-controlled cross-cohort oral microbiome machine-learning analysis.

This script implements the prespecified analysis requested by Dr. Saeid Ghavami:
three disease tasks, genus/species feature levels, nested leave-one-cohort-out
validation, clinical-data auditing, stable-feature analysis, cohort-aware
calibration, locked-pipeline permutation testing, and publication figures.

The script intentionally starts from relative-abundance tables. Taxon filtering,
zero replacement, CLR transformation, scaling, model tuning, calibration, and
threshold selection are fitted using training data only in every validation
iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 20260810
MIN_PREVALENCE = 0.10
BOOTSTRAPS = 1000
PERMUTATIONS = 500

TASK_LABELS = {
    "OLP": "Oral lichen planus vs healthy",
    "OPMD": "All OPMDs vs healthy",
    "OSCC": "OSCC vs healthy",
}
MODEL_LABELS = {
    "elastic_net": "Elastic net",
    "linear_svm": "Linear SVM",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}

# Significant directions reported in the uploaded manuscript's existing
# differential-abundance meta-analysis. These are used only for concordance
# annotation; they are not used for model training or feature selection.
REPORTED_META_SIGNATURES = {
    ("OPMD", "genus"): {
        "Propionibacterium": -1,
        "Lancefieldella": -1,
    },
    ("OPMD", "species"): {
        "Porphyromonas catoniae": 1,
        "Schaalia odontolytica": 1,
        "Serratia marcescens": 1,
        "Selenomonas felix": -1,
        "Megasphaera micronuciformis": -1,
        "Lachnoanaerobaculum orale": -1,
        "Hoylesella shahii": -1,
        "Corynebacterium matruchotii": -1,
    },
    ("OSCC", "genus"): {
        **{x: 1 for x in ["Peptoanaerobacter", "Metamycoplasma", "Dialister", "Peptostreptococcus", "Johnsonella", "Selenomonas", "Filifactor", "Parvimonas", "Treponema", "Catonella", "Eggerthella", "Fusobacterium"]},
        **{x: -1 for x in ["Nanosynococcus", "Actinomyces", "Nanosynbacter", "Nanogingivalis", "Moryella", "Cardiobacterium", "Corynebacterium", "Porphyromonas", "Leptotrichia", "Lancefieldella", "Oribacterium", "Rothia", "Moraxella", "Acinetobacter", "Haemophilus", "Riemerella", "Lautropia", "Lachnoanaerobaculum", "Neisseria", "Schaalia", "Arthrospira", "Granulicatella"]},
    },
    ("OSCC", "species"): {
        **{x: 1 for x in ["Catonella morbi", "Streptococcus gordonii", "Fusobacterium polymorphum", "Capnocytophaga sputigena", "Metamycoplasma salivarium", "Selenomonas sputigena", "Prevotella intermedia", "Parvimonas micra", "Eikenella corrodens", "Eggerthella catenaformis", "Desulfovibrio fairfieldensis", "Johnsonella ignava", "Hoylesella odenae"]},
        **{x: -1 for x in ["Hoylesella shahii", "Campylobacter concisus", "Leptotrichia hofstadii", "Prevotella aurantiaca", "Corynebacterium matruchotii", "Cardiobacterium valvarum", "Rothia dentocariosa", "Desulfomicrobium orale", "Streptococcus mutans", "Porphyromonas catoniae", "Capnocytophaga granulosa", "Oribacterium parvum", "Fusobacterium pseudoperiodonticum", "Fusobacterium nucleatum subsp. strictum", "Neisseria perflava"]},
    },
}


@dataclass
class CLRPreprocessor:
    features: list[str]
    pseudocount: float
    scaler: StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[ok] = tmp
    return out


def read_input_tables(input_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    paths = {
        ("OPMD", "genus"): input_dir / "OPMD_Genus_relative_abundance_OTU_table_with_meta_data(1).txt",
        ("OPMD", "species"): input_dir / "OPMD_Species_relative_abundance_OTU_table_with_meta_data(1).txt",
        ("OSCC", "genus"): input_dir / "OSCC_Genus_relative_abundance_OTU_table_with_meta_data(1).txt",
        ("OSCC", "species"): input_dir / "OSCC_Species_relative_abundance_OTU_table_with_meta_data(1).txt",
    }
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in paths.items():
        df = pd.read_csv(path, sep="\t", na_values=["NA", "N/A", "na", ""], low_memory=False)
        df["sampleid"] = df["sampleid"].astype(str)
        if df["sampleid"].duplicated().any():
            raise ValueError(f"Duplicate sample IDs detected in {path.name}")
        tables[key] = df

    # The OPMD species table does not contain the subtype column. Recover it by
    # exact sample-ID mapping from the uploaded genus table.
    subtype = tables[("OPMD", "genus")][["sampleid", "Disease_status_type"]]
    tables[("OPMD", "species")] = tables[("OPMD", "species")].merge(
        subtype, on="sampleid", how="left", validate="one_to_one"
    )
    return tables


def metadata_columns(df: pd.DataFrame) -> list[str]:
    known = [
        "sampleid",
        "Disease_status",
        "Host_age",
        "Host_sex",
        "Sample_type",
        "Geo_location",
        "Study_name",
        "Disease_status_type",
        "Smoking",
        "smoking",
    ]
    return [c for c in known if c in df.columns]


def harmonize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Host_age"] = pd.to_numeric(out.get("Host_age"), errors="coerce")
    sex = out.get("Host_sex", pd.Series(index=out.index, dtype=object)).astype("string").str.strip().str.lower()
    out["Sex_binary"] = sex.map({"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0})
    smoking_col = "Smoking" if "Smoking" in out.columns else "smoking" if "smoking" in out.columns else None
    if smoking_col is None:
        out["Smoking_binary"] = np.nan
    else:
        smoke = out[smoking_col].astype("string").str.strip().str.lower()
        out["Smoking_binary"] = smoke.map(
            {
                "never": 0.0,
                "no": 0.0,
                "non-smoker": 0.0,
                "nonsmoker": 0.0,
                "0": 0.0,
                "former": 1.0,
                "current": 1.0,
                "ever": 1.0,
                "yes": 1.0,
                "smoker": 1.0,
                "1": 1.0,
            }
        )
    return out


def prepare_task(
    tables: dict[tuple[str, str], pd.DataFrame], task: str, level: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    source = "OSCC" if task == "OSCC" else "OPMD"
    df = harmonize_metadata(tables[(source, level)])
    if task == "OLP":
        df = df[df["Disease_status_type"].isin(["H", "LP"])].copy()
        # Lee_2017 contributes healthy controls and non-OLP OPMDs, but no OLP
        # cases. It cannot be included in an OLP-vs-healthy cohort-wise task.
        olp_cohorts = df.loc[df["Disease_status_type"] == "LP", "Study_name"].unique()
        df = df[df["Study_name"].isin(olp_cohorts)].copy()
        y = (df["Disease_status_type"] == "LP").astype(int).to_numpy()
    elif task == "OPMD":
        y = (df["Disease_status"] == "OPMD").astype(int).to_numpy()
    elif task == "OSCC":
        y = (df["Disease_status"] == "OSCC").astype(int).to_numpy()
    else:
        raise ValueError(task)

    meta = df[
        [
            "sampleid",
            "Study_name",
            "Disease_status",
            "Host_age",
            "Sex_binary",
            "Smoking_binary",
            "Sample_type",
            "Geo_location",
        ]
        + (["Disease_status_type"] if "Disease_status_type" in df.columns else [])
    ].copy()
    taxa = [c for c in df.columns if c not in metadata_columns(df) + ["Sex_binary", "Smoking_binary"]]
    X = df[taxa].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(f"Non-numeric or missing abundance values in {task}/{level}")
    if not np.allclose(X.sum(axis=1).to_numpy(), 1.0, atol=1e-5):
        raise ValueError(f"Rows do not sum to one in {task}/{level}")
    groups = meta["Study_name"].astype(str).to_numpy()
    for cohort in np.unique(groups):
        yy = y[groups == cohort]
        if len(np.unique(yy)) < 2:
            raise ValueError(f"Cohort {cohort} lacks both classes for {task}/{level}")
    return meta.reset_index(drop=True), X.reset_index(drop=True), y, groups


def fit_clr_preprocessor(X: pd.DataFrame, prevalence: float = MIN_PREVALENCE) -> CLRPreprocessor:
    min_present = max(2, int(math.ceil(prevalence * len(X))))
    present = (X > 0).sum(axis=0) >= min_present
    variable = X.var(axis=0) > 0
    features = X.columns[present & variable].tolist()
    if not features:
        raise ValueError("No taxa survived training-only prevalence filtering")
    arr = X[features].to_numpy(dtype=float)
    positive = arr[arr > 0]
    pseudocount = max(float(np.min(positive) / 2.0), 1e-8) if positive.size else 1e-6
    clr = clr_transform_array(arr, pseudocount)
    scaler = StandardScaler().fit(clr)
    return CLRPreprocessor(features=features, pseudocount=pseudocount, scaler=scaler)


def clr_transform_array(arr: np.ndarray, pseudocount: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    arr = arr / row_sum
    arr = arr + pseudocount
    arr = arr / arr.sum(axis=1, keepdims=True)
    logged = np.log(arr)
    return logged - logged.mean(axis=1, keepdims=True)


def transform_clr(pre: CLRPreprocessor, X: pd.DataFrame) -> np.ndarray:
    arr = X.reindex(columns=pre.features, fill_value=0.0).to_numpy(dtype=float)
    clr = clr_transform_array(arr, pre.pseudocount)
    return pre.scaler.transform(clr)


def clinical_transform_fit(meta: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    age = meta["Host_age"].to_numpy(dtype=float)
    mean = float(np.mean(age))
    sd = float(np.std(age, ddof=0)) or 1.0
    arr = np.column_stack([(age - mean) / sd, meta["Sex_binary"].to_numpy(dtype=float)])
    return arr, {"age_mean": mean, "age_sd": sd}


def clinical_transform_apply(meta: pd.DataFrame, state: dict[str, float]) -> np.ndarray:
    return np.column_stack(
        [
            (meta["Host_age"].to_numpy(dtype=float) - state["age_mean"]) / state["age_sd"],
            meta["Sex_binary"].to_numpy(dtype=float),
        ]
    )


def fit_view(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    view: str,
) -> tuple[np.ndarray, dict, list[str]]:
    state: dict = {}
    names: list[str] = []
    parts: list[np.ndarray] = []
    if view in {"microbiome", "combined"}:
        state["clr"] = fit_clr_preprocessor(X)
        parts.append(transform_clr(state["clr"], X))
        names.extend(state["clr"].features)
    if view in {"clinical", "combined"}:
        clin, state["clinical"] = clinical_transform_fit(meta)
        parts.append(clin)
        names.extend(["Age_z", "Male_sex"])
    return np.column_stack(parts), state, names


def apply_view(X: pd.DataFrame, meta: pd.DataFrame, view: str, state: dict) -> np.ndarray:
    parts: list[np.ndarray] = []
    if view in {"microbiome", "combined"}:
        parts.append(transform_clr(state["clr"], X))
    if view in {"clinical", "combined"}:
        parts.append(clinical_transform_apply(meta, state["clinical"]))
    return np.column_stack(parts)


def parameter_grid(model_name: str) -> list[dict]:
    if model_name == "elastic_net":
        return [
            {"C": c, "l1_ratio": l1}
            for c, l1 in product([0.03, 0.3, 3.0], [0.25, 0.75])
        ]
    if model_name == "linear_svm":
        return [{"C": c} for c in [0.03, 0.3, 3.0, 30.0]]
    if model_name == "random_forest":
        return [
            {"max_features": mf, "min_samples_leaf": leaf}
            for mf, leaf in product(["sqrt", 0.30], [2, 5])
        ]
    if model_name == "xgboost":
        return [
            {"max_depth": depth, "learning_rate": lr}
            for depth, lr in product([2, 3], [0.05, 0.10])
        ]
    raise ValueError(model_name)


def build_model(model_name: str, params: dict, seed: int):
    if model_name == "elastic_net":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            class_weight="balanced",
            max_iter=6000,
            tol=1e-4,
            random_state=seed,
        )
    if model_name == "linear_svm":
        return SVC(
            kernel="linear",
            C=params["C"],
            class_weight="balanced",
            probability=False,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=seed,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=250,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=3,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=2,
            random_state=seed,
        )
    raise ValueError(model_name)


def fit_model(model_name: str, model, X: np.ndarray, y: np.ndarray):
    if model_name == "xgboost":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def raw_prediction(model_name: str, model, X: np.ndarray) -> np.ndarray:
    if model_name == "linear_svm":
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def calibration_input(model_name: str, raw: np.ndarray) -> np.ndarray:
    if model_name == "linear_svm":
        return raw.reshape(-1, 1)
    return logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)


def fit_calibrator(model_name: str, raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibrator.fit(calibration_input(model_name, raw), y)
    return calibrator


def apply_calibrator(model_name: str, calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(calibration_input(model_name, raw))[:, 1]


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = tpr[finite] - fpr[finite]
    candidates = thresholds[finite]
    best = np.flatnonzero(score == np.nanmax(score))
    # Among ties, choose the threshold closest to 0.5.
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def fit_predict_split(
    X_train: pd.DataFrame,
    meta_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    model_name: str,
    params: dict,
    view: str,
    seed: int,
):
    Xt, state, names = fit_view(X_train, meta_train, view)
    Xv = apply_view(X_test, meta_test, view, state)
    model = build_model(model_name, params, seed)
    fit_model(model_name, model, Xt, y_train)
    return raw_prediction(model_name, model, Xv), model, state, names, Xt, Xv


def inner_oof(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    params: dict,
    view: str,
    seed: int,
) -> np.ndarray:
    raw = np.full(len(y), np.nan)
    for i, held in enumerate(sorted(np.unique(groups))):
        tr = groups != held
        va = groups == held
        pred, *_ = fit_predict_split(
            X.loc[tr],
            meta.loc[tr],
            y[tr],
            X.loc[va],
            meta.loc[va],
            model_name,
            params,
            view,
            seed + i,
        )
        raw[va] = pred
    if np.isnan(raw).any():
        raise RuntimeError("Incomplete inner out-of-fold predictions")
    return raw


def nested_loco(
    meta: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    level: str,
    model_name: str,
    view: str,
    seed: int,
    bootstraps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metric_rows = []
    importance_rows = []

    for outer_i, held in enumerate(sorted(np.unique(groups))):
        tr = groups != held
        te = groups == held
        inner_groups = groups[tr]
        if len(np.unique(inner_groups)) < 2:
            raise ValueError(f"Nested LOCO requires at least two training cohorts; held={held}")

        scored = []
        inner_raw_by_params = {}
        for param_i, params in enumerate(parameter_grid(model_name)):
            raw = inner_oof(
                X.loc[tr].reset_index(drop=True),
                meta.loc[tr].reset_index(drop=True),
                y[tr],
                inner_groups,
                model_name,
                params,
                view,
                seed + outer_i * 100 + param_i,
            )
            auc = roc_auc_score(y[tr], raw)
            key = json.dumps(params, sort_keys=True)
            scored.append((auc, -param_i, params))
            inner_raw_by_params[key] = raw
        scored.sort(reverse=True, key=lambda z: (z[0], z[1]))
        best_auc, _, best_params = scored[0]
        best_raw = inner_raw_by_params[json.dumps(best_params, sort_keys=True)]
        calibrator = fit_calibrator(model_name, best_raw, y[tr])
        inner_p = apply_calibrator(model_name, calibrator, best_raw)
        threshold = choose_threshold(y[tr], inner_p)

        raw_test, model, state, names, Xt, Xv = fit_predict_split(
            X.loc[tr],
            meta.loc[tr],
            y[tr],
            X.loc[te],
            meta.loc[te],
            model_name,
            best_params,
            view,
            seed + outer_i * 1000,
        )
        p_test = apply_calibrator(model_name, calibrator, raw_test)
        yhat = (p_test >= threshold).astype(int)

        for idx, true, raw_value, prob, pred in zip(
            meta.index[te], y[te], raw_test, p_test, yhat
        ):
            prediction_rows.append(
                {
                    "task": task,
                    "level": level,
                    "model": model_name,
                    "view": view,
                    "held_out_cohort": held,
                    "sampleid": meta.loc[idx, "sampleid"],
                    "y_true": int(true),
                    "raw_score": float(raw_value),
                    "probability": float(prob),
                    "prediction": int(pred),
                    "threshold": threshold,
                }
            )

        met = compute_metrics(y[te], p_test, threshold)
        ci = bootstrap_metric_ci(y[te], p_test, threshold, bootstraps, seed + outer_i)
        metric_rows.append(
            {
                "task": task,
                "level": level,
                "model": model_name,
                "view": view,
                "held_out_cohort": held,
                "n_test": int(te.sum()),
                "n_cases": int(y[te].sum()),
                "n_controls": int(te.sum() - y[te].sum()),
                "features_retained": len(names),
                "inner_auc_selected": best_auc,
                "threshold": threshold,
                "selected_parameters": json.dumps(best_params, sort_keys=True),
                **met,
                **ci,
            }
        )

        values = extract_importance(model_name, model)
        for feature, value in zip(names, values):
            importance_rows.append(
                {
                    "task": task,
                    "level": level,
                    "model": model_name,
                    "view": view,
                    "held_out_cohort": held,
                    "feature": feature,
                    "importance": float(value),
                }
            )

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "auroc": roc_auc_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy_score(y, pred),
        "brier_score": brier_score_loss(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(x, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def stratified_boot_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    return np.concatenate(
        [rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)]
    )


def bootstrap_metric_ci(
    y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in ["auroc", "balanced_accuracy", "sensitivity", "specificity"]}
    for _ in range(n_boot):
        idx = stratified_boot_indices(y, rng)
        met = compute_metrics(y[idx], p[idx], threshold)
        for key in vals:
            vals[key].append(met[key])
    out = {}
    for key, arr in vals.items():
        out[f"{key}_ci_low"] = float(np.nanpercentile(arr, 2.5))
        out[f"{key}_ci_high"] = float(np.nanpercentile(arr, 97.5))
    return out


def extract_importance(model_name: str, model) -> np.ndarray:
    if model_name in {"elastic_net", "linear_svm"}:
        return np.asarray(model.coef_[0], dtype=float)
    return np.asarray(model.feature_importances_, dtype=float)


def summarize_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["task", "level", "model", "view"], as_index=False)
    rows = []
    for key, g in grouped:
        task, level, model, view = key
        rows.append(
            {
                "task": task,
                "level": level,
                "model": model,
                "view": view,
                "cohorts": len(g),
                "macro_auroc_mean": g["auroc"].mean(),
                "macro_auroc_sd": g["auroc"].std(ddof=1),
                "auroc_min": g["auroc"].min(),
                "auroc_max": g["auroc"].max(),
                "macro_balanced_accuracy_mean": g["balanced_accuracy"].mean(),
                "macro_sensitivity_mean": g["sensitivity"].mean(),
                "macro_specificity_mean": g["specificity"].mean(),
                "macro_brier_mean": g["brier_score"].mean(),
                "macro_calibration_intercept_mean": g["calibration_intercept"].mean(),
                "macro_calibration_slope_mean": g["calibration_slope"].mean(),
            }
        )
    return pd.DataFrame(rows)


def clinical_coverage(tables: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task in ["OLP", "OPMD", "OSCC"]:
        meta, _, y, groups = prepare_task(tables, task, "genus")
        for cohort in sorted(np.unique(groups)):
            mask = groups == cohort
            g = meta.loc[mask]
            rows.append(
                {
                    "task": task,
                    "cohort": cohort,
                    "n": int(mask.sum()),
                    "cases": int(y[mask].sum()),
                    "controls": int(mask.sum() - y[mask].sum()),
                    "age_available_n": int(g["Host_age"].notna().sum()),
                    "sex_available_n": int(g["Sex_binary"].notna().sum()),
                    "smoking_available_n": int(g["Smoking_binary"].notna().sum()),
                    "age_available_pct": 100 * g["Host_age"].notna().mean(),
                    "sex_available_pct": 100 * g["Sex_binary"].notna().mean(),
                    "smoking_available_pct": 100 * g["Smoking_binary"].notna().mean(),
                }
            )
    return pd.DataFrame(rows)


def hedges_g(x1: np.ndarray, x0: np.ndarray) -> tuple[float, float]:
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    v1, v0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    pooled_var = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
    if pooled_var <= 0:
        return np.nan, np.nan
    d = (np.mean(x1) - np.mean(x0)) / np.sqrt(pooled_var)
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    g = correction * d
    var_g = (n1 + n0) / (n1 * n0) + (g * g) / (2 * (n1 + n0 - 2))
    return float(g), float(var_g)


def random_effects(effects: list[float], variances: list[float]) -> dict[str, float]:
    e = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    ok = np.isfinite(e) & np.isfinite(v) & (v > 0)
    e, v = e[ok], v[ok]
    k = len(e)
    if k < 2:
        return {k: np.nan for k in ["effect", "se", "ci_low", "ci_high", "p_value", "i2", "tau2", "k"]} | {"k": k}
    w = 1 / v
    fixed = np.sum(w * e) / np.sum(w)
    q = np.sum(w * (e - fixed) ** 2)
    df = k - 1
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (v + tau2)
    effect = np.sum(wr * e) / np.sum(wr)
    se = np.sqrt(1 / np.sum(wr))
    z = effect / se if se > 0 else np.nan
    p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {
        "effect": effect,
        "se": se,
        "ci_low": effect - 1.96 * se,
        "ci_high": effect + 1.96 * se,
        "p_value": p,
        "i2": i2,
        "tau2": tau2,
        "k": k,
    }


def taxon_meta_analysis(
    meta: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, task: str, level: str
) -> pd.DataFrame:
    effects_by_feature: dict[str, list[float]] = {c: [] for c in X.columns}
    vars_by_feature: dict[str, list[float]] = {c: [] for c in X.columns}
    for cohort in sorted(np.unique(groups)):
        mask = groups == cohort
        arr = X.loc[mask].to_numpy(dtype=float)
        positive = arr[arr > 0]
        pseudo = max(float(np.min(positive) / 2), 1e-8) if positive.size else 1e-6
        clr = clr_transform_array(arr, pseudo)
        yy = y[mask]
        for j, feature in enumerate(X.columns):
            g, vg = hedges_g(clr[yy == 1, j], clr[yy == 0, j])
            effects_by_feature[feature].append(g)
            vars_by_feature[feature].append(vg)
    rows = []
    for feature in X.columns:
        re = random_effects(effects_by_feature[feature], vars_by_feature[feature])
        rows.append({"task": task, "level": level, "feature": feature, **re})
    out = pd.DataFrame(rows)
    out["fdr"] = bh_adjust(out["p_value"].to_numpy())
    return out.sort_values(["fdr", "p_value"], na_position="last")


def stability_summary(importances: pd.DataFrame, meta_results: pd.DataFrame) -> pd.DataFrame:
    data = importances[importances["model"] == "elastic_net"].copy()
    rows = []
    for (task, level, view, feature), g in data.groupby(["task", "level", "view", "feature"]):
        vals = g["importance"].to_numpy(dtype=float)
        rows.append(
            {
                "task": task,
                "level": level,
                "view": view,
                "feature": feature,
                "outer_folds_present": len(vals),
                "selection_frequency": np.mean(np.abs(vals) > 1e-8),
                "median_coefficient": np.median(vals),
                "coefficient_q1": np.quantile(vals, 0.25),
                "coefficient_q3": np.quantile(vals, 0.75),
                "direction_consistency": max(np.mean(vals > 1e-8), np.mean(vals < -1e-8)),
            }
        )
    out = pd.DataFrame(rows)
    out = out.merge(
        meta_results[
            ["task", "level", "feature", "effect", "ci_low", "ci_high", "p_value", "fdr", "i2", "k"]
        ],
        on=["task", "level", "feature"],
        how="left",
    )
    out["direction_agreement"] = np.sign(out["median_coefficient"]) == np.sign(out["effect"])
    out["reported_meta_analysis_direction"] = [
        REPORTED_META_SIGNATURES.get((task, level), {}).get(feature, np.nan)
        for task, level, feature in zip(out["task"], out["level"], out["feature"])
    ]
    out["agreement_with_reported_meta_analysis"] = np.where(
        out["reported_meta_analysis_direction"].notna(),
        np.sign(out["median_coefficient"]) == out["reported_meta_analysis_direction"],
        np.nan,
    )
    return out.sort_values(
        ["task", "level", "selection_frequency", "direction_consistency"], ascending=[True, True, False, False]
    )


def run_clinical_sensitivity(
    tables: dict[tuple[str, str], pd.DataFrame], seed: int, bootstraps: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta, X, y, groups = prepare_task(tables, "OPMD", "genus")
    complete = meta[["Host_age", "Sex_binary"]].notna().all(axis=1)
    meta = meta.loc[complete].reset_index(drop=True)
    X = X.loc[complete].reset_index(drop=True)
    y = y[complete.to_numpy()]
    groups = groups[complete.to_numpy()]
    if len(np.unique(groups)) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pred, metrics, imp = [], [], []
    for view in ["clinical", "microbiome", "combined"]:
        p, m, i = nested_loco(
            meta, X, y, groups, "OPMD_complete_case", "genus", "elastic_net", view, seed, bootstraps
        )
        pred.append(p)
        metrics.append(m)
        imp.append(i)
    return pd.concat(pred, ignore_index=True), pd.concat(metrics, ignore_index=True), pd.concat(imp, ignore_index=True)


def locked_pipeline_permutation(
    meta: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    real_metrics: pd.DataFrame,
    task: str,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    """Cohort-stratified locked-pipeline permutation for genus elastic net.

    Hyperparameters are locked to those selected in the real nested analysis.
    Labels are shuffled within cohorts, preserving cohort size and prevalence.
    """
    rng = np.random.default_rng(seed)
    observed = float(real_metrics["auroc"].mean())
    rows = [{"task": task, "permutation": -1, "macro_auroc": observed, "observed": True}]
    selected = {
        row.held_out_cohort: json.loads(row.selected_parameters)
        for row in real_metrics.itertuples()
    }
    # Pre-fit feature transformations by outer fold because they do not depend on labels.
    prepared = {}
    for held in sorted(np.unique(groups)):
        tr, te = groups != held, groups == held
        pre = fit_clr_preprocessor(X.loc[tr])
        prepared[held] = (
            tr,
            te,
            transform_clr(pre, X.loc[tr]),
            transform_clr(pre, X.loc[te]),
            selected[held],
        )
    for perm_i in range(n_perm):
        yp = y.copy()
        for cohort in np.unique(groups):
            idx = np.flatnonzero(groups == cohort)
            yp[idx] = rng.permutation(yp[idx])
        aucs = []
        for fold_i, held in enumerate(sorted(np.unique(groups))):
            tr, te, Xt, Xv, params = prepared[held]
            model = build_model("elastic_net", params, seed + perm_i * 17 + fold_i)
            fit_model("elastic_net", model, Xt, yp[tr])
            raw = raw_prediction("elastic_net", model, Xv)
            aucs.append(roc_auc_score(yp[te], raw))
        rows.append(
            {
                "task": task,
                "permutation": perm_i,
                "macro_auroc": float(np.mean(aucs)),
                "observed": False,
            }
        )
    return pd.DataFrame(rows)


def study_signal_audit(meta: pd.DataFrame, X: pd.DataFrame, seed: int) -> float:
    """Five-fold random-CV study classification; diagnostic, not disease prediction."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    labels = pd.Categorical(meta["Study_name"]).codes
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    preds = np.full(len(labels), -1)
    for tr, te in cv.split(X, labels):
        pre = fit_clr_preprocessor(X.loc[tr])
        Xt = transform_clr(pre, X.loc[tr])
        Xv = transform_clr(pre, X.loc[te])
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=seed
        ).fit(Xt, labels[tr])
        preds[te] = model.predict(Xv)
    return float(balanced_accuracy_score(labels, preds))


def setup_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=400, facecolor="white", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def figure_workflow_and_coverage(coverage: pd.DataFrame, fig_dir: Path):
    setup_style()
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.82, 1.18], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    boxes = [
        (0.02, "Relative-abundance\ntables"),
        (0.21, "Training-only\nfilter + CLR"),
        (0.40, "Nested leave-one-\ncohort-out tuning"),
        (0.60, "Untouched cohort\nexternal testing"),
        (0.79, "Calibration, stable\ntaxa, heterogeneity"),
    ]
    colors = ["#E8F1F5", "#DCEFE8", "#FFF0D8", "#FBE4E6", "#E8E4F3"]
    for i, ((x, label), color) in enumerate(zip(boxes, colors)):
        patch = FancyBboxPatch(
            (x, 0.28), 0.17, 0.44, boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes, facecolor=color, edgecolor="#375A6C", linewidth=1.2
        )
        ax.add_patch(patch)
        ax.text(x + 0.085, 0.50, label, ha="center", va="center", transform=ax.transAxes, fontsize=9.3)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.195, 0.50), xytext=(x + 0.17, 0.50),
                        xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.4, color="#375A6C"))
    ax.text(0.02, 0.88, "A  Leakage-controlled analysis design", transform=ax.transAxes, weight="bold", fontsize=11)

    ax2 = fig.add_subplot(gs[1])
    cov = coverage.copy()
    cov["row"] = cov["task"].map({"OLP": "OLP", "OPMD": "All OPMDs", "OSCC": "OSCC"}) + " | " + cov["cohort"].str.replace("_", " ")
    mat = cov.set_index("row")[["age_available_pct", "sex_available_pct", "smoking_available_pct"]]
    mat.columns = ["Age", "Sex", "Smoking"]
    sns.heatmap(mat, cmap="Blues", vmin=0, vmax=100, annot=True, fmt=".0f", cbar_kws={"label": "% available"}, ax=ax2,
                linewidths=0.5, linecolor="white")
    ax2.set_xlabel("Participant-level variable")
    ax2.set_ylabel("")
    ax2.set_title("B  Clinical-variable availability by prediction task and cohort", loc="left", weight="bold")
    save_figure(fig, fig_dir, "Figure_ML1_Workflow_and_clinical_coverage")


def figure_model_comparison(summary: pd.DataFrame, fig_dir: Path):
    setup_style()
    dat = summary[(summary["view"] == "microbiome")].copy()
    dat["analysis"] = dat["task"].map({"OLP": "OLP", "OPMD": "All OPMDs", "OSCC": "OSCC"}) + "\n" + dat["level"].str.title()
    pivot = dat.pivot(index="analysis", columns="model", values="macro_auroc_mean")
    order_rows = ["OLP\nGenus", "OLP\nSpecies", "All OPMDs\nGenus", "All OPMDs\nSpecies", "OSCC\nGenus", "OSCC\nSpecies"]
    order_cols = ["elastic_net", "linear_svm", "random_forest", "xgboost"]
    pivot = pivot.reindex(index=order_rows, columns=order_cols)
    pivot.columns = [MODEL_LABELS[c] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0.35, vmax=0.90, center=0.5,
                cmap=sns.diverging_palette(240, 15, as_cmap=True), linewidths=0.7, linecolor="white",
                cbar_kws={"label": "Mean held-out-cohort AUROC"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Cross-cohort discrimination varies by disease task and taxonomic level", loc="left", weight="bold", pad=12)
    save_figure(fig, fig_dir, "Figure_ML2_Model_comparison")


def figure_external_cohorts(metrics: pd.DataFrame, fig_dir: Path):
    setup_style()
    dat = metrics[(metrics["model"] == "elastic_net") & (metrics["level"] == "genus") & (metrics["view"] == "microbiome")]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1), sharey=True)
    colors = {"OLP": "#2A9D8F", "OPMD": "#E9C46A", "OSCC": "#D95F59"}
    for ax, task in zip(axes, ["OLP", "OPMD", "OSCC"]):
        g = dat[dat["task"] == task].sort_values("auroc")
        y_pos = np.arange(len(g))
        ax.errorbar(
            g["auroc"], y_pos,
            xerr=[g["auroc"] - g["auroc_ci_low"], g["auroc_ci_high"] - g["auroc"]],
            fmt="o", color=colors[task], ecolor=colors[task], capsize=3, ms=6
        )
        ax.axvline(0.5, color="#777777", ls="--", lw=1)
        ax.set_yticks(y_pos, g["held_out_cohort"].str.replace("_", " "))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Held-out AUROC")
        ax.set_title(TASK_LABELS[task], weight="bold")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Primary elastic-net model performance in untouched cohorts", x=0.02, ha="left", weight="bold", fontsize=11.5)
    save_figure(fig, fig_dir, "Figure_ML3_External_cohort_AUROC")


def figure_roc(predictions: pd.DataFrame, fig_dir: Path):
    setup_style()
    dat = predictions[(predictions["model"] == "elastic_net") & (predictions["level"] == "genus") & (predictions["view"] == "microbiome")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    colors = sns.color_palette("tab10", 6)
    for ax, task in zip(axes, ["OLP", "OPMD", "OSCC"]):
        g = dat[dat["task"] == task]
        for color, (cohort, h) in zip(colors, g.groupby("held_out_cohort")):
            fpr, tpr, _ = roc_curve(h["y_true"], h["probability"])
            auc = roc_auc_score(h["y_true"], h["probability"])
            ax.plot(fpr, tpr, lw=1.4, color=color, label=f"{cohort.replace('_',' ')} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("1 − specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(TASK_LABELS[task], weight="bold")
        ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.suptitle("Cohort-specific receiver-operating characteristics", x=0.02, ha="left", weight="bold", fontsize=11.5)
    save_figure(fig, fig_dir, "Figure_ML4_Cohort_ROC")


def figure_calibration(predictions: pd.DataFrame, fig_dir: Path):
    setup_style()
    dat = predictions[(predictions["model"] == "elastic_net") & (predictions["level"] == "genus") & (predictions["view"] == "microbiome")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    colors = {"OLP": "#2A9D8F", "OPMD": "#E9C46A", "OSCC": "#D95F59"}
    for ax, task in zip(axes, ["OLP", "OPMD", "OSCC"]):
        g = dat[dat["task"] == task].copy()
        g["bin"] = pd.qcut(g["probability"], q=min(8, g["probability"].nunique()), duplicates="drop")
        cal = g.groupby("bin", observed=True).agg(predicted=("probability", "mean"), observed=("y_true", "mean"), n=("y_true", "size"))
        ax.plot(cal["predicted"], cal["observed"], "o-", color=colors[task], lw=1.5, ms=5)
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event fraction")
        ax.set_title(TASK_LABELS[task], weight="bold")
    fig.suptitle("Descriptive calibration of pooled held-out predictions", x=0.02, ha="left", weight="bold", fontsize=11.5)
    save_figure(fig, fig_dir, "Figure_ML5_Calibration")


def figure_stable_taxa(stability: pd.DataFrame, fig_dir: Path):
    setup_style()
    dat = stability[(stability["level"] == "genus") & (stability["view"] == "microbiome")].copy()
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 11.5))
    for ax, task in zip(axes, ["OLP", "OPMD", "OSCC"]):
        g = dat[dat["task"] == task].copy()
        g["rank"] = g["selection_frequency"] * g["direction_consistency"] * np.abs(g["median_coefficient"])
        g = g.sort_values("rank", ascending=False).head(10).sort_values("median_coefficient")
        colors = np.where(g["median_coefficient"] >= 0, "#C95555", "#3A7CA5")
        ax.barh(g["feature"], g["median_coefficient"], color=colors, alpha=0.86)
        for y_pos, (_, row) in enumerate(g.iterrows()):
            span = max(np.abs(g["median_coefficient"]).max(), 0.01)
            offset = 0.025 * span
            ax.text(
                row["median_coefficient"] + (offset if row["median_coefficient"] >= 0 else -offset),
                y_pos,
                f"{row['selection_frequency']:.0%}",
                va="center",
                ha="left" if row["median_coefficient"] >= 0 else "right",
                fontsize=7.2,
            )
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlabel("Median elastic-net coefficient (label = outer-fold selection frequency)")
        ax.set_title(TASK_LABELS[task], weight="bold", loc="left")
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.28, hspace=0.52, top=0.93)
    fig.suptitle("Most stable genus-level bacterial features", x=0.02, ha="left", weight="bold", fontsize=11.5)
    save_figure(fig, fig_dir, "Figure_ML6_Stable_genus_features")


def figure_clinical_sensitivity(metrics: pd.DataFrame, fig_dir: Path):
    if metrics.empty:
        return
    setup_style()
    metrics = metrics.copy()
    metrics["cohort_label"] = metrics["held_out_cohort"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    order = sorted(metrics["cohort_label"].unique())
    palette = {"clinical": "#7A7A7A", "microbiome": "#2A9D8F", "combined": "#D95F59"}
    sns.pointplot(
        data=metrics,
        x="cohort_label",
        y="auroc",
        hue="view",
        order=order,
        hue_order=["clinical", "microbiome", "combined"],
        palette=palette,
        markers=["o", "s", "D"],
        linestyles=["--", "-", "-"],
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0.5, color="#777777", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Held-out cohort")
    ax.set_ylabel("AUROC")
    ax.set_title("Exploratory complete-case comparison in all OPMDs\n(age and sex available; smoking unavailable)", loc="left", weight="bold")
    ax.legend(title="Model", frameon=False, labels=["Clinical only", "Microbiome only", "Combined"])
    save_figure(fig, fig_dir, "Figure_ML7_Clinical_incremental_value")


def figure_permutations(permutations: pd.DataFrame, fig_dir: Path):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    colors = {"OLP": "#2A9D8F", "OPMD": "#E9C46A", "OSCC": "#D95F59"}
    for ax, task in zip(axes, ["OLP", "OPMD", "OSCC"]):
        g = permutations[(permutations["task"] == task) & (~permutations["observed"])]
        obs = permutations[(permutations["task"] == task) & (permutations["observed"])]["macro_auroc"].iloc[0]
        sns.histplot(g["macro_auroc"], bins=25, color=colors[task], alpha=0.75, ax=ax)
        ax.axvline(obs, color="#111111", lw=2, label=f"Observed {obs:.2f}")
        p = (1 + (g["macro_auroc"] >= obs).sum()) / (1 + len(g))
        ax.set_xlabel("Permuted macro-AUROC")
        ax.set_ylabel("Count")
        ax.set_title(f"{TASK_LABELS[task]}\np = {p:.3f}", weight="bold")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Cohort-stratified locked-pipeline permutation tests", x=0.02, ha="left", weight="bold", fontsize=11.5)
    save_figure(fig, fig_dir, "Figure_ML8_Permutation_tests")


def write_run_manifest(out_dir: Path, args: argparse.Namespace, tables: dict):
    manifest = {
        "date": pd.Timestamp.now(tz="America/Winnipeg").isoformat(),
        "seed": args.seed,
        "bootstraps": args.bootstraps,
        "permutations": args.permutations,
        "minimum_training_prevalence": MIN_PREVALENCE,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input_dimensions": {f"{k[0]}_{k[1]}": list(v.shape) for k, v in tables.items()},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = args.output_dir
    table_dir = out_dir / "tables"
    pred_dir = out_dir / "predictions"
    fig_dir = out_dir / "figures"
    for d in [out_dir, table_dir, pred_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tables = read_input_tables(args.input_dir)
    coverage = clinical_coverage(tables)
    coverage.to_csv(table_dir / "Table_ML1_Clinical_metadata_coverage.csv", index=False)

    all_predictions, all_metrics, all_importances, all_meta = [], [], [], []
    study_audit_rows = []
    for task in ["OLP", "OPMD", "OSCC"]:
        for level in ["genus", "species"]:
            print(f"Preparing {task}/{level}", flush=True)
            meta, X, y, groups = prepare_task(tables, task, level)
            study_audit_rows.append(
                {
                    "task": task,
                    "level": level,
                    "study_classification_balanced_accuracy": study_signal_audit(meta, X, args.seed),
                    "chance_level": 1 / len(np.unique(groups)),
                }
            )
            meta_analysis = taxon_meta_analysis(meta, X, y, groups, task, level)
            all_meta.append(meta_analysis)
            for model_name in ["elastic_net", "linear_svm", "random_forest", "xgboost"]:
                print(f"  Nested LOCO {model_name}", flush=True)
                pred, metrics, imp = nested_loco(
                    meta,
                    X,
                    y,
                    groups,
                    task,
                    level,
                    model_name,
                    "microbiome",
                    args.seed,
                    args.bootstraps,
                )
                all_predictions.append(pred)
                all_metrics.append(metrics)
                all_importances.append(imp)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    importances = pd.concat(all_importances, ignore_index=True)
    meta_results = pd.concat(all_meta, ignore_index=True)

    print("Running complete-case clinical sensitivity analysis", flush=True)
    c_pred, c_metrics, c_imp = run_clinical_sensitivity(tables, args.seed, args.bootstraps)
    if not c_pred.empty:
        predictions = pd.concat([predictions, c_pred], ignore_index=True)
        metrics = pd.concat([metrics, c_metrics], ignore_index=True)
        importances = pd.concat([importances, c_imp], ignore_index=True)

    predictions.to_csv(pred_dir / "All_outer_fold_predictions.csv", index=False)
    metrics.to_csv(table_dir / "Table_ML2_External_cohort_performance.csv", index=False)
    summary = summarize_performance(metrics)
    summary.to_csv(table_dir / "Table_ML3_Macro_performance_summary.csv", index=False)
    importances.to_csv(table_dir / "All_outer_fold_feature_importances.csv", index=False)
    meta_results.to_csv(table_dir / "All_taxa_random_effects_meta_analysis.csv", index=False)
    stability = stability_summary(importances, meta_results)
    stability.to_csv(table_dir / "Table_ML4_Elastic_net_feature_stability.csv", index=False)
    pd.DataFrame(study_audit_rows).to_csv(table_dir / "Table_ML5_Residual_study_signal.csv", index=False)

    print("Running locked-pipeline permutation tests", flush=True)
    perm_frames = []
    for task in ["OLP", "OPMD", "OSCC"]:
        meta, X, y, groups = prepare_task(tables, task, "genus")
        real = metrics[
            (metrics["task"] == task)
            & (metrics["level"] == "genus")
            & (metrics["model"] == "elastic_net")
            & (metrics["view"] == "microbiome")
        ]
        perm_frames.append(
            locked_pipeline_permutation(meta, X, y, groups, real, task, args.permutations, args.seed)
        )
    permutations = pd.concat(perm_frames, ignore_index=True)
    permutations.to_csv(table_dir / "Table_ML6_Permutation_results.csv", index=False)
    perm_summary = []
    for task in ["OLP", "OPMD", "OSCC"]:
        g = permutations[permutations["task"] == task]
        obs = g[g["observed"]]["macro_auroc"].iloc[0]
        null = g[~g["observed"]]["macro_auroc"]
        perm_summary.append(
            {
                "task": task,
                "observed_macro_auroc": obs,
                "null_mean": null.mean(),
                "null_sd": null.std(ddof=1),
                "empirical_p_value": (1 + (null >= obs).sum()) / (1 + len(null)),
                "permutations": len(null),
            }
        )
    pd.DataFrame(perm_summary).to_csv(table_dir / "Table_ML7_Permutation_summary.csv", index=False)

    figure_workflow_and_coverage(coverage, fig_dir)
    figure_model_comparison(summary, fig_dir)
    figure_external_cohorts(metrics, fig_dir)
    figure_roc(predictions, fig_dir)
    figure_calibration(predictions, fig_dir)
    figure_stable_taxa(stability, fig_dir)
    figure_clinical_sensitivity(c_metrics, fig_dir)
    figure_permutations(permutations, fig_dir)
    write_run_manifest(out_dir, args, tables)
    print("Analysis complete", flush=True)


if __name__ == "__main__":
    main()

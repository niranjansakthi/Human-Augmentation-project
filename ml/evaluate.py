"""
evaluate.py
===========
Model Evaluation & Reporting
for the AI-Assisted Human Augmentation System.

PURPOSE
-------
Loads saved model artifacts and produces a detailed evaluation report.
Computes and displays all classification and regression metrics.

USAGE
-----
    python ml/evaluate.py

OUTPUT
------
Prints full evaluation report to console.
Reads saved metrics from backend/models/model_metrics.json.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from preprocess import (
    ALL_FEATURES,
    TARGET_ASSISTANCE,
    TARGET_MOVEMENT,
    build_feature_matrix,
    load_label_encoder,
    load_scaler,
)

# -- Paths ---------------------------------------------------------------------
DATA_CSV      = Path(__file__).parent.parent / "backend" / "data" / "synthetic_dataset.csv"
MODELS_DIR    = Path(__file__).parent.parent / "backend" / "models"
METRICS_FILE  = MODELS_DIR / "model_metrics.json"
FEAT_IMP_FILE = MODELS_DIR / "feature_importance.json"

MOVEMENT_CLASSES = [
    "REST", "WALKING", "SIT_TO_STAND", "STAND_TO_SIT", "KNEE_FLEXION", "KNEE_EXTENSION"
]

RANDOM_SEED = 42
TEST_SIZE   = 0.20

ASSISTANCE_CATEGORIES = [
    (0,  20,  "Minimal"),
    (21, 40,  "Low"),
    (41, 60,  "Moderate"),
    (61, 80,  "High"),
    (81, 100, "Very High"),
]


def assistance_category(pct: float) -> str:
    for lo, hi, name in ASSISTANCE_CATEGORIES:
        if lo <= pct <= hi:
            return name
    return "Very High"


def print_separator(char="-", width=60):
    print(char * width)


def evaluate():
    print("=" * 60)
    print("Model Evaluation Report")
    print("AI-Assisted Human Augmentation System  |  SIH 2026")
    print("=" * 60)
    print()
    print("[DISCLAIMER] Evaluation on SYNTHETIC test data only.")
    print("  Metrics do not represent real-world performance.")
    print("  High scores are expected because data is synthetic.")
    print()

    # -- Load artifacts --------------------------------------------------------
    print("Loading model artifacts ...")
    clf  = joblib.load(MODELS_DIR / "movement_classifier.joblib")
    reg  = joblib.load(MODELS_DIR / "assistance_regressor.joblib")
    ohe  = joblib.load(MODELS_DIR / "movement_ohe.joblib")
    le   = load_label_encoder()
    scaler = load_scaler()
    print("  All artifacts loaded.")
    print()

    # -- Load and transform data -----------------------------------------------
    df = pd.read_csv(DATA_CSV)
    X, y_movement, y_assistance = build_feature_matrix(df)
    y_encoded = le.transform(y_movement)

    # -- Reproduce same splits -------------------------------------------------
    X_train, X_test, y_train_mv, y_test_mv = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )
    _, _, y_train_as, y_test_as = train_test_split(
        X, y_assistance.values, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    X_test_sc = scaler.transform(X_test)

    # -- Movement Classifier Evaluation ---------------------------------------
    print_separator("=")
    print("SECTION 1 — MOVEMENT CLASSIFICATION MODEL")
    print("  Model: RandomForestClassifier")
    print_separator("=")
    print()

    y_pred_mv = clf.predict(X_test_sc)
    acc = accuracy_score(y_test_mv, y_pred_mv)

    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print("  Classification Report:")
    print(classification_report(y_test_mv, y_pred_mv, target_names=le.classes_))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test_mv, y_pred_mv)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    print()

    # Class probabilities on test set
    proba = clf.predict_proba(X_test_sc)
    mean_confidence = proba.max(axis=1).mean()
    print(f"  Mean prediction confidence : {mean_confidence:.4f}  ({mean_confidence*100:.2f}%)")
    print()

    # -- Assistance Regressor Evaluation --------------------------------------
    print_separator("=")
    print("SECTION 2 — ASSISTANCE LEVEL ESTIMATION MODEL")
    print("  Model: RandomForestRegressor")
    print_separator("=")
    print()

    # Rebuild augmented test features
    X_test_orig  = X.iloc[X_train.shape[0]:]   # approximate — use same indices
    # Proper: replicate exact split for assistance
    Xa_train, Xa_test, yam_train, yam_test = train_test_split(
        X, y_movement, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    Xa_test_sc     = scaler.transform(Xa_test)
    movement_ohe_t = ohe.transform(yam_test.to_numpy(dtype=str).reshape(-1, 1))
    X_aug_test     = np.hstack([Xa_test_sc, movement_ohe_t])

    y_pred_as = reg.predict(X_aug_test)
    y_pred_as = np.clip(y_pred_as, 0.0, 100.0)

    _, _, ya_train_as, ya_test_as = train_test_split(
        X, y_assistance.values, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    mae  = mean_absolute_error(ya_test_as, y_pred_as)
    rmse = np.sqrt(mean_squared_error(ya_test_as, y_pred_as))
    r2   = r2_score(ya_test_as, y_pred_as)

    print(f"  MAE  : {mae:.4f}  (mean absolute error in % assistance)")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    print()

    y_test_cat = pd.Series(ya_test_as).apply(assistance_category)
    y_pred_cat = pd.Series(y_pred_as).apply(assistance_category)
    cat_acc = (y_test_cat == y_pred_cat).mean()
    print(f"  Category Match Accuracy : {cat_acc:.4f}  ({cat_acc*100:.2f}%)")
    print()

    print("  Per-Category Breakdown:")
    for cat in ["Minimal", "Low", "Moderate", "High", "Very High"]:
        mask = y_test_cat == cat
        if mask.sum() == 0:
            continue
        c_mae = mean_absolute_error(ya_test_as[mask.values], y_pred_as[mask.values])
        c_r2  = r2_score(ya_test_as[mask.values], y_pred_as[mask.values]) if mask.sum() > 1 else float("nan")
        print(f"    {cat:<12} | n={mask.sum():>4} | MAE={c_mae:.2f}% | R²={c_r2:.3f}")
    print()

    # -- Feature Importance ----------------------------------------------------
    if FEAT_IMP_FILE.exists():
        with open(FEAT_IMP_FILE) as f:
            feat_imp = json.load(f)

        print_separator("=")
        print("SECTION 3 — FEATURE IMPORTANCE")
        print_separator("=")
        print()

        if "movement_classifier" in feat_imp:
            print("  Top-10 Features — Movement Classifier:")
            for i, (feat, imp) in enumerate(feat_imp["movement_classifier"].items()):
                if i >= 10:
                    break
                bar = "#" * int(imp * 200)
                print(f"  {i+1:>2}. {feat:<38} {imp:.4f}  {bar}")
            print()

        if "assistance_regressor" in feat_imp:
            print("  Top-10 Features — Assistance Regressor:")
            for i, (feat, imp) in enumerate(feat_imp["assistance_regressor"].items()):
                if i >= 10:
                    break
                bar = "#" * int(imp * 200)
                print(f"  {i+1:>2}. {feat:<38} {imp:.4f}  {bar}")
            print()

    # -- Saved Metrics Summary -------------------------------------------------
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            metrics = json.load(f)

        print_separator("=")
        print("SECTION 4 — SAVED METRICS SUMMARY")
        print_separator("=")
        print()
        print(json.dumps(metrics, indent=2))
        print()

    print_separator("=")
    print("Evaluation complete.")
    print()
    print("[IMPORTANT] Results above are from SYNTHETIC test data.")
    print("  Synthetic data can produce artificially high scores.")
    print("  Real sensor data will yield different (typically lower) metrics.")
    print("  This prototype is for engineering demonstration, not clinical use.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()

"""
train_assistance_model.py
=========================
Assistance Level Estimation Model Training
for the AI-Assisted Human Augmentation System.

MODEL
-----
RandomForestRegressor — estimates a continuous assistance percentage (0–100).
The output is then bucketed into 5 prototype engineering categories.

TASK
----
Input  : engineered sensor features + one-hot encoded movement state
Output : recommended assistance level (0–100 %)

BUCKETS (prototype engineering categories, NOT clinically validated):
  0–20  → Minimal assistance
  21–40 → Low assistance
  41–60 → Moderate assistance
  61–80 → High assistance
  81–100→ Very high assistance

DISCLAIMER
----------
Model trained on SYNTHETIC data only.
Assistance recommendations are prototype design outputs, not medical decisions.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))

from preprocess import (
    ALL_FEATURES,
    TARGET_MOVEMENT,
    TARGET_ASSISTANCE,
    build_feature_matrix,
    load_scaler,
    load_label_encoder,
)

# -- Config --------------------------------------------------------------------
RANDOM_SEED   = 42
TEST_SIZE     = 0.20
DATA_CSV      = Path(__file__).parent.parent / "backend" / "data" / "synthetic_dataset.csv"
MODELS_DIR    = Path(__file__).parent.parent / "backend" / "models"
METRICS_FILE  = MODELS_DIR / "model_metrics.json"
FEAT_IMP_FILE = MODELS_DIR / "feature_importance.json"

ASSISTANCE_CATEGORIES = [
    (0,  20,  "Minimal"),
    (21, 40,  "Low"),
    (41, 60,  "Moderate"),
    (61, 80,  "High"),
    (81, 100, "Very High"),
]

RF_PARAMS = dict(
    n_estimators     = 150,
    max_depth        = 12,
    min_samples_split = 8,
    min_samples_leaf  = 5,
    max_features     = "sqrt",
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

MOVEMENT_CLASSES = [
    "REST", "WALKING", "SIT_TO_STAND", "STAND_TO_SIT", "KNEE_FLEXION", "KNEE_EXTENSION"
]


def assistance_category(pct: float) -> str:
    """Return the prototype engineering category name for an assistance percentage."""
    for lo, hi, name in ASSISTANCE_CATEGORIES:
        if lo <= pct <= hi:
            return name
    return "Very High"


def train_assistance_regressor():
    print("=" * 60)
    print("Assistance Level Estimation Model Training")
    print("AI-Assisted Human Augmentation System  |  SIH 2026")
    print("=" * 60)
    print()
    print("[NOTE] Training on SYNTHETIC data only.")
    print("       Recommendations are prototype outputs, NOT medical decisions.")
    print()

    # -- 1. Load data ----------------------------------------------------------
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_CSV}. "
            "Run ml/generate_dataset.py first."
        )
    print(f"Loading dataset : {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  Rows: {len(df)}  |  Columns: {df.shape[1]}")
    print()

    # -- 2. Feature engineering ------------------------------------------------
    print("Building feature matrix ...")
    X, y_movement, y_assistance = build_feature_matrix(df)
    print(f"  Feature matrix : {X.shape}")
    print(f"  Assistance range: {y_assistance.min():.1f} – {y_assistance.max():.1f}")
    print()

    # -- 3. One-hot encode movement state and append ---------------------------
    ohe = OneHotEncoder(
        categories=[MOVEMENT_CLASSES],
        sparse_output=False,
        handle_unknown="ignore",
    )
    movement_ohe = ohe.fit_transform(y_movement.to_numpy(dtype=str).reshape(-1, 1))
    ohe_cols = [f"movement_{cls}" for cls in MOVEMENT_CLASSES]
    movement_df = pd.DataFrame(movement_ohe, columns=ohe_cols, index=X.index)

    X_augmented = pd.concat([X, movement_df], axis=1)
    all_feature_names = ALL_FEATURES + ohe_cols

    print(f"  Augmented feature matrix: {X_augmented.shape}")
    print()

    # -- 4. Load scaler (fitted during movement training) ---------------------
    # We need to scale only the original features, OHE columns stay as-is
    try:
        scaler = load_scaler()
        X_scaled = scaler.transform(X)
        X_final = np.hstack([X_scaled, movement_ohe])
        print("  Using existing feature scaler.")
    except FileNotFoundError:
        print("  [WARN] Scaler not found — fitting a new one (run movement trainer first).")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_final = np.hstack([X_scaled, movement_ohe])
    print()

    # -- 5. Train/test split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_assistance.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test  set: {len(X_test)} samples")
    print()

    # -- 6. Train model --------------------------------------------------------
    print("Training RandomForestRegressor ...")
    print(f"  Parameters: {RF_PARAMS}")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    print("  Training complete.")
    print()

    # -- 7. Evaluate -----------------------------------------------------------
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 100.0)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("-" * 60)
    print("REGRESSION METRICS (Test Set)")
    print("-" * 60)
    print(f"  MAE  : {mae:.4f}  (mean absolute error, in % assistance)")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    print()

    # Category accuracy (bucket the predictions)
    y_test_cat = pd.Series(y_test).apply(assistance_category)
    y_pred_cat = pd.Series(y_pred).apply(assistance_category)
    cat_acc = (y_test_cat == y_pred_cat).mean()
    print(f"  Category accuracy (bucket match): {cat_acc:.4f}  ({cat_acc*100:.2f}%)")
    print()

    # Per-category stats
    print("Per-Category Prediction Summary:")
    test_cats = sorted(set(y_test_cat))
    for cat in ["Minimal", "Low", "Moderate", "High", "Very High"]:
        mask = y_test_cat == cat
        if mask.sum() == 0:
            continue
        cat_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        print(f"  {cat:<12} | samples={mask.sum():>4} | MAE={cat_mae:.2f}%")
    print()

    # Cross-validation
    print("5-Fold Cross-Validation (R²) ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_r2 = cross_val_score(model, X_final, y_assistance.values, cv=kf, scoring="r2", n_jobs=-1)
    print(f"  CV R² : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print()

    # -- 8. Feature importance -------------------------------------------------
    importances = model.feature_importances_
    feat_imp = {
        feat: round(float(imp), 6)
        for feat, imp in zip(all_feature_names, importances)
    }
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    print("Top-10 Feature Importances (Assistance Regressor):")
    for i, (feat, imp) in enumerate(feat_imp_sorted.items()):
        if i >= 10:
            break
        print(f"  {i+1:>2}. {feat:<40} {imp:.4f}")
    print()

    # -- 9. Save model + OHE ---------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "assistance_regressor.joblib"
    ohe_path   = MODELS_DIR / "movement_ohe.joblib"
    joblib.dump(model, model_path)
    joblib.dump(ohe,   ohe_path)
    print(f"  Model saved   : {model_path}")
    print(f"  OHE saved     : {ohe_path}")

    # -- 10. Save metrics ------------------------------------------------------
    metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            metrics = json.load(f)

    metrics["assistance_regressor"] = {
        "disclaimer": (
            "Trained on SYNTHETIC data. Recommendations are prototype engineering outputs. "
            "Not a medical device. Not clinically validated."
        ),
        "model_type"    : "RandomForestRegressor",
        "n_estimators"  : RF_PARAMS["n_estimators"],
        "test_mae"      : round(float(mae),  4),
        "test_rmse"     : round(float(rmse), 4),
        "test_r2"       : round(float(r2),   4),
        "category_accuracy": round(float(cat_acc), 4),
        "cv_r2_mean"    : round(float(cv_r2.mean()), 4),
        "cv_r2_std"     : round(float(cv_r2.std()),  4),
        "assistance_categories": [
            {"name": name, "range": f"{lo}–{hi}%"}
            for lo, hi, name in ASSISTANCE_CATEGORIES
        ],
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved : {METRICS_FILE}")

    # Save/merge feature importance
    feat_imp_data = {}
    if FEAT_IMP_FILE.exists():
        with open(FEAT_IMP_FILE) as f:
            feat_imp_data = json.load(f)

    feat_imp_data["assistance_regressor"] = feat_imp_sorted

    with open(FEAT_IMP_FILE, "w") as f:
        json.dump(feat_imp_data, f, indent=2)
    print(f"  Feature imp.  : {FEAT_IMP_FILE}")
    print()
    print("[REMINDER] Output is 'prototype model recommendation', not a medical prescription.")
    print("=" * 60)

    return model, ohe, scaler


if __name__ == "__main__":
    train_assistance_regressor()

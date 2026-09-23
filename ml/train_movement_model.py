"""
train_movement_model.py
=======================
Movement Classification Model Training
for the AI-Assisted Human Augmentation System.

MODEL
-----
RandomForestClassifier — chosen because it is:
  - interpretable (feature_importances_)
  - robust to scale (no normalisation strictly needed, but applied for consistency)
  - fast to train on tabular data
  - appropriate for 6-class classification on prototype data

TASK
----
Input  : engineered sensor features
Output : movement state label (one of 6 classes)

DISCLAIMER
----------
Model trained on SYNTHETIC data only.
High accuracy on synthetic data does NOT imply real-world performance.
Not a clinically validated system.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add parent to path so we can import ml modules directly
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import (
    ALL_FEATURES,
    TARGET_MOVEMENT,
    build_feature_matrix,
    fit_and_save_scaler,
    fit_and_save_label_encoder,
)

# -- Config --------------------------------------------------------------------
RANDOM_SEED  = 42
TEST_SIZE    = 0.20          # 80/20 train/test split
DATA_CSV     = Path(__file__).parent.parent / "backend" / "data" / "synthetic_dataset.csv"
MODELS_DIR   = Path(__file__).parent.parent / "backend" / "models"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
FEAT_IMP_FILE = MODELS_DIR / "feature_importance.json"

RF_PARAMS = dict(
    n_estimators    = 150,
    max_depth       = 12,
    min_samples_split = 8,
    min_samples_leaf  = 5,
    max_features    = "sqrt",
    class_weight    = "balanced",
    random_state    = RANDOM_SEED,
    n_jobs          = -1,
)


def train_movement_classifier():
    print("=" * 60)
    print("Movement Classification Model Training")
    print("AI-Assisted Human Augmentation System  |  SIH 2026")
    print("=" * 60)
    print()
    print("[NOTE] Training on SYNTHETIC data only.")
    print("       High accuracy may reflect dataset simplicity,")
    print("       NOT real-world performance.")
    print()

    # -- 1. Load data ----------------------------------------------------------
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_CSV}. "
            "Run ml/generate_dataset.py first."
        )
    print(f"Loading dataset: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  Rows: {len(df)}  |  Columns: {df.shape[1]}")
    print()

    # -- 2. Feature engineering ------------------------------------------------
    print("Building feature matrix ...")
    X, y_movement, _ = build_feature_matrix(df)
    print(f"  Feature matrix : {X.shape}")
    print(f"  Class counts   :")
    for cls, cnt in y_movement.value_counts().items():
        print(f"    {cls:<20} {cnt}")
    print()

    # -- 3. Encode labels ------------------------------------------------------
    le = fit_and_save_label_encoder(y_movement)
    y_encoded = le.transform(y_movement)

    # -- 4. Train/test split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test  set: {len(X_test)} samples")
    print()

    # -- 5. Scale features -----------------------------------------------------
    scaler = fit_and_save_scaler(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print()

    # -- 6. Train model --------------------------------------------------------
    print("Training RandomForestClassifier ...")
    print(f"  Parameters: {RF_PARAMS}")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_sc, y_train)
    print("  Training complete.")
    print()

    # -- 7. Evaluate -----------------------------------------------------------
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)

    print("-" * 60)
    print("EVALUATION RESULTS (Test Set)")
    print("-" * 60)
    print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print("Classification Report:")
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=False
    )
    print(report)

    report_dict = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    print()

    # Cross-validation
    print("5-Fold Cross-Validation on full dataset ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    X_all_sc = scaler.transform(X)
    cv_scores = cross_val_score(model, X_all_sc, y_encoded, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print()

    # -- 8. Feature importance -------------------------------------------------
    importances = model.feature_importances_
    feat_imp = {
        feat: round(float(imp), 6)
        for feat, imp in zip(ALL_FEATURES, importances)
    }
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    print("Top-10 Feature Importances (Movement Classifier):")
    for i, (feat, imp) in enumerate(feat_imp_sorted.items()):
        if i >= 10:
            break
        print(f"  {i+1:>2}. {feat:<35} {imp:.4f}")
    print()

    # -- 9. Save model ---------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "movement_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved   : {model_path}")

    # -- 10. Save metrics ------------------------------------------------------
    metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            metrics = json.load(f)

    metrics["movement_classifier"] = {
        "disclaimer": (
            "Trained on SYNTHETIC data. High accuracy reflects dataset simplicity, "
            "not real-world performance. Not clinically validated."
        ),
        "model_type"  : "RandomForestClassifier",
        "n_estimators": RF_PARAMS["n_estimators"],
        "test_accuracy": round(float(acc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std" : round(float(cv_scores.std()),  4),
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall"   : round(report_dict[cls]["recall"],    4),
                "f1_score" : round(report_dict[cls]["f1-score"],  4),
                "support"  : int(report_dict[cls]["support"]),
            }
            for cls in le.classes_
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(le.classes_),
        "n_features" : len(ALL_FEATURES),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved : {METRICS_FILE}")

    # Save/merge feature importance
    feat_imp_data = {}
    if FEAT_IMP_FILE.exists():
        with open(FEAT_IMP_FILE) as f:
            feat_imp_data = json.load(f)

    feat_imp_data["movement_classifier"] = feat_imp_sorted

    with open(FEAT_IMP_FILE, "w") as f:
        json.dump(feat_imp_data, f, indent=2)
    print(f"  Feature imp.  : {FEAT_IMP_FILE}")
    print()
    print("[REMINDER] Accuracy on synthetic data may not represent real sensor performance.")
    print("=" * 60)

    return model, le, scaler


if __name__ == "__main__":
    train_movement_classifier()

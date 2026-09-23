"""
test_model.py
=============
Manual Testing & Overfitting/Underfitting Diagnostics
for the AI-Assisted Human Augmentation System.

PURPOSE
-------
This file contains:
  1. 10 hand-crafted test samples (edge cases + typical cases)
  2. Prediction on each sample with full output
  3. Train vs Test gap analysis (overfitting check)
  4. Learning curve analysis (underfitting check)
  5. Per-class performance analysis
  6. Out-of-distribution detection (stress test)
  7. Clear pass/fail verdict

IMPORTANT DISCLAIMER
--------------------
All 10 test samples are SYNTHETIC, manually crafted values.
They do NOT represent real sensor data from any person.
Results are for engineering prototype evaluation only.

HOW TO RUN
----------
    python ml/test_model.py

    # Or run specific sections only:
    python ml/test_model.py --section samples     # only the 10 samples
    python ml/test_model.py --section overfit     # only overfitting analysis
    python ml/test_model.py --section learning    # only learning curves
    python ml/test_model.py --section all         # everything (default)

WHAT TO LOOK FOR
----------------
  Overfitting  : Train accuracy >> Test accuracy (gap > 5%)
  Underfitting : Both Train and Test accuracy are low (< 70%)
  Good fit     : Train ~ Test accuracy, both > 85% (for synthetic data)

OUTPUT FILES
------------
  ml/test_results.json  — machine-readable test results
  ml/learning_curve.png — learning curve plot (if matplotlib available)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    ALL_FEATURES,
    TARGET_ASSISTANCE,
    TARGET_MOVEMENT,
    build_feature_matrix,
    compute_single_snapshot_features,
    load_label_encoder,
    load_scaler,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "backend" / "models"
DATA_CSV   = Path(__file__).parent.parent / "backend" / "data" / "synthetic_dataset.csv"
RESULTS_JSON = Path(__file__).parent / "test_results.json"

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


# =============================================================================
# 10 HAND-CRAFTED TEST SAMPLES
# =============================================================================
# Design rationale for each sample is documented inline.
# Values are chosen to cover:
#   - Typical cases for each movement class
#   - Boundary / edge-case values
#   - High-fatigue modifiers
#   - Mixed-signal ambiguous cases (stress test)
#
# Expected label is provided for verification.
# (If the model disagrees, that is NOT necessarily wrong — it shows model logic.)

MANUAL_TEST_SAMPLES = [
    # ------------------------------------------------------------------
    # Sample 1: Clear REST — all signals near-zero, gravity-only accel
    # Expected: REST, Minimal assistance
    # ------------------------------------------------------------------
    {
        "id": 1,
        "description": "Typical rest/idle state — user sitting still",
        "expected_movement": "REST",
        "expected_assistance_range": (0, 20),
        "expected_category": "Minimal",
        "sensor": {
            "knee_angle"               :  4.0,
            "knee_angular_velocity"    :  0.1,
            "knee_angular_acceleration":  0.0,
            "force"                    :  0.04,
            "acceleration_x"           :  0.05,
            "acceleration_y"           :  0.02,
            "acceleration_z"           :  9.80,
            "gyroscope_x"              :  0.3,
            "gyroscope_y"              :  0.1,
            "gyroscope_z"              :  0.0,
            "fatigue_indicator"        :  0.04,
        },
    },

    # ------------------------------------------------------------------
    # Sample 2: Classic walking gait at moderate pace
    # Expected: WALKING, Low-Moderate assistance
    # ------------------------------------------------------------------
    {
        "id": 2,
        "description": "Standard walking gait — moderate pace, healthy user",
        "expected_movement": "WALKING",
        "expected_assistance_range": (10, 45),
        "expected_category": "Low",
        "sensor": {
            "knee_angle"               : 34.0,
            "knee_angular_velocity"    : 58.0,
            "knee_angular_acceleration": 76.0,
            "force"                    :  0.38,
            "acceleration_x"           :  1.40,
            "acceleration_y"           :  0.45,
            "acceleration_z"           :  9.45,
            "gyroscope_x"              : 28.0,
            "gyroscope_y"              :  9.5,
            "gyroscope_z"              :  4.5,
            "fatigue_indicator"        :  0.12,
        },
    },

    # ------------------------------------------------------------------
    # Sample 3: Sit-to-Stand with high joint load
    # Expected: SIT_TO_STAND, High assistance
    # ------------------------------------------------------------------
    {
        "id": 3,
        "description": "Sit-to-Stand — strong knee extension force required",
        "expected_movement": "SIT_TO_STAND",
        "expected_assistance_range": (45, 90),
        "expected_category": "High",
        "sensor": {
            "knee_angle"               : 80.0,
            "knee_angular_velocity"    :-95.0,
            "knee_angular_acceleration":-158.0,
            "force"                    :  0.85,
            "acceleration_x"           :  0.55,
            "acceleration_y"           :  1.90,
            "acceleration_z"           :  8.30,
            "gyroscope_x"              : 22.0,
            "gyroscope_y"              : 36.0,
            "gyroscope_z"              :  6.5,
            "fatigue_indicator"        :  0.22,
        },
    },

    # ------------------------------------------------------------------
    # Sample 4: Stand-to-Sit controlled lowering
    # Expected: STAND_TO_SIT, Low-Moderate assistance
    # ------------------------------------------------------------------
    {
        "id": 4,
        "description": "Stand-to-Sit — controlled descent, normal effort",
        "expected_movement": "STAND_TO_SIT",
        "expected_assistance_range": (20, 55),
        "expected_category": "Moderate",
        "sensor": {
            "knee_angle"               : 62.0,
            "knee_angular_velocity"    : 77.0,
            "knee_angular_acceleration": 118.0,
            "force"                    :  0.52,
            "acceleration_x"           :  0.38,
            "acceleration_y"           :  1.15,
            "acceleration_z"           :  8.85,
            "gyroscope_x"              : 17.0,
            "gyroscope_y"              : 24.0,
            "gyroscope_z"              :  4.0,
            "fatigue_indicator"        :  0.17,
        },
    },

    # ------------------------------------------------------------------
    # Sample 5: Knee Flexion exercise — rehabilitation movement
    # Expected: KNEE_FLEXION, Low assistance
    # ------------------------------------------------------------------
    {
        "id": 5,
        "description": "Knee flexion exercise — rehabilitation, normal effort",
        "expected_movement": "KNEE_FLEXION",
        "expected_assistance_range": (10, 40),
        "expected_category": "Low",
        "sensor": {
            "knee_angle"               : 56.0,
            "knee_angular_velocity"    :-68.0,
            "knee_angular_acceleration":-88.0,
            "force"                    :  0.28,
            "acceleration_x"           :  0.18,
            "acceleration_y"           :  0.75,
            "acceleration_z"           :  9.25,
            "gyroscope_x"              :  9.5,
            "gyroscope_y"              : 19.5,
            "gyroscope_z"              :  2.8,
            "fatigue_indicator"        :  0.18,
        },
    },

    # ------------------------------------------------------------------
    # Sample 6: Knee Extension exercise — rehabilitation movement
    # Expected: KNEE_EXTENSION, Low assistance
    # ------------------------------------------------------------------
    {
        "id": 6,
        "description": "Knee extension exercise — rehabilitation, normal effort",
        "expected_movement": "KNEE_EXTENSION",
        "expected_assistance_range": (10, 40),
        "expected_category": "Low",
        "sensor": {
            "knee_angle"               : 34.0,
            "knee_angular_velocity"    : 63.0,
            "knee_angular_acceleration": 82.0,
            "force"                    :  0.26,
            "acceleration_x"           :  0.18,
            "acceleration_y"           :  0.65,
            "acceleration_z"           :  9.30,
            "gyroscope_x"              :  8.5,
            "gyroscope_y"              : 17.0,
            "gyroscope_z"              :  2.5,
            "fatigue_indicator"        :  0.16,
        },
    },

    # ------------------------------------------------------------------
    # Sample 7: EDGE CASE — Very high fatigue during walking
    # Expected: WALKING, Moderate-High assistance (fatigue drives this up)
    # ------------------------------------------------------------------
    {
        "id": 7,
        "description": "EDGE CASE: Walking under extreme fatigue — assistance should escalate",
        "expected_movement": "WALKING",
        "expected_assistance_range": (35, 80),
        "expected_category": "Moderate",
        "sensor": {
            "knee_angle"               : 30.0,
            "knee_angular_velocity"    : 44.0,
            "knee_angular_acceleration": 62.0,
            "force"                    :  0.35,
            "acceleration_x"           :  1.10,
            "acceleration_y"           :  0.38,
            "acceleration_z"           :  9.50,
            "gyroscope_x"              : 25.0,
            "gyroscope_y"              :  8.5,
            "gyroscope_z"              :  3.8,
            "fatigue_indicator"        :  0.95,   # <-- extreme fatigue
        },
    },

    # ------------------------------------------------------------------
    # Sample 8: EDGE CASE — Sit-to-Stand with maximum effort + fatigue
    # Expected: SIT_TO_STAND, Very High assistance
    # ------------------------------------------------------------------
    {
        "id": 8,
        "description": "EDGE CASE: Sit-to-Stand with high force AND high fatigue",
        "expected_movement": "SIT_TO_STAND",
        "expected_assistance_range": (60, 100),
        "expected_category": "High",
        "sensor": {
            "knee_angle"               : 82.0,
            "knee_angular_velocity"    :-103.0,
            "knee_angular_acceleration":-172.0,
            "force"                    :  0.92,   # <-- near max force
            "acceleration_x"           :  0.72,
            "acceleration_y"           :  2.15,
            "acceleration_z"           :  8.10,
            "gyroscope_x"              : 26.0,
            "gyroscope_y"              : 40.0,
            "gyroscope_z"              :  7.5,
            "fatigue_indicator"        :  0.88,   # <-- high fatigue
        },
    },

    # ------------------------------------------------------------------
    # Sample 9: AMBIGUOUS CASE — Between KNEE_FLEXION and STAND_TO_SIT
    # Knee angle and velocity overlap both classes — tests model decision boundary
    # Expected: either KNEE_FLEXION or STAND_TO_SIT (either is acceptable)
    # ------------------------------------------------------------------
    {
        "id": 9,
        "description": "AMBIGUOUS: Overlapping signals — KNEE_FLEXION vs STAND_TO_SIT boundary",
        "expected_movement": None,   # either is acceptable
        "expected_assistance_range": (15, 55),
        "expected_category": None,   # flexible
        "sensor": {
            "knee_angle"               : 65.0,    # STS-range angle
            "knee_angular_velocity"    :-65.0,    # flexion-direction velocity
            "knee_angular_acceleration":-85.0,
            "force"                    :  0.33,
            "acceleration_x"           :  0.25,
            "acceleration_y"           :  0.90,
            "acceleration_z"           :  9.10,
            "gyroscope_x"              : 12.0,
            "gyroscope_y"              : 22.0,
            "gyroscope_z"              :  3.5,
            "fatigue_indicator"        :  0.20,
        },
    },

    # ------------------------------------------------------------------
    # Sample 10: NEAR-ZERO movement / boundary of REST
    # Very small motion — tests whether model correctly stays near REST
    # Expected: REST (or minimal movement), Minimal assistance
    # ------------------------------------------------------------------
    {
        "id": 10,
        "description": "BOUNDARY: Micro-movement — just above noise floor, near REST",
        "expected_movement": "REST",
        "expected_assistance_range": (0, 20),
        "expected_category": "Minimal",
        "sensor": {
            "knee_angle"               :  6.5,
            "knee_angular_velocity"    :  2.1,    # slight sway
            "knee_angular_acceleration":  1.5,
            "force"                    :  0.07,
            "acceleration_x"           :  0.15,
            "acceleration_y"           :  0.08,
            "acceleration_z"           :  9.78,
            "gyroscope_x"              :  1.5,
            "gyroscope_y"              :  0.5,
            "gyroscope_z"              :  0.3,
            "fatigue_indicator"        :  0.08,
        },
    },
]


# =============================================================================
# SECTION 1: Run 10 Manual Samples
# =============================================================================

def run_manual_samples(clf, reg, ohe, le, scaler) -> list[dict]:
    """Run all 10 manual test samples and print detailed results."""

    print("=" * 70)
    print("SECTION 1: 10 MANUAL TEST SAMPLES")
    print("  [All inputs are SYNTHETIC demonstration values]")
    print("=" * 70)

    results = []
    passed  = 0
    total   = len(MANUAL_TEST_SAMPLES)

    for s in MANUAL_TEST_SAMPLES:
        sid    = s["id"]
        desc   = s["description"]
        sensor = s["sensor"]
        exp_mv = s["expected_movement"]
        exp_rng= s["expected_assistance_range"]

        # Feature engineering
        all_feats = compute_single_snapshot_features(sensor)
        X = pd.DataFrame([all_feats])[ALL_FEATURES]
        X_sc = scaler.transform(X)

        # Movement prediction
        pred_enc   = clf.predict(X_sc)[0]
        probas     = clf.predict_proba(X_sc)[0]
        movement   = le.inverse_transform([pred_enc])[0]
        confidence = float(probas.max())

        # Assistance prediction
        mv_ohe     = ohe.transform([[movement]])
        X_aug      = np.hstack([X_sc, mv_ohe])
        assist_pct = float(np.clip(reg.predict(X_aug)[0], 0.0, 100.0))
        cat        = assistance_category(assist_pct)

        # Verdict
        mv_ok  = (exp_mv is None) or (movement == exp_mv)
        as_ok  = exp_rng[0] <= assist_pct <= exp_rng[1]
        ok     = mv_ok and as_ok
        if ok:
            passed += 1

        status_icon = "PASS" if ok else "FAIL"

        print(f"\n  Sample {sid:>2} | {status_icon} | {desc}")
        print(f"  {'─'*64}")
        print(f"  Predicted Movement : {movement:<20}  (confidence: {confidence*100:.1f}%)")
        if exp_mv:
            exp_icon = "OK" if mv_ok else "!!"
            print(f"  Expected Movement  : {exp_mv:<20}  [{exp_icon}]")
        else:
            print(f"  Expected Movement  : (any — ambiguous case)       [OK]")

        print(f"  Assistance         : {assist_pct:.1f}%  ({cat})")
        print(f"  Expected Range     : {exp_rng[0]}–{exp_rng[1]}%         [{'OK' if as_ok else '!!'}]")

        # Top 3 contributing features
        fi = dict(sorted(
            zip(ALL_FEATURES, clf.feature_importances_),
            key=lambda x: -x[1]
        ))
        top3 = list(fi.items())[:3]
        print(f"  Top Features       : " + " | ".join(
            f"{f}={all_feats[f]:.2f}(imp={i:.3f})" for f, i in top3
        ))

        # Probability breakdown (top 3 classes)
        proba_sorted = sorted(
            zip(le.classes_, probas), key=lambda x: -x[1]
        )[:3]
        print(f"  Probabilities      : " + "  ".join(
            f"{c}={p*100:.1f}%" for c, p in proba_sorted
        ))

        results.append({
            "id"              : sid,
            "description"     : desc,
            "predicted_movement" : movement,
            "expected_movement"  : exp_mv,
            "confidence"         : round(confidence, 4),
            "assistance_pct"     : round(assist_pct, 2),
            "assistance_category": cat,
            "expected_range"     : list(exp_rng),
            "movement_pass"      : mv_ok,
            "assistance_pass"    : as_ok,
            "overall_pass"       : ok,
        })

    print(f"\n  {'='*70}")
    print(f"  MANUAL SAMPLE RESULTS: {passed}/{total} passed")
    pct = passed / total * 100
    print(f"  Pass Rate: {pct:.1f}%")
    if pct == 100:
        print("  -> All samples passed. Model is well-calibrated for typical cases.")
    elif pct >= 80:
        print("  -> Most samples passed. A few edge cases may need attention.")
    else:
        print("  -> Several samples failed. Review model training or test expectations.")
    print(f"  {'='*70}")

    return results


# =============================================================================
# SECTION 2: Overfitting / Underfitting Analysis
# =============================================================================

def analyse_overfit(clf, reg, ohe, X, y_mv_enc, y_as, le, scaler) -> dict:
    """
    Compare Train vs Test accuracy to detect overfitting or underfitting.

    Criterion:
      - Overfitting  : train_acc - test_acc > 0.05  (5 percentage points)
      - Underfitting : test_acc < 0.70
      - Good fit     : gap <= 0.05 AND test_acc >= 0.70
    """
    print("\n" + "=" * 70)
    print("SECTION 2: OVERFITTING / UNDERFITTING ANALYSIS")
    print("=" * 70)

    X_sc = scaler.transform(X)

    # Classification ── train vs test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_mv_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_mv_enc
    )
    clf.fit(X_tr, y_tr)           # re-fit on same split for fair comparison
    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    test_acc  = accuracy_score(y_te, clf.predict(X_te))
    gap_clf   = train_acc - test_acc

    print(f"\n  MOVEMENT CLASSIFIER (RandomForestClassifier)")
    print(f"  {'─'*50}")
    print(f"  Train Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
    print(f"  Test  Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Train-Test Gap : {gap_clf:.4f}  ({gap_clf*100:.2f} pp)")

    clf_verdict = diagnose_fit(train_acc, test_acc, gap_clf, "classifier")

    # 5-fold cross-validation generalisation check
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(clf, X_sc, y_mv_enc, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  5-Fold CV Mean : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  CV Fold Scores : {[round(s, 4) for s in cv_scores]}")

    # Regression ── train vs test
    mv_ohe_all = ohe.transform(le.inverse_transform(y_mv_enc).reshape(-1, 1))
    X_aug_all  = np.hstack([X_sc, mv_ohe_all])
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_aug_all, y_as, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    reg.fit(Xr_tr, yr_tr)
    train_mae = mean_absolute_error(yr_tr, np.clip(reg.predict(Xr_tr), 0, 100))
    test_mae  = mean_absolute_error(yr_te, np.clip(reg.predict(Xr_te), 0, 100))
    train_r2  = r2_score(yr_tr, np.clip(reg.predict(Xr_tr), 0, 100))
    test_r2   = r2_score(yr_te, np.clip(reg.predict(Xr_te), 0, 100))
    gap_reg   = train_r2 - test_r2

    print(f"\n  ASSISTANCE REGRESSOR (RandomForestRegressor)")
    print(f"  {'─'*50}")
    print(f"  Train MAE : {train_mae:.4f}%  |  R2 : {train_r2:.4f}")
    print(f"  Test  MAE : {test_mae:.4f}%  |  R2 : {test_r2:.4f}")
    print(f"  R2 Gap    : {gap_reg:.4f}  (train-test)")

    reg_verdict = diagnose_fit_reg(train_r2, test_r2, gap_reg)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_r2 = cross_val_score(reg, X_aug_all, y_as, cv=kf, scoring="r2", n_jobs=-1)
    print(f"  5-Fold CV R2   : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"  CV Fold R2     : {[round(s, 4) for s in cv_r2]}")

    return {
        "classifier": {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy" : round(test_acc, 4),
            "gap"           : round(gap_clf, 4),
            "cv_mean"       : round(cv_scores.mean(), 4),
            "cv_std"        : round(cv_scores.std(), 4),
            "verdict"       : clf_verdict,
        },
        "regressor": {
            "train_mae" : round(train_mae, 4),
            "test_mae"  : round(test_mae, 4),
            "train_r2"  : round(train_r2, 4),
            "test_r2"   : round(test_r2, 4),
            "gap_r2"    : round(gap_reg, 4),
            "cv_r2_mean": round(cv_r2.mean(), 4),
            "cv_r2_std" : round(cv_r2.std(), 4),
            "verdict"   : reg_verdict,
        },
    }


def diagnose_fit(train_acc, test_acc, gap, model_name) -> str:
    if test_acc < 0.70:
        verdict = "UNDERFITTING"
        print(f"  [!!] VERDICT: {verdict} — test accuracy {test_acc*100:.1f}% is too low.")
        print(f"       Action : Increase model complexity, add more features, check data.")
    elif gap > 0.05:
        verdict = "OVERFITTING"
        print(f"  [!!] VERDICT: {verdict} — train-test gap {gap*100:.2f} pp is large.")
        print(f"       Action : Increase min_samples_leaf, reduce max_depth, add more data.")
    else:
        verdict = "GOOD_FIT"
        print(f"  [OK] VERDICT: {verdict} — gap {gap*100:.2f} pp is acceptable.")
        print(f"       Note   : Scores on synthetic data are inherently higher than real data.")
    return verdict


def diagnose_fit_reg(train_r2, test_r2, gap) -> str:
    if test_r2 < 0.50:
        verdict = "UNDERFITTING"
        print(f"  [!!] VERDICT: {verdict} — test R2={test_r2:.3f} is low.")
    elif gap > 0.10:
        verdict = "OVERFITTING"
        print(f"  [!!] VERDICT: {verdict} — R2 gap {gap:.3f} is large.")
    else:
        verdict = "GOOD_FIT"
        print(f"  [OK] VERDICT: {verdict} — R2 gap {gap:.3f} is acceptable.")
    return verdict


# =============================================================================
# SECTION 3: Learning Curve Analysis
# =============================================================================

def run_learning_curve(clf, X, y_mv_enc, scaler) -> dict:
    """
    Generate learning curve data.
    Plots accuracy vs training set size.
    Rising test curve = model is still learning (good).
    Converged gap = overfitting plateau.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: LEARNING CURVE ANALYSIS")
    print("  Tests how model accuracy changes with more training data.")
    print("=" * 70)

    X_sc = scaler.transform(X)
    train_sizes = np.linspace(0.10, 1.0, 8)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        clf, X_sc, y_mv_enc,
        train_sizes=train_sizes,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    print(f"\n  {'Samples':>10}  {'Train Acc':>12}  {'Val Acc':>12}  {'Gap':>8}")
    print(f"  {'─'*50}")
    for n, tr, vl in zip(train_sizes_abs, train_mean, val_mean):
        gap = tr - vl
        bar_t = "#" * int(tr * 30)
        bar_v = "#" * int(vl * 30)
        flag  = " [!!]" if gap > 0.08 else ""
        print(f"  {int(n):>10}  {tr:>12.4f}  {vl:>12.4f}  {gap:>8.4f}{flag}")

    # Interpret trend
    early_gap = train_mean[1] - val_mean[1]
    final_gap = train_mean[-1] - val_mean[-1]
    trend_improving = val_mean[-1] > val_mean[1]

    print()
    if trend_improving:
        print("  [OK] Validation accuracy IMPROVES with more data — model is learning.")
    else:
        print("  [!!] Validation accuracy NOT improving — possible underfitting.")

    if final_gap < 0.05:
        print("  [OK] Final train-val gap is small — no significant overfitting.")
    elif final_gap < 0.10:
        print("  [~]  Final gap is moderate — minor overfitting, acceptable for prototype.")
    else:
        print("  [!!] Final gap is large — consider regularisation or more data.")

    # Try to save plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std, train_mean + train_std,
            alpha=0.15, color="#2196F3"
        )
        ax.fill_between(
            train_sizes_abs,
            val_mean - val_std, val_mean + val_std,
            alpha=0.15, color="#4CAF50"
        )
        ax.plot(train_sizes_abs, train_mean, "o-", color="#2196F3",
                linewidth=2, label="Training accuracy")
        ax.plot(train_sizes_abs, val_mean,   "s-", color="#4CAF50",
                linewidth=2, label="Validation accuracy (CV)")
        ax.set_xlabel("Training set size (samples)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve — Movement Classifier\n[Synthetic Data]")
        ax.legend(loc="lower right")
        ax.set_ylim(0.6, 1.05)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.5, 0.01,
            "[Synthetic data only — not a real-world performance indicator]",
            transform=ax.transAxes, ha="center", fontsize=7, color="gray"
        )
        out_png = Path(__file__).parent / "learning_curve.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Learning curve plot saved: {out_png}")
    except Exception as e:
        print(f"\n  [NOTE] Could not generate plot: {e}")
        print(f"         Run: pip install matplotlib")

    return {
        "train_sizes"       : [int(n) for n in train_sizes_abs],
        "train_accuracy_mean": [round(v, 4) for v in train_mean],
        "val_accuracy_mean" : [round(v, 4) for v in val_mean],
        "train_accuracy_std": [round(v, 4) for v in train_std],
        "val_accuracy_std"  : [round(v, 4) for v in val_std],
        "final_train_acc"   : round(float(train_mean[-1]), 4),
        "final_val_acc"     : round(float(val_mean[-1]),   4),
        "final_gap"         : round(float(final_gap),      4),
        "trend_improving"   : bool(trend_improving),
    }


# =============================================================================
# SECTION 4: Per-Class Deep Dive
# =============================================================================

def per_class_analysis(clf, X, y_mv_enc, le, scaler) -> dict:
    """Show per-class precision/recall/F1 on the held-out test set."""

    print("\n" + "=" * 70)
    print("SECTION 4: PER-CLASS PERFORMANCE ANALYSIS")
    print("  Checks whether any movement class is misclassified more than others.")
    print("=" * 70)

    X_sc = scaler.transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_mv_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_mv_enc
    )

    y_pred = clf.predict(X_te)
    report_dict = {}
    print()
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    from sklearn.metrics import classification_report as cr
    report = cr(y_te, y_pred, target_names=le.classes_, output_dict=True)

    print("  Classes where recall < 0.85 (higher risk of missed detection):")
    found_weak = False
    for cls in le.classes_:
        r = report[cls]["recall"]
        if r < 0.85:
            print(f"  [!!] {cls:<20} recall={r:.3f}")
            found_weak = True
    if not found_weak:
        print("  [OK] All classes have recall >= 0.85.")

    return report


# =============================================================================
# SECTION 5: Out-of-Distribution Stress Test
# =============================================================================

def ood_stress_test(clf, reg, ohe, le, scaler) -> list[dict]:
    """
    Feed extreme / out-of-distribution sensor values.
    The model should still return a valid class (graceful degradation),
    even if confidence is low.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: OUT-OF-DISTRIBUTION STRESS TEST")
    print("  Tests model behaviour on extreme / unusual sensor values.")
    print("  Expects: valid class returned, possibly low confidence.")
    print("=" * 70)

    ood_samples = [
        {
            "name": "Zero sensors (all zeros)",
            "sensor": {k: 0.0 for k in [
                "knee_angle", "knee_angular_velocity", "knee_angular_acceleration",
                "force", "acceleration_x", "acceleration_y", "acceleration_z",
                "gyroscope_x", "gyroscope_y", "gyroscope_z", "fatigue_indicator"
            ]},
        },
        {
            "name": "Maximum-range values",
            "sensor": {
                "knee_angle": 130.0, "knee_angular_velocity": 199.0,
                "knee_angular_acceleration": 499.0, "force": 1.0,
                "acceleration_x": 14.9, "acceleration_y": 14.9, "acceleration_z": 14.9,
                "gyroscope_x": 179.0, "gyroscope_y": 179.0, "gyroscope_z": 179.0,
                "fatigue_indicator": 1.0,
            },
        },
        {
            "name": "Negative knee angle (hyperextension)",
            "sensor": {
                "knee_angle": -8.0, "knee_angular_velocity": -5.0,
                "knee_angular_acceleration": -3.0, "force": 0.08,
                "acceleration_x": 0.05, "acceleration_y": 0.02, "acceleration_z": 9.81,
                "gyroscope_x": 0.5, "gyroscope_y": 0.2, "gyroscope_z": 0.1,
                "fatigue_indicator": 0.05,
            },
        },
    ]

    results = []
    for s in ood_samples:
        feats = compute_single_snapshot_features(s["sensor"])
        X = pd.DataFrame([feats])[ALL_FEATURES]
        X_sc = scaler.transform(X)
        pred_enc   = clf.predict(X_sc)[0]
        probas     = clf.predict_proba(X_sc)[0]
        movement   = le.inverse_transform([pred_enc])[0]
        confidence = float(probas.max())

        mv_ohe = ohe.transform([[movement]])
        X_aug  = np.hstack([X_sc, mv_ohe])
        assist = float(np.clip(reg.predict(X_aug)[0], 0.0, 100.0))

        conf_flag = "OK" if confidence > 0.40 else "LOW_CONF"
        print(f"\n  [{conf_flag}] {s['name']}")
        print(f"         -> Predicted: {movement}  ({confidence*100:.1f}% confidence)")
        print(f"         -> Assistance: {assist:.1f}%")
        results.append({
            "name": s["name"], "movement": movement,
            "confidence": round(confidence, 4), "assistance": round(assist, 2),
        })

    print("\n  [NOTE] Model must return a valid class even for out-of-distribution inputs.")
    print("         Low confidence on extreme values is EXPECTED and CORRECT behaviour.")
    return results


# =============================================================================
# MAIN
# =============================================================================

def load_artifacts():
    """Load all model artifacts."""
    clf    = joblib.load(MODELS_DIR / "movement_classifier.joblib")
    reg    = joblib.load(MODELS_DIR / "assistance_regressor.joblib")
    ohe    = joblib.load(MODELS_DIR / "movement_ohe.joblib")
    le     = load_label_encoder()
    scaler = load_scaler()
    return clf, reg, ohe, le, scaler


def main(section: str = "all"):
    print()
    print("=" * 70)
    print("  AI-ASSISTED HUMAN AUGMENTATION SYSTEM  |  SIH 2026")
    print("  ML MODEL TESTING & VALIDATION REPORT")
    print("=" * 70)
    print("  [DISCLAIMER] All values are SYNTHETIC. Not real sensor data.")
    print("  [PURPOSE]    Engineering prototype validation only.")
    print()

    # Load artifacts
    print("Loading model artifacts ...")
    clf, reg, ohe, le, scaler = load_artifacts()

    # Load dataset
    df = pd.read_csv(DATA_CSV)
    from preprocess import build_feature_matrix
    X, y_movement, y_assistance = build_feature_matrix(df)
    y_mv_enc = le.transform(y_movement)
    y_as     = y_assistance.values

    all_results = {}

    if section in ("all", "samples"):
        all_results["manual_samples"] = run_manual_samples(clf, reg, ohe, le, scaler)

    if section in ("all", "overfit"):
        all_results["overfit_analysis"] = analyse_overfit(
            clf, reg, ohe, X, y_mv_enc, y_as, le, scaler
        )

    if section in ("all", "learning"):
        all_results["learning_curve"] = run_learning_curve(clf, X, y_mv_enc, scaler)

    if section in ("all",):
        all_results["per_class"] = per_class_analysis(clf, X, y_mv_enc, le, scaler)
        all_results["ood_stress"] = ood_stress_test(clf, reg, ohe, le, scaler)

    # Save results
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {RESULTS_JSON}")

    # Final summary
    print()
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    if "manual_samples" in all_results:
        passed = sum(1 for r in all_results["manual_samples"] if r["overall_pass"])
        total  = len(all_results["manual_samples"])
        print(f"  Manual Samples : {passed}/{total} passed")

    if "overfit_analysis" in all_results:
        oa = all_results["overfit_analysis"]
        print(f"  Classifier     : {oa['classifier']['verdict']}")
        print(f"  Regressor      : {oa['regressor']['verdict']}")

    if "learning_curve" in all_results:
        lc = all_results["learning_curve"]
        print(f"  Learning Trend : {'IMPROVING' if lc['trend_improving'] else 'NOT_IMPROVING'}")

    print()
    print("  NOTE: All results are on SYNTHETIC data.")
    print("  Real-world sensor data will produce DIFFERENT (typically lower) scores.")
    print("  This test suite validates engineering correctness, not clinical validity.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Model Testing & Overfitting/Underfitting Diagnostics"
    )
    parser.add_argument(
        "--section",
        choices=["all", "samples", "overfit", "learning"],
        default="all",
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()
    main(section=args.section)

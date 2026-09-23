"""
predict.py
==========
Sample Prediction Demo
for the AI-Assisted Human Augmentation System.

PURPOSE
-------
Demonstrates the full inference pipeline on synthetic sensor inputs.
Shows how a real sensor reading would flow through the system.

USAGE
-----
    python ml/predict.py

OUTPUT
------
For each sample scenario, prints:
  - Input sensor values
  - Predicted movement state + confidence
  - Recommended assistance level
  - Category (prototype engineering bucket)
  - Top contributing features (explainability)

DISCLAIMER
----------
All inputs are synthetic demonstration values.
Not real sensor data, not medical recommendations.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from preprocess import (
    ALL_FEATURES,
    RAW_FEATURES,
    compute_single_snapshot_features,
    load_label_encoder,
    load_scaler,
)

# -- Paths ---------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent.parent / "backend" / "models"

MOVEMENT_CLASSES_LIST = [
    "REST", "WALKING", "SIT_TO_STAND", "STAND_TO_SIT", "KNEE_FLEXION", "KNEE_EXTENSION"
]

ASSISTANCE_CATEGORIES = [
    (0,  20,  "Minimal",   "Minimal physical support needed"),
    (21, 40,  "Low",       "Light support for comfort or stability"),
    (41, 60,  "Moderate",  "Meaningful support to assist movement"),
    (61, 80,  "High",      "Significant support required"),
    (81, 100, "Very High", "Maximum available support"),
]


def get_assistance_category(pct: float) -> tuple[str, str]:
    for lo, hi, name, desc in ASSISTANCE_CATEGORIES:
        if lo <= pct <= hi:
            return name, desc
    return "Very High", "Maximum available support"


def build_explanation(feat_imp: dict, feature_values: dict, top_n: int = 5) -> list[dict]:
    """Build a plain-English explanation of the top contributing features."""
    explanations = []
    for feat, imp in list(feat_imp.items())[:top_n]:
        val = feature_values.get(feat, "N/A")
        val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
        explanations.append({
            "feature"   : feat,
            "importance": imp,
            "value"     : val_str,
        })
    return explanations


def predict_single(sensor_dict: dict, verbose: bool = True) -> dict:
    """
    Run the full inference pipeline on a single sensor snapshot.

    Parameters
    ----------
    sensor_dict : dict
        Raw sensor values (keys from RAW_FEATURES).

    Returns
    -------
    dict with prediction results.
    """
    # Load artifacts
    clf    = joblib.load(MODELS_DIR / "movement_classifier.joblib")
    reg    = joblib.load(MODELS_DIR / "assistance_regressor.joblib")
    ohe    = joblib.load(MODELS_DIR / "movement_ohe.joblib")
    le     = load_label_encoder()
    scaler = load_scaler()

    import json
    with open(MODELS_DIR / "feature_importance.json") as f:
        feat_imp_all = json.load(f)

    # -- Feature engineering ---------------------------------------------------
    all_feats = compute_single_snapshot_features(sensor_dict)
    X = pd.DataFrame([all_feats])[ALL_FEATURES]
    X_sc = scaler.transform(X)

    # -- Movement prediction ---------------------------------------------------
    y_pred_enc = clf.predict(X_sc)[0]
    probas     = clf.predict_proba(X_sc)[0]
    movement   = le.inverse_transform([y_pred_enc])[0]
    confidence = float(probas.max())

    proba_dict = {
        le.classes_[i]: round(float(p), 4)
        for i, p in enumerate(probas)
    }

    # -- Assistance prediction -------------------------------------------------
    movement_ohe_vec = ohe.transform([[movement]])
    X_aug = np.hstack([X_sc, movement_ohe_vec])
    assistance_pct = float(np.clip(reg.predict(X_aug)[0], 0.0, 100.0))
    cat_name, cat_desc = get_assistance_category(assistance_pct)

    # -- Explainability --------------------------------------------------------
    # Use movement classifier importances for per-feature explanation
    feat_imp = feat_imp_all.get("movement_classifier", {})
    explanation = build_explanation(feat_imp, all_feats, top_n=5)

    result = {
        "predicted_movement"    : movement,
        "confidence"            : round(confidence, 4),
        "movement_probabilities": proba_dict,
        "recommended_assistance": round(assistance_pct, 2),
        "assistance_category"   : cat_name,
        "assistance_description": cat_desc,
        "top_features"          : explanation,
        "input_features"        : {k: round(v, 4) if isinstance(v, float) else v for k, v in all_feats.items()},
    }

    if verbose:
        print()
        print("+- PREDICTION RESULT " + "-" * 40)
        print(f"|  Predicted Movement   : {movement}")
        print(f"|  Confidence           : {confidence*100:.1f}%")
        print(f"|  Recommended Assist.  : {assistance_pct:.1f}%")
        print(f"|  Category             : {cat_name}  ({cat_desc})")
        print("|")
        print("|  [PROTOTYPE MODEL RECOMMENDATION — NOT A MEDICAL/CLINICAL DECISION]")
        print("|")
        print("|  Movement Probabilities:")
        for cls, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
            bar = "#" * int(p * 30)
            print(f"|    {cls:<20} {p*100:5.1f}%  {bar}")
        print("|")
        print("|  Top Contributing Features (Prototype Explainability):")
        for i, e in enumerate(explanation):
            print(f"|    {i+1}. {e['feature']:<35} imp={e['importance']:.4f}  val={e['value']}")
        print("+" + "-" * 60)

    return result


# -- Demo scenarios ------------------------------------------------------------

DEMO_SCENARIOS = [
    {
        "name": "Normal Walking",
        "sensor": {
            "knee_angle"               :  35.0,
            "knee_angular_velocity"    :  62.0,
            "knee_angular_acceleration":  80.0,
            "force"                    :   0.40,
            "acceleration_x"           :   1.5,
            "acceleration_y"           :   0.5,
            "acceleration_z"           :   9.5,
            "gyroscope_x"              :  30.0,
            "gyroscope_y"              :  10.0,
            "gyroscope_z"              :   5.0,
            "fatigue_indicator"        :   0.15,
        },
    },
    {
        "name": "Sit-to-Stand (High Effort)",
        "sensor": {
            "knee_angle"               :  78.0,
            "knee_angular_velocity"    : -98.0,
            "knee_angular_acceleration": -165.0,
            "force"                    :   0.88,
            "acceleration_x"           :   0.7,
            "acceleration_y"           :   2.1,
            "acceleration_z"           :   8.2,
            "gyroscope_x"              :  24.0,
            "gyroscope_y"              :  38.0,
            "gyroscope_z"              :   7.0,
            "fatigue_indicator"        :   0.75,
        },
    },
    {
        "name": "Rest / Idle",
        "sensor": {
            "knee_angle"               :   5.0,
            "knee_angular_velocity"    :   0.2,
            "knee_angular_acceleration":   0.1,
            "force"                    :   0.05,
            "acceleration_x"           :   0.1,
            "acceleration_y"           :   0.0,
            "acceleration_z"           :   9.8,
            "gyroscope_x"              :   0.5,
            "gyroscope_y"              :   0.0,
            "gyroscope_z"              :   0.0,
            "fatigue_indicator"        :   0.05,
        },
    },
    {
        "name": "Knee Flexion Exercise",
        "sensor": {
            "knee_angle"               :  58.0,
            "knee_angular_velocity"    : -72.0,
            "knee_angular_acceleration": -91.0,
            "force"                    :   0.31,
            "acceleration_x"           :   0.2,
            "acceleration_y"           :   0.8,
            "acceleration_z"           :   9.2,
            "gyroscope_x"              :  11.0,
            "gyroscope_y"              :  21.0,
            "gyroscope_z"              :   3.0,
            "fatigue_indicator"        :   0.20,
        },
    },
    {
        "name": "High Fatigue Walking",
        "sensor": {
            "knee_angle"               :  31.0,
            "knee_angular_velocity"    :  48.0,
            "knee_angular_acceleration":  68.0,
            "force"                    :   0.37,
            "acceleration_x"           :   1.2,
            "acceleration_y"           :   0.4,
            "acceleration_z"           :   9.5,
            "gyroscope_x"              :  27.0,
            "gyroscope_y"              :   9.0,
            "gyroscope_z"              :   4.0,
            "fatigue_indicator"        :   0.82,
        },
    },
]


def main():
    print("=" * 60)
    print("Sample Prediction Demo")
    print("AI-Assisted Human Augmentation System  |  SIH 2026")
    print("=" * 60)
    print()
    print("[DISCLAIMER] All input values are SYNTHETIC demonstration values.")
    print("  These are NOT real sensor readings.")
    print("  Outputs are PROTOTYPE MODEL RECOMMENDATIONS.")
    print("  NOT medical advice, diagnosis, or treatment plans.")
    print()

    results = []
    for scenario in DEMO_SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"{'=' * 60}")
        print("  Input sensor values (SIMULATED):")
        for k, v in scenario["sensor"].items():
            print(f"    {k:<35} : {v}")

        result = predict_single(scenario["sensor"], verbose=True)
        results.append({"scenario": scenario["name"], **result})

    print()
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Scenario':<28} {'Movement':<20} {'Conf%':>6} {'Assist%':>8} {'Category':<14}")
    print("  " + "-" * 78)
    for r in results:
        print(
            f"  {r['scenario']:<28} "
            f"{r['predicted_movement']:<20} "
            f"{r['confidence']*100:>5.1f}% "
            f"{r['recommended_assistance']:>7.1f}% "
            f"{r['assistance_category']:<14}"
        )
    print()
    print("All outputs above are PROTOTYPE MODEL RECOMMENDATIONS.")
    print("Not clinically validated. Not a medical device.")
    print("=" * 60)


if __name__ == "__main__":
    main()

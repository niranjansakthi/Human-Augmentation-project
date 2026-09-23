"""
assistance_engine.py
====================
Orchestrates the full sensor → prediction → recommendation pipeline.

Flow:
    raw sensor dict
        → feature engineering  (feature_processor)
        → scale features       (StandardScaler)
        → movement classify    (RandomForestClassifier)
        → one-hot movement     (OneHotEncoder)
        → assistance estimate  (RandomForestRegressor)
        → build response       (PredictionResponse)

DISCLAIMER:
    All outputs are PROTOTYPE MODEL RECOMMENDATIONS.
    Not a medical device. Not clinically validated.
    Never directly commands a physical actuator.
"""
import time
from typing import Any

import numpy as np

from ..schemas.prediction import (
    FeatureImportanceItem,
    MovementProbability,
    PredictionResponse,
)
from ..services.feature_processor import build_feature_vector
from ..services.model_loader import ModelBundle

# Assistance category definitions (prototype engineering buckets)
ASSISTANCE_CATEGORIES = [
    (0,  20,  "Minimal",   "Minimal physical support needed"),
    (21, 40,  "Low",       "Light support for comfort or stability"),
    (41, 60,  "Moderate",  "Meaningful support to assist movement"),
    (61, 80,  "High",      "Significant support required"),
    (81, 100, "Very High", "Maximum available support"),
]

# Plain-English explanation templates
EXPLANATION_TEMPLATES = {
    "REST":         "Low movement signals indicate rest state; minimal assistance is recommended by the prototype model.",
    "WALKING":      "Gait-pattern signals (knee velocity, acceleration) indicate walking; assistance is scaled with effort and fatigue indicators.",
    "SIT_TO_STAND": "High joint load and rapid angular deceleration indicate sit-to-stand transition; elevated assistance is recommended by the prototype model.",
    "STAND_TO_SIT": "Controlled loading and gradual angular acceleration indicate stand-to-sit movement; moderate assistance is recommended.",
    "KNEE_FLEXION":  "Flexion-direction velocity and joint angle indicate knee flexion exercise; assistance is set to support controlled movement.",
    "KNEE_EXTENSION":"Extension-direction velocity and angle indicate knee extension exercise; assistance is adjusted for controlled return motion.",
}


def get_assistance_category(pct: float) -> tuple[str, str]:
    for lo, hi, name, desc in ASSISTANCE_CATEGORIES:
        if lo <= pct <= hi:
            return name, desc
    return "Very High", "Maximum available support"


def build_explanation_text(movement: str, top_features: list[FeatureImportanceItem], assistance_pct: float) -> str:
    base = EXPLANATION_TEMPLATES.get(movement, "Movement pattern signals contributed to the recommendation.")
    feat_names = [f.feature.replace("_", " ") for f in top_features[:3]]
    feat_str = ", ".join(feat_names)
    return (
        f"{base} "
        f"Primary contributing features: {feat_str}. "
        f"Prototype model recommendation: {assistance_pct:.1f}% assistance."
    )


class AssistanceEngine:
    """
    Stateless inference engine.
    Accepts a ModelBundle and a raw sensor dict.
    Returns a fully populated PredictionResponse.
    """

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    def predict(self, sensor_dict: dict) -> PredictionResponse:
        """
        Run the full pipeline on a single sensor snapshot.

        Parameters
        ----------
        sensor_dict : dict
            Raw sensor values (11 keys matching RAW_FEATURES).

        Returns
        -------
        PredictionResponse
        """
        bundle = self.bundle
        if not bundle.loaded:
            raise RuntimeError("Models not loaded.")

        # 1. Feature engineering
        X = build_feature_vector(sensor_dict)

        # 2. Scale
        X_sc = bundle.scaler.transform(X)

        # 3. Movement classification
        pred_enc = bundle.classifier.predict(X_sc)[0]
        probas   = bundle.classifier.predict_proba(X_sc)[0]
        movement = bundle.label_encoder.inverse_transform([pred_enc])[0]
        confidence = float(probas.max())

        movement_probs = [
            MovementProbability(movement=cls, probability=round(float(p), 4))
            for cls, p in sorted(
                zip(bundle.label_encoder.classes_, probas),
                key=lambda x: -x[1]
            )
        ]

        # 4. Assistance estimation
        mv_ohe  = bundle.ohe.transform([[movement]])
        X_aug   = np.hstack([X_sc, mv_ohe])
        assist_pct = float(np.clip(bundle.regressor.predict(X_aug)[0], 0.0, 100.0))
        cat_name, cat_desc = get_assistance_category(assist_pct)

        # 5. Feature importance (top 5 from classifier)
        clf_importances = bundle.classifier.feature_importances_
        feature_names   = X.columns.tolist()
        feat_imp_pairs  = sorted(
            zip(feature_names, clf_importances),
            key=lambda x: -x[1]
        )[:5]
        top_features = [
            FeatureImportanceItem(
                feature=name,
                importance=round(float(imp), 4),
                value=round(float(X[name].iloc[0]), 4),
            )
            for name, imp in feat_imp_pairs
        ]

        # 6. Plain-English explanation
        explanation = build_explanation_text(movement, top_features, assist_pct)

        return PredictionResponse(
            predicted_movement=movement,
            confidence=round(confidence, 4),
            movement_probabilities=movement_probs,
            recommended_assistance=round(assist_pct, 2),
            assistance_category=cat_name,
            assistance_description=cat_desc,
            top_features=top_features,
            explanation_text=explanation,
            timestamp=time.time(),
            data_source="SIMULATED",
            disclaimer="PROTOTYPE MODEL RECOMMENDATION — NOT A MEDICAL/CLINICAL DECISION",
        )

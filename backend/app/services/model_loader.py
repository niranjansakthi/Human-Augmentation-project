"""
model_loader.py
===============
Loads and caches all trained ML model artifacts.
Provides a singleton ModelBundle so models are loaded once at startup.

All models are trained on SYNTHETIC data.
Predictions are PROTOTYPE MODEL RECOMMENDATIONS only.
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "backend" / "models"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
FEAT_IMP_FILE = MODELS_DIR / "feature_importance.json"


class ModelBundle:
    """Holds all loaded model artifacts. Loaded once at application startup."""

    def __init__(self):
        self.classifier: Optional[RandomForestClassifier] = None
        self.regressor: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.ohe: Optional[OneHotEncoder] = None
        self.metrics: dict = {}
        self.feature_importance: dict = {}
        self.loaded: bool = False
        self.error: Optional[str] = None

    def load(self) -> "ModelBundle":
        """Load all artifacts from disk. Call once at startup."""
        try:
            self.classifier    = joblib.load(MODELS_DIR / "movement_classifier.joblib")
            self.regressor     = joblib.load(MODELS_DIR / "assistance_regressor.joblib")
            self.scaler        = joblib.load(MODELS_DIR / "feature_scaler.joblib")
            self.label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
            self.ohe           = joblib.load(MODELS_DIR / "movement_ohe.joblib")

            if METRICS_FILE.exists():
                with open(METRICS_FILE) as f:
                    self.metrics = json.load(f)

            if FEAT_IMP_FILE.exists():
                with open(FEAT_IMP_FILE) as f:
                    self.feature_importance = json.load(f)

            self.loaded = True
            self.error = None
        except Exception as e:
            self.loaded = False
            self.error = str(e)
            raise RuntimeError(f"Failed to load ML models: {e}") from e

        return self

    @property
    def classes(self) -> list[str]:
        if self.label_encoder is None:
            return []
        return list(self.label_encoder.classes_)

    def info(self) -> dict:
        return {
            "loaded": self.loaded,
            "error": self.error,
            "models_dir": str(MODELS_DIR),
            "classifier": "RandomForestClassifier" if self.classifier else None,
            "regressor": "RandomForestRegressor" if self.regressor else None,
            "movement_classes": self.classes,
            "n_features": len(self.classifier.feature_importances_) if self.classifier else 0,
            "metrics_summary": {
                "movement_classifier": {
                    "test_accuracy": self.metrics.get("movement_classifier", {}).get("test_accuracy"),
                    "cv_accuracy_mean": self.metrics.get("movement_classifier", {}).get("cv_accuracy_mean"),
                },
                "assistance_regressor": {
                    "test_mae": self.metrics.get("assistance_regressor", {}).get("test_mae"),
                    "test_r2": self.metrics.get("assistance_regressor", {}).get("test_r2"),
                },
            } if self.metrics else {},
            "disclaimer": (
                "Models trained on SYNTHETIC data only. "
                "Not clinically validated. Not a medical device."
            ),
        }


# Global singleton — populated at app startup
_bundle: Optional[ModelBundle] = None


def get_model_bundle() -> ModelBundle:
    """FastAPI dependency: returns the loaded model bundle."""
    global _bundle
    if _bundle is None or not _bundle.loaded:
        raise RuntimeError("Models not loaded. Was startup() called?")
    return _bundle


def startup_load_models() -> ModelBundle:
    """Called once during FastAPI lifespan startup."""
    global _bundle
    _bundle = ModelBundle().load()
    return _bundle

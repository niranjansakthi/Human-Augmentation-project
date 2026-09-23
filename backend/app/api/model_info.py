"""
GET /api/model/info      — model metadata, version, accuracy
GET /api/model/features  — feature importance from both models
GET /api/metrics         — full saved evaluation metrics
"""
from fastapi import APIRouter, Depends
from ..services.model_loader import ModelBundle, get_model_bundle

router = APIRouter(prefix="/api", tags=["Model Info"])


@router.get("/model/info")
def model_info(bundle: ModelBundle = Depends(get_model_bundle)):
    """Returns metadata about the loaded ML models."""
    return bundle.info()


@router.get("/model/features")
def feature_importance(bundle: ModelBundle = Depends(get_model_bundle)):
    """Returns feature importance for both classifier and regressor."""
    return {
        "feature_importance": bundle.feature_importance,
        "disclaimer": (
            "Feature importances are from models trained on SYNTHETIC data. "
            "They reflect the synthetic data distribution, not real biomechanics."
        ),
    }


@router.get("/metrics")
def get_metrics(bundle: ModelBundle = Depends(get_model_bundle)):
    """Returns the full saved evaluation metrics from training."""
    return {
        "metrics": bundle.metrics,
        "disclaimer": (
            "All metrics computed on SYNTHETIC test data. "
            "Real-world performance will differ. Not clinically validated."
        ),
    }

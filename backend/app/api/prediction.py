"""
POST /api/predict — Run the full ML pipeline on a sensor payload.
This is the core ML inference endpoint.
Returns movement classification + assistance recommendation.
"""
from fastapi import APIRouter, Depends, HTTPException

from ..schemas.prediction import PredictionRequest, PredictionResponse
from ..services.assistance_engine import AssistanceEngine
from ..services.model_loader import ModelBundle, get_model_bundle

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
):
    """
    Run the complete inference pipeline:
      1. Feature engineering from raw sensor values
      2. Movement classification (RandomForestClassifier)
      3. Assistance level estimation (RandomForestRegressor)
      4. Explainability (feature importance)

    INPUT: Raw sensor values (11 fields)
    OUTPUT: Predicted movement, assistance %, explanation

    DISCLAIMER: PROTOTYPE MODEL RECOMMENDATION — NOT A MEDICAL/CLINICAL DECISION.
    All input values should be SIMULATED for this prototype.
    """
    try:
        sensor_dict = request.model_dump()
        engine = AssistanceEngine(bundle)
        return engine.predict(sensor_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

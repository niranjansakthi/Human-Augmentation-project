"""
Pydantic schemas for ML predictions and assistance recommendations.

DISCLAIMER: All predictions are prototype model outputs on simulated data.
Not a medical device. Not clinically validated.
"""
from typing import Optional
from pydantic import BaseModel, Field


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    value: Optional[float] = None


class MovementProbability(BaseModel):
    movement: str
    probability: float


class PredictionRequest(BaseModel):
    """Input to the /api/predict endpoint."""
    knee_angle: float = Field(..., ge=-10, le=135)
    knee_angular_velocity: float = Field(..., ge=-200, le=200)
    knee_angular_acceleration: float = Field(..., ge=-500, le=500)
    force: float = Field(..., ge=0.0, le=1.0)
    acceleration_x: float = Field(..., ge=-15, le=15)
    acceleration_y: float = Field(..., ge=-15, le=15)
    acceleration_z: float = Field(..., ge=-15, le=15)
    gyroscope_x: float = Field(..., ge=-180, le=180)
    gyroscope_y: float = Field(..., ge=-180, le=180)
    gyroscope_z: float = Field(..., ge=-180, le=180)
    fatigue_indicator: float = Field(..., ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "knee_angle": 35.0,
                "knee_angular_velocity": 60.0,
                "knee_angular_acceleration": 80.0,
                "force": 0.40,
                "acceleration_x": 1.5,
                "acceleration_y": 0.5,
                "acceleration_z": 9.5,
                "gyroscope_x": 30.0,
                "gyroscope_y": 10.0,
                "gyroscope_z": 5.0,
                "fatigue_indicator": 0.15,
            }
        }


class PredictionResponse(BaseModel):
    """
    Full prediction response from the assistance engine.
    All outputs are PROTOTYPE MODEL RECOMMENDATIONS, not medical decisions.
    """
    # Movement classification
    predicted_movement: str = Field(..., description="Classified movement state")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0-1)")
    movement_probabilities: list[MovementProbability] = Field(
        ..., description="Per-class probabilities"
    )

    # Assistance recommendation
    recommended_assistance: float = Field(
        ..., ge=0.0, le=100.0,
        description="Prototype recommended assistance percentage (0-100)"
    )
    assistance_category: str = Field(
        ..., description="Engineering category: Minimal/Low/Moderate/High/Very High"
    )
    assistance_description: str = Field(
        ..., description="Human-readable description of the category"
    )

    # Explainability
    top_features: list[FeatureImportanceItem] = Field(
        ..., description="Top contributing features (prototype explainability)"
    )
    explanation_text: str = Field(
        ..., description="Plain-English explanation of the recommendation"
    )

    # Metadata
    timestamp: float = Field(..., description="Unix timestamp of prediction")
    data_source: str = Field(default="SIMULATED", description="Always SIMULATED for prototype")
    disclaimer: str = Field(
        default="PROTOTYPE MODEL RECOMMENDATION — NOT A MEDICAL/CLINICAL DECISION",
        description="Safety disclaimer"
    )

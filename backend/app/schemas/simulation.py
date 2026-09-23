"""Pydantic schemas for simulation control and status."""
from typing import Optional
from pydantic import BaseModel, Field


AVAILABLE_SCENARIOS = [
    "rest",
    "normal_walking",
    "sit_to_stand",
    "stand_to_sit",
    "knee_flexion",
    "knee_extension",
]


class SimulationConfig(BaseModel):
    scenario: str = Field(
        default="normal_walking",
        description=f"Scenario name. One of: {AVAILABLE_SCENARIOS}",
    )
    frequency_hz: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Simulation update rate in Hz (0.5-10)",
    )

    class Config:
        json_schema_extra = {
            "example": {"scenario": "normal_walking", "frequency_hz": 2.0}
        }


class SimulationStatus(BaseModel):
    running: bool
    session_id: Optional[str] = None
    scenario: Optional[str] = None
    frequency_hz: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    sample_count: Optional[int] = None
    data_source: str = "SIMULATED"

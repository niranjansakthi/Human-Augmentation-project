"""Pydantic schemas for session recording and history."""
from typing import Optional
from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    session_id: str
    scenario: str
    started_at: float
    ended_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    sample_count: int = 0
    avg_assistance: Optional[float] = None
    avg_knee_angle: Optional[float] = None
    avg_force: Optional[float] = None
    avg_confidence: Optional[float] = None
    movement_distribution: dict[str, int] = {}
    data_source: str = "SIMULATED"


class SessionRecord(BaseModel):
    session_id: str
    scenario: str
    started_at: float
    ended_at: Optional[float] = None
    samples: list[dict] = []
    summary: Optional[SessionSummary] = None
    data_source: str = "SIMULATED"
    disclaimer: str = "All data is SIMULATED. Not real sensor measurements."

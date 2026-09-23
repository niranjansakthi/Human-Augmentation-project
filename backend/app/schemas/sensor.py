"""
Pydantic schemas for sensor data.
All values represent SIMULATED sensor readings — not real hardware.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """
    A single raw sensor snapshot.
    In a real system, this would come from IMU + force + angle sensors.
    Currently: SIMULATED DATA ONLY.
    """
    timestamp: float = Field(..., description="Relative time in seconds from session start")
    knee_angle: float = Field(..., ge=-10, le=135, description="Knee joint angle in degrees")
    knee_angular_velocity: float = Field(..., ge=-200, le=200, description="Knee angular velocity deg/s")
    knee_angular_acceleration: float = Field(..., ge=-500, le=500, description="Knee angular acceleration deg/s2")
    force: float = Field(..., ge=0.0, le=1.0, description="Normalised ground reaction force (0-1)")
    acceleration_x: float = Field(..., ge=-15, le=15, description="IMU X-axis acceleration m/s2")
    acceleration_y: float = Field(..., ge=-15, le=15, description="IMU Y-axis acceleration m/s2")
    acceleration_z: float = Field(..., ge=-15, le=15, description="IMU Z-axis acceleration m/s2")
    gyroscope_x: float = Field(..., ge=-180, le=180, description="IMU X-axis angular velocity deg/s")
    gyroscope_y: float = Field(..., ge=-180, le=180, description="IMU Y-axis angular velocity deg/s")
    gyroscope_z: float = Field(..., ge=-180, le=180, description="IMU Z-axis angular velocity deg/s")
    fatigue_indicator: float = Field(..., ge=0.0, le=1.0, description="Simulated fatigue proxy (0=fresh, 1=fatigued)")

    data_source: str = Field(default="SIMULATED", description="Always SIMULATED for this prototype")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1.5,
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
                "data_source": "SIMULATED",
            }
        }

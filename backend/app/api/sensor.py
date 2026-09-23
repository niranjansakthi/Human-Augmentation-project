"""
GET /api/sensor/current  — latest simulated sensor reading
GET /api/sensor/simulate — alias (same thing)
Both return SIMULATED data only.
"""
from fastapi import APIRouter, Depends
from ..simulation.simulated_provider import get_sensor_provider, SimulatedSensorProvider
from ..schemas.sensor import SensorReading

router = APIRouter(prefix="/api/sensor", tags=["Sensor"])


@router.get("/current", response_model=SensorReading)
def get_current_sensor(provider: SimulatedSensorProvider = Depends(get_sensor_provider)):
    """
    Returns the latest simulated sensor reading.
    DATA SOURCE: SIMULATED — not real hardware.
    """
    return provider.get_reading()


@router.get("/simulate", response_model=SensorReading)
def simulate_sensor(provider: SimulatedSensorProvider = Depends(get_sensor_provider)):
    """Alias for /current. Returns one simulated sensor snapshot."""
    return provider.get_reading()

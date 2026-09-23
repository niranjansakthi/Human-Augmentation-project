"""GET /health — system health check."""
import time
from fastapi import APIRouter
from ..services.model_loader import get_model_bundle

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check():
    try:
        bundle = get_model_bundle()
        ml_status = "LOADED" if bundle.loaded else "ERROR"
    except Exception:
        ml_status = "NOT_LOADED"

    return {
        "status": "ok",
        "timestamp": time.time(),
        "ml_model": ml_status,
        "sensor_source": "SIMULATED",
        "hardware": "NOT_CONNECTED",
        "disclaimer": "Software-in-the-Loop prototype. Simulated sensor data only.",
    }

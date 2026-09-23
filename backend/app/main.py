"""
main.py
=======
FastAPI Application Entry Point
AI-Assisted Human Augmentation System — SIH 2026

DISCLAIMER:
    This is a Software-in-the-Loop engineering prototype.
    All sensor data is SIMULATED.
    ML predictions are PROTOTYPE MODEL RECOMMENDATIONS only.
    Not a clinically validated medical device.

Usage:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .services.model_loader import startup_load_models
from .services.session_manager import get_session_manager

from .api import health, sensor, prediction, simulation, model_info, sessions


# ── Lifespan: load models once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("[STARTUP] Loading ML model artifacts ...")
    try:
        bundle = startup_load_models()
        print(f"[STARTUP] Models loaded. Classes: {bundle.classes}")
        print("[STARTUP] Data source: SIMULATED SENSOR DATA")
        print("[STARTUP] Disclaimer: Engineering prototype only. Not a medical device.")
    except Exception as e:
        print(f"[STARTUP ERROR] Failed to load models: {e}")
        print("[STARTUP] Some endpoints may not function correctly.")

    # Pre-warm session manager
    get_session_manager()
    print("[STARTUP] Session manager ready.")
    print("[STARTUP] Backend ready.")

    yield  # Application runs

    # SHUTDOWN
    print("[SHUTDOWN] Application shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Assisted Human Augmentation System",
    description=(
        "Software-in-the-Loop prototype for SIH 2026.\n\n"
        "**DISCLAIMER**: All sensor data is SIMULATED. "
        "ML predictions are PROTOTYPE MODEL RECOMMENDATIONS ONLY. "
        "Not a clinically validated medical device. "
        "Not for medical diagnosis, treatment, or patient care."
    ),
    version="1.0.0-prototype",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow React frontend on any local port) ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ───────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(sensor.router)
app.include_router(prediction.router)
app.include_router(simulation.router)
app.include_router(model_info.router)
app.include_router(sessions.router)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "name"       : "AI-Assisted Human Augmentation System",
        "version"    : "1.0.0-prototype",
        "status"     : "running",
        "data_source": "SIMULATED SENSOR DATA",
        "disclaimer" : (
            "Software-in-the-Loop engineering prototype. "
            "All sensor data is SIMULATED. "
            "Predictions are PROTOTYPE MODEL RECOMMENDATIONS only. "
            "Not a medical device."
        ),
        "docs"       : "/docs",
        "endpoints"  : {
            "health"          : "GET /health",
            "sensor_current"  : "GET /api/sensor/current",
            "predict"         : "POST /api/predict",
            "sim_start"       : "POST /api/simulation/start",
            "sim_stop"        : "POST /api/simulation/stop",
            "sim_status"      : "GET /api/simulation/status",
            "sim_stream"      : "GET /api/simulation/stream",
            "model_info"      : "GET /api/model/info",
            "model_features"  : "GET /api/model/features",
            "metrics"         : "GET /api/metrics",
            "sessions"        : "GET /api/sessions",
            "session_detail"  : "GET /api/sessions/{id}",
            "session_csv"     : "GET /api/sessions/{id}/csv",
        },
    }

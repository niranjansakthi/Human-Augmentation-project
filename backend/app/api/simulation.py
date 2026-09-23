"""
Simulation control endpoints.

POST /api/simulation/start  — start a simulation session
POST /api/simulation/stop   — stop and save the session
GET  /api/simulation/status — current simulation state
GET  /api/simulation/stream — Server-Sent Events stream (live data)
"""
import asyncio
import json
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from ..schemas.simulation import SimulationConfig, SimulationStatus, AVAILABLE_SCENARIOS
from ..services.assistance_engine import AssistanceEngine
from ..services.model_loader import ModelBundle, get_model_bundle
from ..services.session_manager import SessionManager, get_session_manager
from ..simulation.simulated_provider import get_sensor_provider, SimulatedSensorProvider

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])

# In-memory simulation state
_sim_running: bool = False
_sim_scenario: str = "normal_walking"
_sim_freq: float = 2.0
_sim_task: Optional[asyncio.Task] = None
_latest_payload: Optional[dict] = None


async def simulation_loop(provider: SimulatedSensorProvider, session_mgr: SessionManager, bundle: ModelBundle):
    """Background task that runs the simulation independently of API requests."""
    global _sim_running, _latest_payload
    engine = AssistanceEngine(bundle)
    
    while _sim_running:
        interval = 1.0 / _sim_freq
        try:
            reading = provider.get_reading()
            sensor_dict = reading.model_dump(exclude={"data_source"})
            prediction  = engine.predict(sensor_dict)

            # Record to session
            session_mgr.record_sample(
                sensor_dict,
                {
                    "predicted_movement"    : prediction.predicted_movement,
                    "confidence"            : prediction.confidence,
                    "recommended_assistance": prediction.recommended_assistance,
                    "assistance_category"   : prediction.assistance_category,
                },
            )

            _latest_payload = {
                "sensor"    : reading.model_dump(),
                "prediction": prediction.model_dump(),
                "timestamp" : time.time(),
            }
            await asyncio.sleep(interval)
        except Exception as e:
            print(f"[SIM ERROR] {e}")
            break


@router.post("/start", response_model=SimulationStatus)
async def start_simulation(
    config: SimulationConfig,
    provider: SimulatedSensorProvider = Depends(get_sensor_provider),
    session_mgr: SessionManager = Depends(get_session_manager),
    bundle: ModelBundle = Depends(get_model_bundle),
):
    """
    Start the simulation session.
    Creates a new session ID and begins generating sensor data in the background.
    """
    global _sim_running, _sim_scenario, _sim_freq, _sim_task, _latest_payload

    if config.scenario not in AVAILABLE_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{config.scenario}'. "
                   f"Available: {AVAILABLE_SCENARIOS}",
        )

    # Stop existing if running
    if _sim_running and _sim_task:
        _sim_running = False
        await _sim_task

    _sim_running = True
    _sim_scenario = config.scenario
    _sim_freq = config.frequency_hz
    _latest_payload = None

    provider.set_scenario(config.scenario)
    session_id = session_mgr.start(config.scenario)

    # Start background loop
    _sim_task = asyncio.create_task(simulation_loop(provider, session_mgr, bundle))

    return SimulationStatus(
        running=True,
        session_id=session_id,
        scenario=config.scenario,
        frequency_hz=config.frequency_hz,
        elapsed_seconds=0.0,
        sample_count=0,
        data_source="SIMULATED",
    )


@router.post("/stop")
async def stop_simulation(
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """Stop simulation and save session to disk."""
    global _sim_running, _sim_task
    
    if _sim_running and _sim_task:
        _sim_running = False
        await _sim_task
        
    summary = session_mgr.stop()
    if summary is None:
        return {"status": "stopped", "message": "No active session."}
        
    return {
        "status": "stopped",
        "session_id": summary.session_id,
        "duration_seconds": summary.duration_seconds,
        "sample_count": summary.sample_count,
        "avg_assistance": summary.avg_assistance,
        "movement_distribution": summary.movement_distribution,
    }


@router.get("/status", response_model=SimulationStatus)
def get_status(
    session_mgr: SessionManager = Depends(get_session_manager),
):
    """Return current simulation status."""
    elapsed = None
    if session_mgr.active_started_at:
        elapsed = round(time.time() - session_mgr.active_started_at, 2)

    return SimulationStatus(
        running=_sim_running,
        session_id=session_mgr.active_session_id,
        scenario=_sim_scenario if _sim_running else None,
        frequency_hz=_sim_freq if _sim_running else None,
        elapsed_seconds=elapsed,
        sample_count=session_mgr.active_sample_count,
        data_source="SIMULATED",
    )


@router.get("/stream")
async def stream_simulation():
    """
    Server-Sent Events stream of live sensor + prediction data.
    Frontend connects once and receives continuous updates.
    Stops when simulation is stopped.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        last_ts = 0
        while _sim_running:
            if _latest_payload and _latest_payload["timestamp"] != last_ts:
                last_ts = _latest_payload["timestamp"]
                yield f"data: {json.dumps(_latest_payload)}\n\n"
            
            # Poll slightly faster than generation frequency
            await asyncio.sleep(1.0 / (_sim_freq * 2.0))
            
        yield f"data: {json.dumps({'status': 'stopped'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

"""
GET  /api/sessions              — list all saved sessions
GET  /api/sessions/{session_id} — full session detail
GET  /api/sessions/{session_id}/csv — download session as CSV
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from ..services.session_manager import SessionManager, get_session_manager

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.get("")
def list_sessions(mgr: SessionManager = Depends(get_session_manager)):
    """List all saved simulation sessions (most recent first)."""
    return {
        "sessions": mgr.list_sessions(),
        "data_source": "SIMULATED",
        "disclaimer": "All sessions contain simulated sensor data only.",
    }


@router.get("/{session_id}")
def get_session(session_id: str, mgr: SessionManager = Depends(get_session_manager)):
    """Return full session data including all recorded samples."""
    data = mgr.get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return data


@router.get("/{session_id}/csv", response_class=PlainTextResponse)
def export_session_csv(session_id: str, mgr: SessionManager = Depends(get_session_manager)):
    """Export a session's sensor+prediction data as a CSV string."""
    csv_data = mgr.session_to_csv(session_id)
    if csv_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or has no samples.",
        )
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}.csv"},
    )

"""
session_manager.py
==================
Records, saves and lists simulation sessions.
Storage: JSON flat files in backend/sessions/ — no database needed for prototype.

Each session file: sessions/{session_id}.json
Index file:        sessions/index.json
"""
import json
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

from ..schemas.session import SessionRecord, SessionSummary

SESSIONS_DIR = Path(__file__).parent.parent.parent.parent / "backend" / "sessions"
INDEX_FILE   = SESSIONS_DIR / "index.json"


class SessionManager:
    """Manages in-memory session state and persists to JSON files."""

    def __init__(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._active: Optional[SessionRecord] = None

    # ── Active session ─────────────────────────────────────────────────────

    def start(self, scenario: str) -> str:
        """Create a new session. Returns session_id."""
        session_id = str(uuid.uuid4())[:8]
        self._active = SessionRecord(
            session_id=session_id,
            scenario=scenario,
            started_at=time.time(),
            samples=[],
        )
        return session_id

    def record_sample(self, sensor_dict: dict, prediction_dict: dict) -> None:
        """Append one timestamped sensor+prediction record to the active session."""
        if self._active is None:
            return
        self._active.samples.append({
            "t": time.time(),
            "sensor": sensor_dict,
            "prediction": prediction_dict,
        })

    def stop(self) -> Optional[SessionSummary]:
        """Stop the active session, compute summary, save to disk."""
        if self._active is None:
            return None

        self._active.ended_at = time.time()
        summary = self._compute_summary(self._active)
        self._active.summary = summary
        self._save(self._active)
        self._active = None
        return summary

    @property
    def active_session_id(self) -> Optional[str]:
        return self._active.session_id if self._active else None

    @property
    def active_sample_count(self) -> int:
        return len(self._active.samples) if self._active else 0

    @property
    def active_started_at(self) -> Optional[float]:
        return self._active.started_at if self._active else None

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self, session: SessionRecord) -> None:
        path = SESSIONS_DIR / f"{session.session_id}.json"
        with open(path, "w") as f:
            json.dump(session.model_dump(), f, indent=2, default=str)
        self._update_index(session)

    def _update_index(self, session: SessionRecord) -> None:
        index = self._load_index()
        # Update or insert summary
        entry = {
            "session_id": session.session_id,
            "scenario"  : session.scenario,
            "started_at": session.started_at,
            "ended_at"  : session.ended_at,
        }
        if session.summary:
            entry.update({
                "sample_count"  : session.summary.sample_count,
                "avg_assistance": session.summary.avg_assistance,
                "duration_seconds": session.summary.duration_seconds,
            })
        index = [e for e in index if e["session_id"] != session.session_id]
        index.append(entry)
        index.sort(key=lambda x: x.get("started_at", 0), reverse=True)
        with open(INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2, default=str)

    def _load_index(self) -> list[dict]:
        if not INDEX_FILE.exists():
            return []
        with open(INDEX_FILE) as f:
            return json.load(f)

    # ── Query ──────────────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        return self._load_index()

    def get_session(self, session_id: str) -> Optional[dict]:
        # If it's the currently active session, return the in-memory data
        if self._active and self._active.session_id == session_id:
            return self._active.model_dump()
            
        path = SESSIONS_DIR / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def session_to_csv(self, session_id: str) -> Optional[str]:
        """Return CSV string for a session's samples."""
        data = self.get_session(session_id)
        if not data or not data.get("samples"):
            return None
        rows = []
        header = None
        for s in data["samples"]:
            row = {"t": s["t"]}
            row.update(s.get("sensor", {}))
            row.update({
                "predicted_movement": s.get("prediction", {}).get("predicted_movement", ""),
                "confidence": s.get("prediction", {}).get("confidence", ""),
                "recommended_assistance": s.get("prediction", {}).get("recommended_assistance", ""),
                "assistance_category": s.get("prediction", {}).get("assistance_category", ""),
            })
            if header is None:
                header = list(row.keys())
                rows.append(",".join(header))
            rows.append(",".join(str(row.get(k, "")) for k in header))
        return "\n".join(rows)

    # ── Summary computation ────────────────────────────────────────────────

    @staticmethod
    def _compute_summary(session: SessionRecord) -> SessionSummary:
        samples = session.samples
        n = len(samples)
        duration = (session.ended_at or time.time()) - session.started_at

        assists = [s["prediction"].get("recommended_assistance", 0) for s in samples if "prediction" in s]
        angles  = [s["sensor"].get("knee_angle", 0) for s in samples if "sensor" in s]
        forces  = [s["sensor"].get("force", 0) for s in samples if "sensor" in s]
        confs   = [s["prediction"].get("confidence", 0) for s in samples if "prediction" in s]
        movements = [s["prediction"].get("predicted_movement", "") for s in samples if "prediction" in s]

        return SessionSummary(
            session_id=session.session_id,
            scenario=session.scenario,
            started_at=session.started_at,
            ended_at=session.ended_at,
            duration_seconds=round(duration, 2),
            sample_count=n,
            avg_assistance=round(sum(assists) / len(assists), 2) if assists else None,
            avg_knee_angle=round(sum(angles)  / len(angles),  2) if angles  else None,
            avg_force=round(sum(forces) / len(forces), 4) if forces else None,
            avg_confidence=round(sum(confs) / len(confs), 4) if confs else None,
            movement_distribution=dict(Counter(movements)),
        )


# Global singleton
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

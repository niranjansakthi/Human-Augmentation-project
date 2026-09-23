"""
simulated_provider.py
=====================
SimulatedSensorProvider — generates smooth, physically plausible
synthetic sensor readings that change over time.

IMPORTANT:
    All values are SYNTHETIC — not from real sensors or human subjects.
    This provider implements the same SensorProvider interface that a
    real hardware provider would implement in future.

Signal generation:
    - Primary joint angle: sinusoidal oscillation around scenario center
    - Derivatives: computed analytically from the primary signal
    - IMU: correlated with joint motion + small Gaussian noise
    - Fatigue: slow drift upward over time (scenario-dependent rate)
    - All signals: clipped to physical validity ranges
"""
import math
import time

import numpy as np

from ..schemas.sensor import SensorReading
from .sensor_provider import SensorProvider
from .scenarios import SCENARIOS, ScenarioProfile

RNG = np.random.default_rng(seed=None)   # non-deterministic for live sim


class SimulatedSensorProvider(SensorProvider):
    """
    Produces smooth sinusoidal sensor streams that mimic a wearable
    lower-limb sensor system during various movement scenarios.

    Implements SensorProvider — can be replaced by FutureRealSensorProvider
    without changing any other code.
    """

    def __init__(self, scenario_name: str = "normal_walking"):
        self._scenario_name = scenario_name
        self._profile: ScenarioProfile = SCENARIOS.get(
            scenario_name, SCENARIOS["normal_walking"]
        )
        self._start_time: float = time.time()
        self._t: float = 0.0          # elapsed time in seconds

    def reset(self) -> None:
        self._start_time = time.time()
        self._t = 0.0

    def set_scenario(self, scenario_name: str) -> None:
        self._scenario_name = scenario_name
        self._profile = SCENARIOS.get(scenario_name, SCENARIOS["normal_walking"])
        self.reset()

    @property
    def source_label(self) -> str:
        return "SIMULATED"

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    def get_reading(self) -> SensorReading:
        """Generate the next simulated sensor reading."""
        self._t = time.time() - self._start_time
        p = self._profile

        # ── Primary signal: knee angle oscillation ────────────────────────
        omega = 2 * math.pi * p.cycle_hz   # angular frequency
        phase = omega * self._t

        knee_angle = (
            p.knee_angle_center
            + p.knee_angle_amplitude * math.sin(phase)
            + self._noise(1.5)
        )

        # Derivatives (analytical + noise)
        knee_angular_velocity = (
            p.knee_angle_amplitude * omega * math.cos(phase) * p.knee_velocity_scale
            + self._noise(3.0)
        )
        knee_angular_acceleration = (
            -p.knee_angle_amplitude * omega ** 2 * math.sin(phase) * p.knee_velocity_scale
            + self._noise(8.0)
        )

        # ── Force (correlated with knee angle) ────────────────────────────
        force = (
            p.force_base
            + p.force_amplitude * abs(math.sin(phase))
            + self._noise(0.02)
        )

        # ── IMU acceleration ──────────────────────────────────────────────
        acceleration_x = (
            p.accel_x_scale * math.sin(phase)
            + self._noise(0.15)
        )
        acceleration_y = (
            0.4 * p.accel_x_scale * math.cos(phase * 0.5)
            + self._noise(0.1)
        )
        acceleration_z = (
            p.accel_z_base
            - 0.3 * abs(math.sin(phase))
            + self._noise(0.1)
        )

        # ── IMU gyroscope ─────────────────────────────────────────────────
        gyroscope_x = (
            p.gyro_x_scale * math.cos(phase)
            + self._noise(2.0)
        )
        gyroscope_y = (
            p.gyro_x_scale * 0.35 * math.sin(phase * 0.7)
            + self._noise(1.5)
        )
        gyroscope_z = (
            p.gyro_x_scale * 0.15 * math.sin(phase * 1.3)
            + self._noise(1.0)
        )

        # ── Fatigue: slowly drifts upward ─────────────────────────────────
        fatigue = min(
            1.0,
            p.fatigue_base + p.fatigue_drift * self._t + self._noise(0.01)
        )

        # ── Clip all values to physical bounds ────────────────────────────
        return SensorReading(
            timestamp=round(self._t, 3),
            knee_angle=round(float(np.clip(knee_angle, -10, 135)), 2),
            knee_angular_velocity=round(float(np.clip(knee_angular_velocity, -200, 200)), 2),
            knee_angular_acceleration=round(float(np.clip(knee_angular_acceleration, -500, 500)), 2),
            force=round(float(np.clip(force, 0.0, 1.0)), 4),
            acceleration_x=round(float(np.clip(acceleration_x, -15, 15)), 3),
            acceleration_y=round(float(np.clip(acceleration_y, -15, 15)), 3),
            acceleration_z=round(float(np.clip(acceleration_z, -15, 15)), 3),
            gyroscope_x=round(float(np.clip(gyroscope_x, -180, 180)), 2),
            gyroscope_y=round(float(np.clip(gyroscope_y, -180, 180)), 2),
            gyroscope_z=round(float(np.clip(gyroscope_z, -180, 180)), 2),
            fatigue_indicator=round(float(np.clip(fatigue, 0.0, 1.0)), 4),
            data_source="SIMULATED",
        )

    @staticmethod
    def _noise(scale: float) -> float:
        """Small Gaussian noise for realism."""
        return float(RNG.normal(0, scale * 0.15))


# ── Global provider singleton ─────────────────────────────────────────────────
_provider: SimulatedSensorProvider = SimulatedSensorProvider("normal_walking")


def get_sensor_provider() -> SimulatedSensorProvider:
    """FastAPI dependency: returns the active sensor provider."""
    return _provider

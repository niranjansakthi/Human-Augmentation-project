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
    - Sensors are generated using sinusoidal oscillations centered precisely 
      around the training distribution means (mu) with amplitudes matching 
      the training standard deviations (sigma). 
    - This ensures the simulation remains perfectly within the training distribution 
      (avoiding classification drift) while still changing smoothly over time.
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
            scenario_name, SCENARIOS.get("normal_walking")
        )
        self._start_time: float = time.time()
        self._t: float = 0.0          # elapsed time in seconds

    def reset(self) -> None:
        self._start_time = time.time()
        self._t = 0.0

    def set_scenario(self, scenario_name: str) -> None:
        self._scenario_name = scenario_name
        self._profile = SCENARIOS.get(scenario_name, SCENARIOS.get("normal_walking"))
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

        # ── Oscillation phase ─────────────────────────────────────────────
        omega = 2 * math.pi * p.cycle_hz
        phase = omega * self._t

        # Helper to generate a smooth signal bounded roughly within mu +/- 1.5*sigma
        def wave(mu: float, sigma: float, phase_offset: float = 0.0) -> float:
            return mu + sigma * 1.5 * math.sin(phase + phase_offset) + self._noise(sigma * 0.1)

        knee_angle                = wave(p.knee_angle_mu, p.knee_angle_sigma, 0.0)
        # Velocity and acceleration use phase offsets to mimic derivatives while 
        # staying strictly within their own training distribution bounds.
        knee_angular_velocity     = wave(p.knee_vel_mu, p.knee_vel_sigma, math.pi/2)
        knee_angular_acceleration = wave(p.knee_acc_mu, p.knee_acc_sigma, math.pi)
        
        force = wave(p.force_mu, p.force_sigma, 0.0)
        
        acceleration_x = wave(p.acc_x_mu, p.acc_x_sigma, 0.0)
        acceleration_y = wave(p.acc_y_mu, p.acc_y_sigma, math.pi/4)
        acceleration_z = wave(p.acc_z_mu, p.acc_z_sigma, 0.0)
        
        gyroscope_x = wave(p.gyr_x_mu, p.gyr_x_sigma, math.pi/2)
        gyroscope_y = wave(p.gyr_y_mu, p.gyr_y_sigma, math.pi/3)
        gyroscope_z = wave(p.gyr_z_mu, p.gyr_z_sigma, 0.0)

        # ── Fatigue: slowly drifts upward ─────────────────────────────────
        fatigue = min(
            1.0,
            p.fatigue_mu + p.fatigue_drift * self._t + self._noise(0.01)
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
        return float(RNG.normal(0, max(scale, 0.001)))


# ── Global provider singleton ─────────────────────────────────────────────────
_provider: SimulatedSensorProvider = SimulatedSensorProvider("normal_walking")


def get_sensor_provider() -> SimulatedSensorProvider:
    """FastAPI dependency: returns the active sensor provider."""
    return _provider

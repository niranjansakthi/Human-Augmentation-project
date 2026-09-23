"""
scenarios.py
============
Simulation scenario parameter definitions.
Each scenario defines the biomechanical profile that the simulated
sensor stream will follow.

ALL VALUES ARE SYNTHETIC — not from real human measurements.
"""
from dataclasses import dataclass


@dataclass
class ScenarioProfile:
    name: str
    display_name: str
    description: str
    # Knee dynamics
    knee_angle_center: float    # degrees — midpoint of oscillation
    knee_angle_amplitude: float # degrees — oscillation range
    knee_velocity_scale: float  # relative speed factor
    # Force
    force_base: float           # normalised 0-1
    force_amplitude: float
    # IMU
    accel_x_scale: float
    accel_z_base: float
    gyro_x_scale: float
    # Fatigue (increases over time for fatigue scenario)
    fatigue_base: float
    fatigue_drift: float        # per second increase
    # Frequency of the primary movement cycle (Hz)
    cycle_hz: float


SCENARIOS: dict[str, ScenarioProfile] = {
    "normal_walking": ScenarioProfile(
        name="normal_walking",
        display_name="Normal Walking",
        description="Steady walking gait at moderate pace with low fatigue.",
        knee_angle_center=30.0,
        knee_angle_amplitude=20.0,
        knee_velocity_scale=1.0,
        force_base=0.38,
        force_amplitude=0.12,
        accel_x_scale=1.4,
        accel_z_base=9.5,
        gyro_x_scale=28.0,
        fatigue_base=0.10,
        fatigue_drift=0.005,
        cycle_hz=1.0,
    ),
    "high_effort_walking": ScenarioProfile(
        name="high_effort_walking",
        display_name="High Effort Walking",
        description="Walking with increased load — higher force and acceleration.",
        knee_angle_center=34.0,
        knee_angle_amplitude=22.0,
        knee_velocity_scale=1.3,
        force_base=0.62,
        force_amplitude=0.18,
        accel_x_scale=2.0,
        accel_z_base=9.3,
        gyro_x_scale=38.0,
        fatigue_base=0.18,
        fatigue_drift=0.008,
        cycle_hz=1.1,
    ),
    "sit_to_stand": ScenarioProfile(
        name="sit_to_stand",
        display_name="Sit-to-Stand",
        description="Repetitive sit-to-stand transitions with high knee load.",
        knee_angle_center=50.0,
        knee_angle_amplitude=40.0,
        knee_velocity_scale=1.5,
        force_base=0.72,
        force_amplitude=0.20,
        accel_x_scale=0.6,
        accel_z_base=8.5,
        gyro_x_scale=22.0,
        fatigue_base=0.20,
        fatigue_drift=0.010,
        cycle_hz=0.4,
    ),
    "knee_exercise": ScenarioProfile(
        name="knee_exercise",
        display_name="Knee Exercise",
        description="Controlled knee flexion/extension rehabilitation exercise.",
        knee_angle_center=45.0,
        knee_angle_amplitude=25.0,
        knee_velocity_scale=0.8,
        force_base=0.28,
        force_amplitude=0.08,
        accel_x_scale=0.2,
        accel_z_base=9.2,
        gyro_x_scale=10.0,
        fatigue_base=0.15,
        fatigue_drift=0.006,
        cycle_hz=0.5,
    ),
    "fatigue_scenario": ScenarioProfile(
        name="fatigue_scenario",
        display_name="Fatigue Scenario",
        description="Walking with progressive fatigue — assistance should increase over time.",
        knee_angle_center=28.0,
        knee_angle_amplitude=16.0,
        knee_velocity_scale=0.75,
        force_base=0.34,
        force_amplitude=0.10,
        accel_x_scale=1.1,
        accel_z_base=9.5,
        gyro_x_scale=24.0,
        fatigue_base=0.40,
        fatigue_drift=0.020,   # rapid fatigue increase
        cycle_hz=0.9,
    ),
}

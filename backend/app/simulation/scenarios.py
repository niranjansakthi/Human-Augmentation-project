"""
scenarios.py
============
Simulation scenario parameter definitions.
Each scenario defines the biomechanical profile that the simulated
sensor stream will follow.

Values match the training data distributions (means and stds) so that the
simulation remains within the expected distribution, avoiding classification drift.
"""
from dataclasses import dataclass

@dataclass
class ScenarioProfile:
    name: str
    display_name: str
    description: str
    
    # Feature means (mu) and standard deviations (sigma)
    # matching the training dataset profiles for "normal" variation
    knee_angle_mu: float
    knee_angle_sigma: float
    knee_vel_mu: float
    knee_vel_sigma: float
    knee_acc_mu: float
    knee_acc_sigma: float
    force_mu: float
    force_sigma: float
    acc_x_mu: float
    acc_x_sigma: float
    acc_y_mu: float
    acc_y_sigma: float
    acc_z_mu: float
    acc_z_sigma: float
    gyr_x_mu: float
    gyr_x_sigma: float
    gyr_y_mu: float
    gyr_y_sigma: float
    gyr_z_mu: float
    gyr_z_sigma: float
    fatigue_mu: float
    fatigue_drift: float
    
    cycle_hz: float

SCENARIOS: dict[str, ScenarioProfile] = {
    "rest": ScenarioProfile(
        name="rest", display_name="Rest", description="Resting state",
        knee_angle_mu=5, knee_angle_sigma=2,
        knee_vel_mu=0, knee_vel_sigma=1,
        knee_acc_mu=0, knee_acc_sigma=1,
        force_mu=0.05, force_sigma=0.02,
        acc_x_mu=0.1, acc_x_sigma=0.05,
        acc_y_mu=0.0, acc_y_sigma=0.05,
        acc_z_mu=9.7, acc_z_sigma=0.1,
        gyr_x_mu=1, gyr_x_sigma=0.5,
        gyr_y_mu=0, gyr_y_sigma=0.5,
        gyr_z_mu=0, gyr_z_sigma=0.5,
        fatigue_mu=0.05, fatigue_drift=0.001,
        cycle_hz=0.2,
    ),
    "normal_walking": ScenarioProfile(
        name="normal_walking", display_name="Normal Walking", description="Steady walking gait",
        knee_angle_mu=35, knee_angle_sigma=10,
        knee_vel_mu=60, knee_vel_sigma=15,
        knee_acc_mu=80, knee_acc_sigma=25,
        force_mu=0.40, force_sigma=0.08,
        acc_x_mu=1.5, acc_x_sigma=0.4,
        acc_y_mu=0.5, acc_y_sigma=0.2,
        acc_z_mu=9.5, acc_z_sigma=0.5,
        gyr_x_mu=30, gyr_x_sigma=8,
        gyr_y_mu=10, gyr_y_sigma=4,
        gyr_z_mu=5, gyr_z_sigma=2,
        fatigue_mu=0.15, fatigue_drift=0.005,
        cycle_hz=1.0,
    ),
    "sit_to_stand": ScenarioProfile(
        name="sit_to_stand", display_name="Sit-to-Stand", description="Standing up from a chair",
        knee_angle_mu=75, knee_angle_sigma=15,
        knee_vel_mu=-90, knee_vel_sigma=25,
        knee_acc_mu=-150, knee_acc_sigma=40,
        force_mu=0.70, force_sigma=0.12,
        acc_x_mu=0.5, acc_x_sigma=0.2,
        acc_y_mu=1.5, acc_y_sigma=0.4,
        acc_z_mu=8.5, acc_z_sigma=0.8,
        gyr_x_mu=20, gyr_x_sigma=8,
        gyr_y_mu=30, gyr_y_sigma=10,
        gyr_z_mu=5, gyr_z_sigma=3,
        fatigue_mu=0.20, fatigue_drift=0.010,
        cycle_hz=0.5,
    ),
    "stand_to_sit": ScenarioProfile(
        name="stand_to_sit", display_name="Stand-to-Sit", description="Sitting down",
        knee_angle_mu=60, knee_angle_sigma=18,
        knee_vel_mu=80, knee_vel_sigma=22,
        knee_acc_mu=120, knee_acc_sigma=35,
        force_mu=0.55, force_sigma=0.10,
        acc_x_mu=0.4, acc_x_sigma=0.2,
        acc_y_mu=1.2, acc_y_sigma=0.4,
        acc_z_mu=8.8, acc_z_sigma=0.6,
        gyr_x_mu=18, gyr_x_sigma=7,
        gyr_y_mu=25, gyr_y_sigma=9,
        gyr_z_mu=4, gyr_z_sigma=2,
        fatigue_mu=0.18, fatigue_drift=0.010,
        cycle_hz=0.5,
    ),
    "knee_flexion": ScenarioProfile(
        name="knee_flexion", display_name="Knee Flexion", description="Flexing the knee",
        knee_angle_mu=55, knee_angle_sigma=20,
        knee_vel_mu=-70, knee_vel_sigma=18,
        knee_acc_mu=-90, knee_acc_sigma=28,
        force_mu=0.30, force_sigma=0.08,
        acc_x_mu=0.2, acc_x_sigma=0.1,
        acc_y_mu=0.8, acc_y_sigma=0.25,
        acc_z_mu=9.2, acc_z_sigma=0.4,
        gyr_x_mu=10, gyr_x_sigma=4,
        gyr_y_mu=20, gyr_y_sigma=7,
        gyr_z_mu=3, gyr_z_sigma=1,
        fatigue_mu=0.15, fatigue_drift=0.005,
        cycle_hz=0.8,
    ),
    "knee_extension": ScenarioProfile(
        name="knee_extension", display_name="Knee Extension", description="Extending the knee",
        knee_angle_mu=35, knee_angle_sigma=18,
        knee_vel_mu=65, knee_vel_sigma=18,
        knee_acc_mu=85, knee_acc_sigma=28,
        force_mu=0.28, force_sigma=0.07,
        acc_x_mu=0.2, acc_x_sigma=0.1,
        acc_y_mu=0.7, acc_y_sigma=0.22,
        acc_z_mu=9.3, acc_z_sigma=0.4,
        gyr_x_mu=9, gyr_x_sigma=4,
        gyr_y_mu=18, gyr_y_sigma=6,
        gyr_z_mu=3, gyr_z_sigma=1,
        fatigue_mu=0.14, fatigue_drift=0.005,
        cycle_hz=0.8,
    ),
}

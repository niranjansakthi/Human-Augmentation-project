"""
generate_dataset.py
====================
Synthetic / Representative Dataset Generator
for the AI-Assisted Human Augmentation System.

PURPOSE
-------
Generates a labelled tabular dataset that represents plausible sensor readings
from a lower-limb assistive / exoskeleton system.

IMPORTANT DISCLAIMER
--------------------
- All data is SYNTHETIC and generated programmatically.
- No real human subjects, patients, or measured sensor hardware were used.
- The values are chosen to be biomechanically plausible, NOT clinically validated.
- This dataset is for engineering prototype / proof-of-concept use only.
- Do NOT use this data for medical diagnosis, treatment planning, or clinical research.

REPRODUCIBILITY
---------------
A fixed random seed (RANDOM_SEED = 42) is used throughout so the dataset is
deterministic across runs.

USAGE
-----
    python ml/generate_dataset.py

OUTPUT
------
    backend/data/synthetic_dataset.csv
    backend/data/dataset_stats.json
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# -- Configuration ------------------------------------------------------------
RANDOM_SEED   = 42
SAMPLES_PER_CLASS = 500          # 500 × 6 classes = 3 000 total rows
NOISE_SCALE   = 0.05             # Gaussian noise fraction added to all signals
OUTPUT_DIR    = Path(__file__).parent.parent / "backend" / "data"
OUTPUT_CSV    = OUTPUT_DIR / "synthetic_dataset.csv"
STATS_JSON    = OUTPUT_DIR / "dataset_stats.json"

# Movement class labels
MOVEMENT_CLASSES = [
    "REST",
    "WALKING",
    "SIT_TO_STAND",
    "STAND_TO_SIT",
    "KNEE_FLEXION",
    "KNEE_EXTENSION",
]

# Variation sub-types (sampled uniformly within each class)
VARIATIONS = ["normal", "slow", "high_force", "low_stability", "high_fatigue"]

rng = np.random.default_rng(RANDOM_SEED)


# -- Biomechanical parameter tables -------------------------------------------
# Each entry defines the *mean* and *std* of each raw sensor channel for a
# specific (movement_class, variation) combination.
# All angles in degrees, velocities in deg/s, forces in N (normalised 0–1 for
# prototype), accelerations in m/s², gyroscope in deg/s.

class MovementProfile:
    """Holds the generative parameters for one (class, variation) combination."""

    def __init__(
        self,
        knee_angle_mean, knee_angle_std,
        knee_angular_velocity_mean, knee_angular_velocity_std,
        knee_angular_acceleration_mean, knee_angular_acceleration_std,
        force_mean, force_std,
        accel_x_mean, accel_x_std,
        accel_y_mean, accel_y_std,
        accel_z_mean, accel_z_std,
        gyro_x_mean, gyro_x_std,
        gyro_y_mean, gyro_y_std,
        gyro_z_mean, gyro_z_std,
        fatigue_mean, fatigue_std,
        assistance_mean, assistance_std,
    ):
        self.params = {
            "knee_angle":                  (knee_angle_mean,                  max(knee_angle_std, 0.5)),
            "knee_angular_velocity":       (knee_angular_velocity_mean,       max(knee_angular_velocity_std, 0.5)),
            "knee_angular_acceleration":   (knee_angular_acceleration_mean,   max(knee_angular_acceleration_std, 0.5)),
            "force":                       (force_mean,                       max(force_std, 0.01)),
            "acceleration_x":              (accel_x_mean,                     max(accel_x_std, 0.05)),
            "acceleration_y":              (accel_y_mean,                     max(accel_y_std, 0.05)),
            "acceleration_z":              (accel_z_mean,                     max(accel_z_std, 0.05)),
            "gyroscope_x":                 (gyro_x_mean,                      max(gyro_x_std, 0.5)),
            "gyroscope_y":                 (gyro_y_mean,                      max(gyro_y_std, 0.5)),
            "gyroscope_z":                 (gyro_z_mean,                      max(gyro_z_std, 0.5)),
            "fatigue_indicator":           (fatigue_mean,                     max(fatigue_std, 0.01)),
            "assistance_level":            (assistance_mean,                  max(assistance_std, 1.0)),
        }

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        return {
            col: np.clip(
                rng.normal(mu, sigma, n),
                *self._bounds(col),
            )
            for col, (mu, sigma) in self.params.items()
        }

    @staticmethod
    def _bounds(col: str):
        bounds = {
            "knee_angle":                  (-10, 135),
            "knee_angular_velocity":       (-200, 200),
            "knee_angular_acceleration":   (-500, 500),
            "force":                       (0.0, 1.0),
            "acceleration_x":              (-15, 15),
            "acceleration_y":              (-15, 15),
            "acceleration_z":              (-15, 15),
            "gyroscope_x":                 (-180, 180),
            "gyroscope_y":                 (-180, 180),
            "gyroscope_z":                 (-180, 180),
            "fatigue_indicator":           (0.0, 1.0),
            "assistance_level":            (0.0, 100.0),
        }
        return bounds.get(col, (-1e6, 1e6))


# -- Profile definitions -------------------------------------------------------
# Parameters chosen to represent plausible human lower-limb biomechanics.
# References: representative ranges from gait analysis literature (not validated).

PROFILES: dict[tuple, MovementProfile] = {
    # -- REST ------------------------------------------------------------------
    ("REST", "normal"):       MovementProfile( 5, 2,    0, 1,    0, 1,   0.05, 0.02,  0.1,0.05,  0.0,0.05,  9.7,0.1,   1,0.5,  0,0.5,  0,0.5,  0.05,0.02,  5, 3),
    ("REST", "slow"):         MovementProfile( 5, 2,    0, 0.5,  0, 0.5, 0.05, 0.01,  0.0,0.03,  0.0,0.03,  9.8,0.05,  0,0.3,  0,0.3,  0,0.3,  0.08,0.02,  5, 2),
    ("REST", "high_force"):   MovementProfile( 5, 3,    0, 1,    0, 1,   0.15, 0.03,  0.1,0.05,  0.1,0.05,  9.7,0.1,   1,0.5,  0,0.5,  0,0.5,  0.05,0.02,  8, 3),
    ("REST", "low_stability"):MovementProfile( 6, 4,    1, 2,    1, 2,   0.07, 0.03,  0.3,0.15,  0.2,0.10,  9.6,0.2,   3,1.5,  2,1.0,  1,0.5,  0.10,0.03,  6, 3),
    ("REST", "high_fatigue"): MovementProfile( 5, 3,    0, 1,    0, 1,   0.05, 0.02,  0.1,0.05,  0.0,0.05,  9.7,0.1,   1,0.5,  0,0.5,  0,0.5,  0.60,0.10, 12, 4),

    # -- WALKING ---------------------------------------------------------------
    ("WALKING", "normal"):       MovementProfile(35, 10,  60, 15,  80, 25,  0.40, 0.08,  1.5,0.4,  0.5,0.2,  9.5,0.5,  30, 8,  10, 4,   5, 2,  0.15,0.05, 25,  8),
    ("WALKING", "slow"):         MovementProfile(28, 8,   35, 10,  45, 15,  0.30, 0.06,  0.8,0.3,  0.3,0.15, 9.6,0.3,  18, 6,   6, 3,   3, 1,  0.20,0.05, 20,  6),
    ("WALKING", "high_force"):   MovementProfile(38, 10,  65, 15,  90, 30,  0.65, 0.10,  2.0,0.5,  0.7,0.25, 9.3,0.6,  35, 9,  12, 5,   6, 3,  0.20,0.06, 45, 10),
    ("WALKING", "low_stability"):MovementProfile(33, 14,  58, 20,  85, 35,  0.42, 0.12,  2.2,0.7,  0.8,0.35, 9.4,0.8,  38,14,  15, 7,   8, 4,  0.25,0.07, 35, 10),
    ("WALKING", "high_fatigue"): MovementProfile(32, 10,  50, 15,  70, 25,  0.38, 0.08,  1.3,0.4,  0.5,0.2,  9.5,0.5,  28, 8,   9, 4,   4, 2,  0.70,0.08, 50, 10),

    # -- SIT_TO_STAND ----------------------------------------------------------
    ("SIT_TO_STAND", "normal"):       MovementProfile(75, 15,  -90, 25,  -150, 40,  0.70, 0.12,  0.5,0.2,  1.5,0.4,  8.5,0.8,  20, 8,  30,10,   5, 3,  0.20,0.05, 55, 10),
    ("SIT_TO_STAND", "slow"):         MovementProfile(80, 12,  -55, 15,   -80, 25,  0.60, 0.10,  0.3,0.15, 0.9,0.3,  8.8,0.5,  12, 5,  18, 7,   3, 2,  0.25,0.05, 45,  8),
    ("SIT_TO_STAND", "high_force"):   MovementProfile(75, 15, -100, 28,  -170, 50,  0.90, 0.15,  0.7,0.25, 2.0,0.5,  8.2,1.0,  25,10,  38,12,   7, 4,  0.25,0.06, 72, 12),
    ("SIT_TO_STAND", "low_stability"):MovementProfile(78, 20,  -88, 30,  -145, 55,  0.72, 0.15,  0.9,0.35, 1.8,0.6,  8.3,1.2,  28,12,  35,14,   9, 5,  0.30,0.07, 62, 12),
    ("SIT_TO_STAND", "high_fatigue"): MovementProfile(75, 15,  -75, 22,  -120, 38,  0.68, 0.12,  0.5,0.2,  1.4,0.4,  8.5,0.8,  18, 8,  26, 9,   5, 3,  0.75,0.08, 78, 12),

    # -- STAND_TO_SIT ----------------------------------------------------------
    ("STAND_TO_SIT", "normal"):       MovementProfile(60, 18,   80, 22,  120, 35,  0.55, 0.10,  0.4,0.2,  1.2,0.4,  8.8,0.6,  18, 7,  25, 9,   4, 2,  0.18,0.05, 42,  8),
    ("STAND_TO_SIT", "slow"):         MovementProfile(65, 12,   45, 15,   65, 22,  0.45, 0.08,  0.2,0.12, 0.7,0.25, 9.0,0.4,  10, 4,  14, 5,   2, 1,  0.22,0.05, 32,  7),
    ("STAND_TO_SIT", "high_force"):   MovementProfile(60, 18,   90, 25,  140, 45,  0.78, 0.13,  0.6,0.22, 1.8,0.5,  8.5,0.8,  24, 9,  32,11,   6, 3,  0.22,0.06, 58, 10),
    ("STAND_TO_SIT", "low_stability"):MovementProfile(62, 22,   78, 28,  125, 45,  0.58, 0.13,  0.7,0.3,  1.5,0.5,  8.6,1.0,  26,11,  30,13,   7, 4,  0.28,0.07, 50, 10),
    ("STAND_TO_SIT", "high_fatigue"): MovementProfile(60, 18,   68, 20,   98, 32,  0.53, 0.10,  0.3,0.18, 1.0,0.35, 8.8,0.6,  15, 6,  22, 8,   4, 2,  0.72,0.08, 62, 10),

    # -- KNEE_FLEXION ----------------------------------------------------------
    ("KNEE_FLEXION", "normal"):       MovementProfile(55, 20,  -70, 18,  -90, 28,  0.30, 0.08,  0.2,0.1,  0.8,0.25, 9.2,0.4,  10, 4,  20, 7,   3, 1,  0.15,0.04, 28,  7),
    ("KNEE_FLEXION", "slow"):         MovementProfile(55, 18,  -40, 12,  -52, 18,  0.22, 0.06,  0.1,0.07, 0.5,0.18, 9.4,0.3,   6, 3,  12, 5,   2, 1,  0.18,0.04, 20,  6),
    ("KNEE_FLEXION", "high_force"):   MovementProfile(52, 20,  -78, 20, -105, 35,  0.52, 0.10,  0.3,0.13, 1.2,0.35, 9.0,0.6,  13, 5,  26, 9,   4, 2,  0.20,0.05, 45,  8),
    ("KNEE_FLEXION", "low_stability"):MovementProfile(58, 25,  -68, 25,  -92, 40,  0.32, 0.10,  0.4,0.2,  1.0,0.35, 9.1,0.7,  14, 6,  24,10,   5, 3,  0.25,0.06, 35,  8),
    ("KNEE_FLEXION", "high_fatigue"): MovementProfile(55, 20,  -58, 18,  -75, 26,  0.28, 0.07,  0.2,0.1,  0.7,0.22, 9.2,0.4,   9, 4,  17, 6,   3, 1,  0.68,0.08, 42,  8),

    # -- KNEE_EXTENSION --------------------------------------------------------
    ("KNEE_EXTENSION", "normal"):       MovementProfile(35, 18,  65, 18,   85, 28,  0.28, 0.07,  0.2,0.1,  0.7,0.22, 9.3,0.4,   9, 4,  18, 6,   3, 1,  0.14,0.04, 25,  7),
    ("KNEE_EXTENSION", "slow"):         MovementProfile(35, 15,  38, 12,   48, 18,  0.20, 0.05,  0.1,0.07, 0.4,0.15, 9.5,0.3,   5, 3,  10, 4,   2, 1,  0.16,0.04, 18,  5),
    ("KNEE_EXTENSION", "high_force"):   MovementProfile(32, 18,  72, 20,  100, 35,  0.48, 0.09,  0.3,0.13, 1.1,0.32, 9.1,0.5,  12, 5,  23, 8,   4, 2,  0.18,0.05, 42,  8),
    ("KNEE_EXTENSION", "low_stability"):MovementProfile(38, 25,  62, 25,   88, 40,  0.30, 0.10,  0.4,0.18, 0.9,0.32, 9.1,0.6,  13, 6,  22, 9,   5, 2,  0.23,0.06, 32,  8),
    ("KNEE_EXTENSION", "high_fatigue"): MovementProfile(35, 18,  52, 17,   70, 26,  0.25, 0.06,  0.2,0.1,  0.6,0.20, 9.3,0.4,   8, 4,  15, 6,   2, 1,  0.65,0.08, 38,  8),
}


# -- Generator -----------------------------------------------------------------

def generate_dataset() -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Each sample simulates a single sensor snapshot (one time-step) from the
    conceptual wearable device. The timestamp field is a relative time in seconds
    from the start of a hypothetical session.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw sensor columns + movement_state + variation + assistance_level.
    """
    all_rows: list[pd.DataFrame] = []
    samples_per_variation = SAMPLES_PER_CLASS // len(VARIATIONS)  # 100 per variation

    for movement_class in MOVEMENT_CLASSES:
        for variation in VARIATIONS:
            key = (movement_class, variation)
            profile = PROFILES[key]

            n = samples_per_variation
            data = profile.sample(n, rng)

            # Add temporal context: timestamps spaced 0.1 s apart with small jitter
            data["timestamp"] = (
                np.arange(n) * 0.1
                + rng.normal(0, 0.002, n)
            )

            # Labels
            data["movement_state"] = movement_class
            data["variation"]       = variation

            # Add small Gaussian noise on top of profile noise (sensor noise model)
            for col in [
                "knee_angle", "knee_angular_velocity", "knee_angular_acceleration",
                "force", "acceleration_x", "acceleration_y", "acceleration_z",
                "gyroscope_x", "gyroscope_y", "gyroscope_z",
            ]:
                noise = rng.normal(0, NOISE_SCALE * abs(data[col].mean() + 1e-6), n)
                data[col] = data[col] + noise

            all_rows.append(pd.DataFrame(data))

    df = pd.concat(all_rows, ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Clip numeric columns to valid physical ranges
    range_clips = {
        "knee_angle":                (-10, 135),
        "knee_angular_velocity":     (-200, 200),
        "knee_angular_acceleration": (-500, 500),
        "force":                     (0.0, 1.0),
        "acceleration_x":            (-15, 15),
        "acceleration_y":            (-15, 15),
        "acceleration_z":            (-15, 15),
        "gyroscope_x":               (-180, 180),
        "gyroscope_y":               (-180, 180),
        "gyroscope_z":               (-180, 180),
        "fatigue_indicator":         (0.0, 1.0),
        "assistance_level":          (0.0, 100.0),
    }
    for col, (lo, hi) in range_clips.items():
        df[col] = df[col].clip(lo, hi)

    # Column order
    col_order = [
        "timestamp",
        "knee_angle", "knee_angular_velocity", "knee_angular_acceleration",
        "force",
        "acceleration_x", "acceleration_y", "acceleration_z",
        "gyroscope_x", "gyroscope_y", "gyroscope_z",
        "fatigue_indicator",
        "movement_state", "variation",
        "assistance_level",
    ]
    return df[col_order]


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the dataset."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    stats = {
        "total_samples": len(df),
        "class_distribution": df["movement_state"].value_counts().to_dict(),
        "variation_distribution": df["variation"].value_counts().to_dict(),
        "feature_stats": {},
        "disclaimer": (
            "SYNTHETIC DATA ONLY. Generated programmatically with fixed seed=42. "
            "Not collected from real human subjects. Not clinically validated. "
            "For engineering prototype use only."
        ),
    }
    for col in numeric_cols:
        stats["feature_stats"][col] = {
            "mean":  round(float(df[col].mean()), 4),
            "std":   round(float(df[col].std()),  4),
            "min":   round(float(df[col].min()),  4),
            "max":   round(float(df[col].max()),  4),
            "25%":   round(float(df[col].quantile(0.25)), 4),
            "75%":   round(float(df[col].quantile(0.75)), 4),
        }
    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AI-Assisted Human Augmentation System")
    print("Synthetic Dataset Generator  |  SIH 2026")
    print("=" * 60)
    print()
    print("[DISCLAIMER] All data is SYNTHETIC. No real sensors or")
    print("             human subjects were used.")
    print()

    print("Generating dataset ...")
    df = generate_dataset()

    print(f"  Total samples  : {len(df)}")
    print(f"  Total features : {df.shape[1]}")
    print()
    print("Class distribution:")
    for cls, cnt in df["movement_state"].value_counts().items():
        print(f"  {cls:<20} {cnt}")
    print()
    print("Assistance level statistics:")
    print(df["assistance_level"].describe().round(2).to_string())
    print()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Dataset saved  : {OUTPUT_CSV}")

    stats = compute_stats(df)
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved    : {STATS_JSON}")
    print()
    print("Dataset generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
preprocess.py
=============
Feature Engineering Pipeline
for the AI-Assisted Human Augmentation System.

PURPOSE
-------
Transforms raw sensor columns into a richer feature set used by the ML models.

FEATURE TAXONOMY
----------------
RAW SENSOR FEATURES (directly from sensor columns):
  - knee_angle
  - knee_angular_velocity
  - knee_angular_acceleration
  - force
  - acceleration_x/y/z
  - gyroscope_x/y/z
  - fatigue_indicator

DERIVED FEATURES (computed from raw sensor values):
  - abs_knee_angle            : |knee_angle|
  - angular_velocity_magnitude: sqrt(gyro_x²+gyro_y²+gyro_z²)
  - accel_magnitude           : sqrt(ax²+ay²+az²)
  - knee_velocity_abs         : |knee_angular_velocity|
  - knee_accel_abs            : |knee_angular_acceleration|
  - vertical_accel_deviation  : |accel_z - 9.81| (deviation from gravity)
  - lateral_accel             : sqrt(ax²+ay²)
  - force_x_knee              : force * cos(knee_angle in radians) [proxy]
  - movement_intensity        : composite of velocity + force + accel_mag

ROLLING FEATURES (computed over a window; for streaming, window = 5 last readings):
  - knee_angle_rolling_mean
  - knee_angle_rolling_std
  - force_rolling_mean
  - force_rolling_std
  - knee_velocity_rolling_mean
  - movement_smoothness       : 1 / (1 + knee_angle_rolling_std)

NOTE: When predicting from a SINGLE snapshot (no history), rolling features
      are set to the instantaneous value (mean) and 0.0 (std).

IMPORTANT
---------
Derived features are computed from simulated sensor values. They are
engineering proxies, not validated clinical measurements.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# -- Constants -----------------------------------------------------------------
GRAVITY       = 9.81          # m/s² — standard gravity
ROLLING_WIN   = 5             # samples for rolling statistics
MODELS_DIR    = Path(__file__).parent.parent / "backend" / "models"


# -- Raw feature columns -------------------------------------------------------
RAW_FEATURES = [
    "knee_angle",
    "knee_angular_velocity",
    "knee_angular_acceleration",
    "force",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "fatigue_indicator",
]

# -- Feature name lists --------------------------------------------------------
# Used by both training and inference code to ensure consistent column ordering.

DERIVED_FEATURES = [
    "abs_knee_angle",
    "knee_velocity_abs",
    "knee_accel_abs",
    "angular_velocity_magnitude",
    "accel_magnitude",
    "vertical_accel_deviation",
    "lateral_accel",
    "force_x_knee",
    "movement_intensity",
]

ROLLING_FEATURES = [
    "knee_angle_rolling_mean",
    "knee_angle_rolling_std",
    "force_rolling_mean",
    "force_rolling_std",
    "knee_velocity_rolling_mean",
    "movement_smoothness",
]

ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES + ROLLING_FEATURES

TARGET_MOVEMENT   = "movement_state"
TARGET_ASSISTANCE = "assistance_level"


# -- Core derived-feature computation -----------------------------------------

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived (engineered) feature columns to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all RAW_FEATURES columns.

    Returns
    -------
    pd.DataFrame with DERIVED_FEATURES columns added (in-place copy).
    """
    df = df.copy()

    # Absolute knee angle — useful when model shouldn't care about flexion direction
    df["abs_knee_angle"] = df["knee_angle"].abs()

    # Absolute velocity and acceleration
    df["knee_velocity_abs"] = df["knee_angular_velocity"].abs()
    df["knee_accel_abs"]    = df["knee_angular_acceleration"].abs()

    # 3-axis gyroscope magnitude
    df["angular_velocity_magnitude"] = np.sqrt(
        df["gyroscope_x"] ** 2 + df["gyroscope_y"] ** 2 + df["gyroscope_z"] ** 2
    )

    # 3-axis accelerometer magnitude (total IMU acceleration)
    df["accel_magnitude"] = np.sqrt(
        df["acceleration_x"] ** 2 + df["acceleration_y"] ** 2 + df["acceleration_z"] ** 2
    )

    # Deviation of vertical acceleration from gravity (motion in vertical axis)
    df["vertical_accel_deviation"] = (df["acceleration_z"] - GRAVITY).abs()

    # Lateral (horizontal plane) acceleration
    df["lateral_accel"] = np.sqrt(
        df["acceleration_x"] ** 2 + df["acceleration_y"] ** 2
    )

    # Force projected along knee angle direction (engineering proxy, not clinical)
    knee_rad = np.deg2rad(df["knee_angle"])
    df["force_x_knee"] = df["force"] * np.cos(knee_rad)

    # Movement intensity composite: captures overall effort level
    df["movement_intensity"] = (
        0.35 * df["knee_velocity_abs"] / 200.0      # normalised to max range
        + 0.35 * df["force"]                         # already 0–1
        + 0.30 * df["accel_magnitude"] / (GRAVITY * np.sqrt(3))  # normalised
    ).clip(0.0, 1.0)

    return df


def compute_rolling_features(df: pd.DataFrame, window: int = ROLLING_WIN) -> pd.DataFrame:
    """
    Add rolling-window statistical features.

    For datasets where temporal ordering is meaningful (same session),
    these capture recent movement trends.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain knee_angle, force, knee_angular_velocity columns.
    window : int
        Window size in samples.

    Returns
    -------
    pd.DataFrame with ROLLING_FEATURES columns added.
    """
    df = df.copy()

    df["knee_angle_rolling_mean"] = (
        df["knee_angle"].rolling(window, min_periods=1).mean()
    )
    df["knee_angle_rolling_std"] = (
        df["knee_angle"].rolling(window, min_periods=1).std().fillna(0.0)
    )
    df["force_rolling_mean"] = (
        df["force"].rolling(window, min_periods=1).mean()
    )
    df["force_rolling_std"] = (
        df["force"].rolling(window, min_periods=1).std().fillna(0.0)
    )
    df["knee_velocity_rolling_mean"] = (
        df["knee_angular_velocity"].rolling(window, min_periods=1).mean()
    )

    # Movement smoothness: 1 / (1 + variability) — higher = smoother
    df["movement_smoothness"] = 1.0 / (1.0 + df["knee_angle_rolling_std"])

    return df


def compute_single_snapshot_features(sensor_dict: dict) -> dict:
    """
    Compute all features for a SINGLE sensor snapshot (no rolling history).

    Used by the FastAPI predict endpoint at inference time.

    Parameters
    ----------
    sensor_dict : dict
        Keys: RAW_FEATURES columns.

    Returns
    -------
    dict with ALL_FEATURES.
    """
    # Build single-row DataFrame
    df = pd.DataFrame([sensor_dict])

    # Ensure all raw columns present (fill missing with 0)
    for col in RAW_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = compute_derived_features(df)

    # Rolling features from single point = instantaneous value / 0 std
    df["knee_angle_rolling_mean"]    = df["knee_angle"]
    df["knee_angle_rolling_std"]     = 0.0
    df["force_rolling_mean"]         = df["force"]
    df["force_rolling_std"]          = 0.0
    df["knee_velocity_rolling_mean"] = df["knee_angular_velocity"]
    df["movement_smoothness"]        = 1.0   # perfectly smooth at single point

    return df[ALL_FEATURES].iloc[0].to_dict()


# -- Full pipeline -------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    rolling_window: int = ROLLING_WIN,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the complete feature engineering pipeline on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (output of generate_dataset.py).
    rolling_window : int
        Window for rolling features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with ALL_FEATURES columns.
    y_movement : pd.Series
        Movement class labels.
    y_assistance : pd.Series
        Assistance level (0–100).
    """
    df = compute_derived_features(df)
    df = compute_rolling_features(df, window=rolling_window)

    X = df[ALL_FEATURES].copy()
    y_movement   = df[TARGET_MOVEMENT].copy()
    y_assistance = df[TARGET_ASSISTANCE].copy()

    return X, y_movement, y_assistance


def fit_and_save_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data and save to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")
    print(f"  Scaler saved  : {MODELS_DIR / 'feature_scaler.joblib'}")
    return scaler


def fit_and_save_label_encoder(y: pd.Series) -> LabelEncoder:
    """Fit LabelEncoder on movement labels and save to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    le = LabelEncoder()
    le.fit(y)
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    print(f"  Label encoder : {MODELS_DIR / 'label_encoder.joblib'}")
    print(f"  Classes       : {list(le.classes_)}")
    return le


def load_scaler() -> StandardScaler:
    return joblib.load(MODELS_DIR / "feature_scaler.joblib")


def load_label_encoder() -> LabelEncoder:
    return joblib.load(MODELS_DIR / "label_encoder.joblib")


# -- Quick test ----------------------------------------------------------------
if __name__ == "__main__":
    DATA_CSV = Path(__file__).parent.parent / "backend" / "data" / "synthetic_dataset.csv"
    if not DATA_CSV.exists():
        print("Dataset not found. Run generate_dataset.py first.")
    else:
        df = pd.read_csv(DATA_CSV)
        X, y_mv, y_as = build_feature_matrix(df)
        print("Feature matrix shape:", X.shape)
        print("Features:", X.columns.tolist())
        print("\nAssistance level range:", y_as.min(), "–", y_as.max())
        print("\nMovement class counts:")
        print(y_mv.value_counts())

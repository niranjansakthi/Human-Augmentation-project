# ML Pipeline — AI-Assisted Human Augmentation System
## SIH 2026 — Software-in-the-Loop Prototype

---

## ⚠️ IMPORTANT DISCLAIMER

> **ALL data in this directory is SYNTHETIC and generated programmatically.**
> No real human subjects, patients, or physical sensors were used.
> The dataset is for engineering prototype / proof-of-concept purposes only.
> This system is NOT a clinically validated medical device.
> Do NOT use outputs for medical diagnosis, treatment planning, or patient care.

---

## Overview

This ML pipeline demonstrates how an AI-assisted lower-limb rehabilitation system
would process sensor data to:

1. **Classify movement state** (what is the user doing?)
2. **Estimate required assistance level** (how much support is needed?)

---

## Files

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Synthetic dataset generator (seed=42) |
| `preprocess.py` | Feature engineering pipeline |
| `train_movement_model.py` | Movement classifier training |
| `train_assistance_model.py` | Assistance regressor training |
| `evaluate.py` | Full evaluation report |
| `predict.py` | Sample prediction demo (5 scenarios) |

---

## Dataset

### Source
Synthetically generated using `generate_dataset.py` with `RANDOM_SEED = 42`.

### Classes
| Class | Description |
|-------|-------------|
| `REST` | User at rest / stationary |
| `WALKING` | Walking gait cycle |
| `SIT_TO_STAND` | Rising from seated position |
| `STAND_TO_SIT` | Lowering to seated position |
| `KNEE_FLEXION` | Controlled knee bending |
| `KNEE_EXTENSION` | Controlled knee straightening |

### Size
- **3,000 total samples** (500 per class)
- **5 variation sub-types** per class (100 samples each):
  - `normal` — typical movement
  - `slow` — reduced velocity
  - `high_force` — increased ground reaction force
  - `low_stability` — reduced movement smoothness
  - `high_fatigue` — elevated fatigue indicator

---

## Raw Sensor Features

| Feature | Unit | Description |
|---------|------|-------------|
| `timestamp` | seconds | Relative time from session start |
| `knee_angle` | degrees | Sagittal plane knee joint angle (0° = full extension) |
| `knee_angular_velocity` | deg/s | Angular velocity of knee joint |
| `knee_angular_acceleration` | deg/s² | Angular acceleration of knee joint |
| `force` | normalised (0–1) | Ground reaction / joint load proxy |
| `acceleration_x` | m/s² | IMU X-axis linear acceleration |
| `acceleration_y` | m/s² | IMU Y-axis linear acceleration |
| `acceleration_z` | m/s² | IMU Z-axis linear acceleration (~9.81 at rest) |
| `gyroscope_x` | deg/s | IMU X-axis angular velocity |
| `gyroscope_y` | deg/s | IMU Y-axis angular velocity |
| `gyroscope_z` | deg/s | IMU Z-axis angular velocity |
| `fatigue_indicator` | normalised (0–1) | Simulated fatigue proxy (0=fresh, 1=highly fatigued) |

---

## Derived (Engineered) Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `abs_knee_angle` | `|knee_angle|` | Direction-independent magnitude |
| `knee_velocity_abs` | `|knee_angular_velocity|` | Speed regardless of direction |
| `knee_accel_abs` | `|knee_angular_acceleration|` | Acceleration magnitude |
| `angular_velocity_magnitude` | `√(gx²+gy²+gz²)` | Total IMU rotational speed |
| `accel_magnitude` | `√(ax²+ay²+az²)` | Total IMU linear acceleration |
| `vertical_accel_deviation` | `|az - 9.81|` | Deviation from gravity = vertical motion |
| `lateral_accel` | `√(ax²+ay²)` | Horizontal plane movement |
| `force_x_knee` | `force × cos(knee_angle)` | Force component along thigh axis (proxy) |
| `movement_intensity` | Weighted composite | Overall effort/activity level (0–1) |

---

## Rolling (Window) Features

Computed over a window of 5 samples (configurable):

| Feature | Description |
|---------|-------------|
| `knee_angle_rolling_mean` | Recent average knee angle |
| `knee_angle_rolling_std` | Recent knee angle variability |
| `force_rolling_mean` | Recent average force |
| `force_rolling_std` | Recent force variability |
| `knee_velocity_rolling_mean` | Recent average angular velocity |
| `movement_smoothness` | `1 / (1 + knee_angle_rolling_std)` — smoothness index |

> For single-snapshot inference (API), rolling features are set to instantaneous values.

---

## Models

### Model A — Movement Classifier
- **Type**: `RandomForestClassifier`
- **n_estimators**: 200 | **max_depth**: 15 | **class_weight**: balanced
- **Input**: 26 engineered features (11 raw + 9 derived + 6 rolling)
- **Output**: Movement class + per-class probability (confidence)

### Model B — Assistance Regressor
- **Type**: `RandomForestRegressor`
- **n_estimators**: 200 | **max_depth**: 15
- **Input**: 26 features + 6 one-hot encoded movement class columns (32 total)
- **Output**: Continuous assistance % (0–100), bucketed into:

| Range | Category |
|-------|---------|
| 0–20% | Minimal |
| 21–40% | Low |
| 41–60% | Moderate |
| 61–80% | High |
| 81–100% | Very High |

> These buckets are **prototype engineering design decisions**, not clinical thresholds.

---

## How to Run

```bash
# 1. Generate synthetic dataset
python ml/generate_dataset.py

# 2. Train movement classifier (also fits scaler + label encoder)
python ml/train_movement_model.py

# 3. Train assistance regressor
python ml/train_assistance_model.py

# 4. Full evaluation report
python ml/evaluate.py

# 5. Sample predictions demo
python ml/predict.py
```

---

## Saved Artifacts

All saved to `backend/models/`:

| File | Contents |
|------|---------|
| `movement_classifier.joblib` | Trained RandomForestClassifier |
| `assistance_regressor.joblib` | Trained RandomForestRegressor |
| `feature_scaler.joblib` | Fitted StandardScaler |
| `label_encoder.joblib` | Fitted LabelEncoder for movement classes |
| `movement_ohe.joblib` | Fitted OneHotEncoder for movement state |
| `model_metrics.json` | All evaluation metrics |
| `feature_importance.json` | Feature importances for both models |

---

## Limitations

1. **Synthetic data**: Dataset is programmatically generated, not from real human subjects.
2. **High accuracy warning**: Accuracy on synthetic data is expected to be high. This does NOT reflect real-world performance.
3. **No temporal modelling**: A snapshot-based approach is used. Real systems might benefit from LSTM/temporal models.
4. **No domain validation**: Features are biomechanically plausible, not clinically validated.
5. **No noise model**: The noise added is simple Gaussian noise, not a realistic IMU/sensor noise model.
6. **No individual variation**: Real human movement varies significantly between individuals.

---

## Future Improvements

- Replace synthetic data with real sensor measurements
- Add LSTM/temporal model for sequence-aware predictions
- Implement domain adaptation for individual user profiles
- Add uncertainty quantification (e.g., conformal prediction)
- Validate feature engineering with biomechanics experts

---

*This pipeline is for engineering prototype / proof-of-concept use only.*
*SIH 2026 — AI-Assisted Human Augmentation System*

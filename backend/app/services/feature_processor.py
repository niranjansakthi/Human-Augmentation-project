"""
feature_processor.py
====================
Converts raw sensor values into the full 26-feature vector
expected by the ML models.

This is a thin wrapper around ml/preprocess.py so the backend
never imports ml/ directly — clean separation of concerns.
"""
import sys
from pathlib import Path

# Make ml/ importable from the backend
ML_DIR = Path(__file__).parent.parent.parent.parent / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from preprocess import (          # noqa: E402
    ALL_FEATURES,
    RAW_FEATURES,
    compute_single_snapshot_features,
)

import pandas as pd


def build_feature_vector(sensor_dict: dict) -> pd.DataFrame:
    """
    Given a raw sensor dict, return a single-row DataFrame
    with all 26 engineered features in the correct column order.

    Parameters
    ----------
    sensor_dict : dict
        Keys matching RAW_FEATURES (11 columns).

    Returns
    -------
    pd.DataFrame, shape (1, 26)
    """
    all_feats = compute_single_snapshot_features(sensor_dict)
    df = pd.DataFrame([all_feats])[ALL_FEATURES]
    return df


def get_feature_names() -> list[str]:
    """Return the ordered list of all feature names."""
    return ALL_FEATURES


def get_raw_feature_names() -> list[str]:
    """Return the raw sensor column names."""
    return RAW_FEATURES

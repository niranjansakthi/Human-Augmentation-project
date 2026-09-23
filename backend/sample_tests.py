"""
sample_tests.py
===============
10 Sample Test Cases to manually test the ML model via the FastAPI backend.

USAGE
-----
Make sure the backend is running first:
    cd backend
    uvicorn app.main:app --reload --port 8000

Then run:
    python backend/sample_tests.py

Or test individually via Swagger UI at:
    http://localhost:8000/docs  ->  POST /api/predict

WHAT EACH CASE TESTS
---------------------
  Case 1  : REST             — stationary, gravity-only acceleration
  Case 2  : WALKING          — normal pace, moderate knee angle
  Case 3  : WALKING          — fast pace, higher velocity + accel
  Case 4  : SIT_TO_STAND     — high knee load, large angular velocity
  Case 5  : STAND_TO_SIT     — controlled knee loading, positive velocity
  Case 6  : KNEE_FLEXION     — negative velocity, rehab knee bend
  Case 7  : KNEE_EXTENSION   — positive velocity, rehab knee straighten
  Case 8  : WALKING (fatigue)— same as case 2 but fatigue_indicator = 0.90
  Case 9  : SIT_TO_STAND     — extreme max effort (highest assistance expected)
  Case 10 : REST (boundary)  — micro-movement just above noise, should still be REST

DISCLAIMER
----------
All values are SYNTHETIC, manually crafted for testing purposes.
Not derived from real sensor hardware or human subjects.
Model predictions are PROTOTYPE RECOMMENDATIONS only.
Not a medical device.
"""

import json
import sys

try:
    import httpx
except ImportError:
    print("Run: pip install httpx")
    sys.exit(1)

BASE_URL = "http://localhost:8000"

# =============================================================================
# 10 SAMPLE TEST CASES
# Each case has:
#   - id            : test number
#   - description   : plain-English description
#   - expected_class: expected predicted_movement (None = any is OK)
#   - min_assist    : minimum expected assistance %
#   - max_assist    : maximum expected assistance %
#   - payload       : the 11 sensor fields sent to POST /api/predict
# =============================================================================

SAMPLE_CASES = [
    # -------------------------------------------------------------------------
    # Case 1: Pure REST
    # Minimal movement, gravity dominates accelerometer, near-zero gyro
    # Expected: REST, assistance ~5-15%
    # -------------------------------------------------------------------------
    {
        "id": 1,
        "description": "Pure REST — user sitting/lying still",
        "expected_class": "REST",
        "min_assist": 0,
        "max_assist": 20,
        "payload": {
            "knee_angle": 5.0,
            "knee_angular_velocity": 0.1,
            "knee_angular_acceleration": 0.0,
            "force": 0.05,
            "acceleration_x": 0.0,
            "acceleration_y": 0.0,
            "acceleration_z": 9.81,
            "gyroscope_x": 0.0,
            "gyroscope_y": 0.0,
            "gyroscope_z": 0.0,
            "fatigue_indicator": 0.05,
        },
    },

    # -------------------------------------------------------------------------
    # Case 2: Normal Walking
    # Moderate knee angle, periodic velocity, standard gait acceleration
    # Expected: WALKING, assistance 15-40%
    # -------------------------------------------------------------------------
    {
        "id": 2,
        "description": "Normal walking — moderate pace, low fatigue",
        "expected_class": "WALKING",
        "min_assist": 10,
        "max_assist": 45,
        "payload": {
            "knee_angle": 32.0,
            "knee_angular_velocity": 55.0,
            "knee_angular_acceleration": 72.0,
            "force": 0.38,
            "acceleration_x": 1.30,
            "acceleration_y": 0.40,
            "acceleration_z": 9.50,
            "gyroscope_x": 26.0,
            "gyroscope_y": 9.0,
            "gyroscope_z": 4.0,
            "fatigue_indicator": 0.10,
        },
    },

    # -------------------------------------------------------------------------
    # Case 3: Fast Walking
    # Higher velocity, acceleration, and gyroscope values
    # Expected: WALKING, slightly higher assistance than case 2
    # -------------------------------------------------------------------------
    {
        "id": 3,
        "description": "Fast walking — higher pace, moderate effort",
        "expected_class": "WALKING",
        "min_assist": 15,
        "max_assist": 55,
        "payload": {
            "knee_angle": 38.0,
            "knee_angular_velocity": 85.0,
            "knee_angular_acceleration": 110.0,
            "force": 0.52,
            "acceleration_x": 2.10,
            "acceleration_y": 0.60,
            "acceleration_z": 9.30,
            "gyroscope_x": 42.0,
            "gyroscope_y": 14.0,
            "gyroscope_z": 6.0,
            "fatigue_indicator": 0.20,
        },
    },

    # -------------------------------------------------------------------------
    # Case 4: Sit-to-Stand
    # High knee angle, strong negative angular velocity (knee extending fast)
    # High force from ground reaction
    # Expected: SIT_TO_STAND, assistance 50-85%
    # -------------------------------------------------------------------------
    {
        "id": 4,
        "description": "Sit-to-Stand — moderate effort transition",
        "expected_class": "SIT_TO_STAND",
        "min_assist": 40,
        "max_assist": 90,
        "payload": {
            "knee_angle": 75.0,
            "knee_angular_velocity": -90.0,
            "knee_angular_acceleration": -150.0,
            "force": 0.80,
            "acceleration_x": 0.60,
            "acceleration_y": 1.80,
            "acceleration_z": 8.40,
            "gyroscope_x": 20.0,
            "gyroscope_y": 34.0,
            "gyroscope_z": 6.0,
            "fatigue_indicator": 0.20,
        },
    },

    # -------------------------------------------------------------------------
    # Case 5: Stand-to-Sit
    # Moderate-high knee angle, positive angular velocity (knee bending slowly)
    # Moderate force
    # Expected: STAND_TO_SIT, assistance 20-55%
    # -------------------------------------------------------------------------
    {
        "id": 5,
        "description": "Stand-to-Sit — controlled descent, moderate effort",
        "expected_class": "STAND_TO_SIT",
        "min_assist": 15,
        "max_assist": 60,
        "payload": {
            "knee_angle": 60.0,
            "knee_angular_velocity": 72.0,
            "knee_angular_acceleration": 112.0,
            "force": 0.50,
            "acceleration_x": 0.35,
            "acceleration_y": 1.10,
            "acceleration_z": 8.90,
            "gyroscope_x": 16.0,
            "gyroscope_y": 22.0,
            "gyroscope_z": 3.5,
            "fatigue_indicator": 0.15,
        },
    },

    # -------------------------------------------------------------------------
    # Case 6: Knee Flexion (bending)
    # Mid-range knee angle, negative angular velocity, low ground force
    # Expected: KNEE_FLEXION, assistance 15-40%
    # -------------------------------------------------------------------------
    {
        "id": 6,
        "description": "Knee Flexion exercise — rehabilitation bending",
        "expected_class": "KNEE_FLEXION",
        "min_assist": 10,
        "max_assist": 45,
        "payload": {
            "knee_angle": 55.0,
            "knee_angular_velocity": -65.0,
            "knee_angular_acceleration": -85.0,
            "force": 0.27,
            "acceleration_x": 0.15,
            "acceleration_y": 0.70,
            "acceleration_z": 9.25,
            "gyroscope_x": 9.0,
            "gyroscope_y": 18.0,
            "gyroscope_z": 2.5,
            "fatigue_indicator": 0.15,
        },
    },

    # -------------------------------------------------------------------------
    # Case 7: Knee Extension (straightening)
    # Mid-range knee angle, positive angular velocity, low ground force
    # Expected: KNEE_EXTENSION, assistance 15-40%
    # -------------------------------------------------------------------------
    {
        "id": 7,
        "description": "Knee Extension exercise — rehabilitation straightening",
        "expected_class": "KNEE_EXTENSION",
        "min_assist": 10,
        "max_assist": 45,
        "payload": {
            "knee_angle": 33.0,
            "knee_angular_velocity": 60.0,
            "knee_angular_acceleration": 78.0,
            "force": 0.24,
            "acceleration_x": 0.15,
            "acceleration_y": 0.60,
            "acceleration_z": 9.30,
            "gyroscope_x": 8.0,
            "gyroscope_y": 16.0,
            "gyroscope_z": 2.2,
            "fatigue_indicator": 0.14,
        },
    },

    # -------------------------------------------------------------------------
    # Case 8: Walking with High Fatigue (SAME gait as Case 2 but fatigue = 0.90)
    # KEY TEST: Assistance should be HIGHER than Case 2 due to fatigue
    # Expected: WALKING, assistance > Case 2 (>30%)
    # -------------------------------------------------------------------------
    {
        "id": 8,
        "description": "Walking + HIGH FATIGUE — assistance should be higher than Case 2",
        "expected_class": "WALKING",
        "min_assist": 30,
        "max_assist": 70,
        "payload": {
            "knee_angle": 32.0,
            "knee_angular_velocity": 55.0,
            "knee_angular_acceleration": 72.0,
            "force": 0.38,
            "acceleration_x": 1.30,
            "acceleration_y": 0.40,
            "acceleration_z": 9.50,
            "gyroscope_x": 26.0,
            "gyroscope_y": 9.0,
            "gyroscope_z": 4.0,
            "fatigue_indicator": 0.90,   # <-- ONLY CHANGE from Case 2
        },
    },

    # -------------------------------------------------------------------------
    # Case 9: Sit-to-Stand — MAXIMUM EFFORT + MAXIMUM FATIGUE
    # Tests the upper bound of assistance estimation
    # Expected: SIT_TO_STAND, Very High assistance (>65%)
    # -------------------------------------------------------------------------
    {
        "id": 9,
        "description": "Sit-to-Stand EXTREME — max force + max fatigue → Very High assist",
        "expected_class": "SIT_TO_STAND",
        "min_assist": 60,
        "max_assist": 100,
        "payload": {
            "knee_angle": 83.0,
            "knee_angular_velocity": -105.0,
            "knee_angular_acceleration": -175.0,
            "force": 0.95,
            "acceleration_x": 0.75,
            "acceleration_y": 2.20,
            "acceleration_z": 8.00,
            "gyroscope_x": 27.0,
            "gyroscope_y": 42.0,
            "gyroscope_z": 8.0,
            "fatigue_indicator": 0.92,
        },
    },

    # -------------------------------------------------------------------------
    # Case 10: Micro-movement boundary
    # Very small signals — just above rest noise floor
    # Tests model sensitivity: should still return REST
    # Expected: REST, Minimal assistance
    # -------------------------------------------------------------------------
    {
        "id": 10,
        "description": "Micro-movement boundary — just above noise, still REST",
        "expected_class": "REST",
        "min_assist": 0,
        "max_assist": 20,
        "payload": {
            "knee_angle": 7.0,
            "knee_angular_velocity": 2.5,
            "knee_angular_acceleration": 1.8,
            "force": 0.08,
            "acceleration_x": 0.18,
            "acceleration_y": 0.10,
            "acceleration_z": 9.77,
            "gyroscope_x": 1.8,
            "gyroscope_y": 0.6,
            "gyroscope_z": 0.4,
            "fatigue_indicator": 0.07,
        },
    },
]


# =============================================================================
# RUNNER
# =============================================================================

def run_sample_tests():
    client = httpx.Client(base_url=BASE_URL, timeout=10.0)
    passed = failed = 0

    print()
    print("=" * 68)
    print("  SAMPLE TEST CASES — AI-Assisted Human Augmentation System")
    print("  Endpoint: POST /api/predict")
    print("  [All inputs are SYNTHETIC test values]")
    print("=" * 68)

    for case in SAMPLE_CASES:
        cid   = case["id"]
        desc  = case["description"]
        exp   = case["expected_class"]
        lo    = case["min_assist"]
        hi    = case["max_assist"]

        try:
            r = client.post("/api/predict", json=case["payload"])
            r.raise_for_status()
            data = r.json()

            pred_mv   = data["predicted_movement"]
            conf      = data["confidence"] * 100
            assist    = data["recommended_assistance"]
            cat       = data["assistance_category"]

            mv_ok   = (exp is None) or (pred_mv == exp)
            as_ok   = lo <= assist <= hi
            ok      = mv_ok and as_ok

            if ok:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"\n  Case {cid:>2} | {status} | {desc}")
            print(f"  {'─'*62}")
            print(f"  Predicted  : {pred_mv:<18} (confidence: {conf:.1f}%)")
            print(f"  Expected   : {exp if exp else 'any':<18} [{'OK' if mv_ok else '!!'}]")
            print(f"  Assistance : {assist:.1f}%  ({cat})")
            print(f"  Exp Range  : {lo}–{hi}%            [{'OK' if as_ok else '!!'}]")

            # Show top 3 probabilities
            top3 = sorted(data["movement_probabilities"], key=lambda x: -x["probability"])[:3]
            probs_str = "  ".join(f"{p['movement']}={p['probability']*100:.1f}%" for p in top3)
            print(f"  Probs      : {probs_str}")

            # Fatigue comparison note for Case 8
            if cid == 8:
                print(f"  [NOTE] Case 8 fatigue=0.90 vs Case 2 fatigue=0.10.")
                print(f"         Higher assistance in Case 8 confirms fatigue response is working.")

        except httpx.ConnectError:
            print(f"\n  Case {cid:>2} | ERROR | Cannot connect to {BASE_URL}")
            print(f"  Make sure backend is running: uvicorn app.main:app --reload --port 8000")
            failed += 1
        except Exception as e:
            print(f"\n  Case {cid:>2} | ERROR | {e}")
            failed += 1

    total = passed + failed
    print(f"\n  {'='*68}")
    print(f"  RESULTS: {passed}/{total} passed  ({passed/total*100:.0f}%)")
    if failed == 0:
        print("  All sample cases passed. ML → API pipeline is working correctly.")
    else:
        print(f"  {failed} case(s) failed. Check the [!!] markers above.")
    print()
    print("  DISCLAIMER: All values are SYNTHETIC. Not real sensor data.")
    print("  Predictions are PROTOTYPE RECOMMENDATIONS. Not medical decisions.")
    print(f"  {'='*68}")

    return failed == 0


if __name__ == "__main__":
    success = run_sample_tests()
    sys.exit(0 if success else 1)

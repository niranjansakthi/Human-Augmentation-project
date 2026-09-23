"""
test_api.py
===========
Automated API Test Suite
AI-Assisted Human Augmentation System — SIH 2026

Tests every backend endpoint to verify the ML model → FastAPI connection.

USAGE
-----
    # Make sure backend is running first:
    #   cd backend && uvicorn app.main:app --reload --port 8000

    python backend/test_api.py

WHAT IS TESTED
--------------
    1. GET /health
    2. GET /api/sensor/current
    3. GET /api/sensor/simulate
    4. POST /api/predict (walking scenario)
    5. POST /api/predict (sit-to-stand scenario)
    6. POST /api/predict (rest scenario)
    7. GET /api/model/info
    8. GET /api/model/features
    9. GET /api/metrics
    10. POST /api/simulation/start
    11. GET /api/simulation/status
    12. POST /api/simulation/stop
    13. GET /api/sessions
"""
import json
import sys
import time

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

BASE = "http://localhost:8000"
TIMEOUT = 10.0

client = httpx.Client(base_url=BASE, timeout=TIMEOUT)
passed = 0
failed = 0
results = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    icon = "[OK]" if condition else "[!!]"
    print(f"  {icon} {name}")
    if not condition and detail:
        print(f"       -> {detail}")
    results.append({"test": name, "passed": condition, "detail": detail})


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_tests():
    print()
    print("=" * 60)
    print("  API TEST SUITE — AI-Assisted Human Augmentation System")
    print("  Backend: http://localhost:8000")
    print("=" * 60)
    print("  [All sensor values used are SYNTHETIC demo values]")

    # ── Test 1: Health ────────────────────────────────────────────────
    section("1. Health Check")
    try:
        r = client.get("/health")
        data = r.json()
        check("GET /health — status 200",           r.status_code == 200)
        check("GET /health — status is 'ok'",       data.get("status") == "ok")
        check("GET /health — ml_model LOADED",       data.get("ml_model") == "LOADED")
        check("GET /health — sensor_source SIMULATED", data.get("sensor_source") == "SIMULATED")
        check("GET /health — hardware NOT_CONNECTED", data.get("hardware") == "NOT_CONNECTED")
    except Exception as e:
        check("GET /health", False, str(e))

    # ── Test 2: Sensor endpoints ──────────────────────────────────────
    section("2. Sensor Endpoints")
    for endpoint in ["/api/sensor/current", "/api/sensor/simulate"]:
        try:
            r = client.get(endpoint)
            data = r.json()
            check(f"GET {endpoint} — status 200",     r.status_code == 200)
            check(f"GET {endpoint} — has knee_angle", "knee_angle" in data)
            check(f"GET {endpoint} — data_source SIMULATED", data.get("data_source") == "SIMULATED")
            check(f"GET {endpoint} — force in [0,1]", 0.0 <= data.get("force", -1) <= 1.0)
            check(f"GET {endpoint} — fatigue in [0,1]", 0.0 <= data.get("fatigue_indicator", -1) <= 1.0)
        except Exception as e:
            check(f"GET {endpoint}", False, str(e))

    # ── Test 3: Prediction endpoint ───────────────────────────────────
    section("3. POST /api/predict — Core ML Endpoint")

    predict_cases = [
        {
            "name": "Walking scenario",
            "payload": {
                "knee_angle": 35.0, "knee_angular_velocity": 60.0,
                "knee_angular_acceleration": 80.0, "force": 0.40,
                "acceleration_x": 1.5, "acceleration_y": 0.5, "acceleration_z": 9.5,
                "gyroscope_x": 30.0, "gyroscope_y": 10.0, "gyroscope_z": 5.0,
                "fatigue_indicator": 0.15,
            },
            "expected_movement": "WALKING",
        },
        {
            "name": "Sit-to-Stand scenario",
            "payload": {
                "knee_angle": 78.0, "knee_angular_velocity": -98.0,
                "knee_angular_acceleration": -165.0, "force": 0.88,
                "acceleration_x": 0.7, "acceleration_y": 2.1, "acceleration_z": 8.2,
                "gyroscope_x": 24.0, "gyroscope_y": 38.0, "gyroscope_z": 7.0,
                "fatigue_indicator": 0.75,
            },
            "expected_movement": "SIT_TO_STAND",
        },
        {
            "name": "REST scenario",
            "payload": {
                "knee_angle": 5.0, "knee_angular_velocity": 0.2,
                "knee_angular_acceleration": 0.1, "force": 0.05,
                "acceleration_x": 0.1, "acceleration_y": 0.0, "acceleration_z": 9.8,
                "gyroscope_x": 0.5, "gyroscope_y": 0.0, "gyroscope_z": 0.0,
                "fatigue_indicator": 0.05,
            },
            "expected_movement": "REST",
        },
    ]

    for case in predict_cases:
        try:
            r = client.post("/api/predict", json=case["payload"])
            data = r.json()
            name = case["name"]
            exp_mv = case["expected_movement"]

            check(f"[{name}] status 200",          r.status_code == 200)
            check(f"[{name}] has predicted_movement", "predicted_movement" in data)
            check(f"[{name}] predicted={exp_mv}",  data.get("predicted_movement") == exp_mv)
            check(f"[{name}] confidence in [0,1]", 0.0 <= data.get("confidence", -1) <= 1.0)
            check(f"[{name}] assistance in [0,100]",
                  0.0 <= data.get("recommended_assistance", -1) <= 100.0)
            check(f"[{name}] has disclaimer",       "disclaimer" in data)
            check(f"[{name}] has top_features",     len(data.get("top_features", [])) > 0)
            check(f"[{name}] has explanation_text", bool(data.get("explanation_text")))

            print(f"       -> Predicted: {data.get('predicted_movement')} "
                  f"({data.get('confidence', 0)*100:.1f}%) "
                  f"| Assist: {data.get('recommended_assistance')}% "
                  f"({data.get('assistance_category')})")
        except Exception as e:
            check(f"POST /api/predict [{case['name']}]", False, str(e))

    # ── Test 4: Model info endpoints ──────────────────────────────────
    section("4. Model Info Endpoints")
    try:
        r = client.get("/api/model/info")
        data = r.json()
        check("GET /api/model/info — status 200",    r.status_code == 200)
        check("GET /api/model/info — loaded=True",   data.get("loaded") == True)
        check("GET /api/model/info — 6 classes",     len(data.get("movement_classes", [])) == 6)
        check("GET /api/model/info — n_features=26 or 32",
              data.get("n_features") in (26, 32))
    except Exception as e:
        check("GET /api/model/info", False, str(e))

    try:
        r = client.get("/api/model/features")
        data = r.json()
        check("GET /api/model/features — status 200",       r.status_code == 200)
        check("GET /api/model/features — has movement_classifier",
              "movement_classifier" in data.get("feature_importance", {}))
    except Exception as e:
        check("GET /api/model/features", False, str(e))

    try:
        r = client.get("/api/metrics")
        data = r.json()
        check("GET /api/metrics — status 200",        r.status_code == 200)
        check("GET /api/metrics — has metrics",       bool(data.get("metrics")))
        clf_acc = (data.get("metrics", {})
                       .get("movement_classifier", {})
                       .get("test_accuracy", 0))
        check(f"GET /api/metrics — classifier acc={clf_acc:.4f} > 0.85", clf_acc > 0.85)
    except Exception as e:
        check("GET /api/metrics", False, str(e))

    # ── Test 5: Simulation lifecycle ──────────────────────────────────
    section("5. Simulation Lifecycle")
    session_id = None
    try:
        r = client.post("/api/simulation/start",
                        json={"scenario": "normal_walking", "frequency_hz": 2.0})
        data = r.json()
        check("POST /api/simulation/start — status 200", r.status_code == 200)
        check("POST /api/simulation/start — running=True", data.get("running") == True)
        check("POST /api/simulation/start — has session_id", bool(data.get("session_id")))
        session_id = data.get("session_id")

        time.sleep(1)

        r2 = client.get("/api/simulation/status")
        d2 = r2.json()
        check("GET /api/simulation/status — running=True", d2.get("running") == True)
        check("GET /api/simulation/status — correct session", d2.get("session_id") == session_id)

        r3 = client.post("/api/simulation/stop")
        d3 = r3.json()
        check("POST /api/simulation/stop — status 200", r3.status_code == 200)
        check("POST /api/simulation/stop — stopped",    d3.get("status") == "stopped")

    except Exception as e:
        check("Simulation lifecycle", False, str(e))

    # ── Test 6: Sessions ──────────────────────────────────────────────
    section("6. Session History")
    try:
        r = client.get("/api/sessions")
        data = r.json()
        check("GET /api/sessions — status 200",    r.status_code == 200)
        check("GET /api/sessions — has sessions",  "sessions" in data)
        check("GET /api/sessions — data_source SIMULATED", data.get("data_source") == "SIMULATED")
        print(f"       -> {len(data.get('sessions', []))} session(s) found")

        if session_id:
            r2 = client.get(f"/api/sessions/{session_id}")
            check(f"GET /api/sessions/{session_id} — status 200", r2.status_code == 200)

            r3 = client.get(f"/api/sessions/{session_id}/csv")
            check(f"GET /api/sessions/{session_id}/csv — status 200 or 404",
                  r3.status_code in (200, 404))
    except Exception as e:
        check("Sessions", False, str(e))

    # ── Test 7: Invalid inputs ────────────────────────────────────────
    section("7. Validation & Error Handling")
    try:
        r = client.post("/api/predict", json={"knee_angle": 999})  # invalid
        check("POST /api/predict with invalid data — returns 422", r.status_code == 422)

        r2 = client.post("/api/simulation/start", json={"scenario": "nonexistent"})
        check("POST /api/simulation/start with bad scenario — returns 400 or 422",
              r2.status_code in (400, 422))
    except Exception as e:
        check("Validation tests", False, str(e))

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: {passed}/{passed+failed} passed")
    pct = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
    print(f"  Pass Rate : {pct:.1f}%")
    if failed == 0:
        print("  ALL TESTS PASSED. Backend is fully functional.")
    else:
        print(f"  {failed} test(s) failed. See [!!] markers above.")

    print()
    print("  [NOTE] Tests use SYNTHETIC sensor values.")
    print("  Not a clinical validation — engineering prototype verification only.")
    print("=" * 60)

    # Save results
    with open("api_test_results.json", "w") as f:
        json.dump({
            "total": passed + failed,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pct, 2),
            "results": results,
        }, f, indent=2)
    print(f"  Results saved: backend/api_test_results.json")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

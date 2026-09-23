import httpx
import time
import json
import statistics
from collections import Counter
import sys

BASE_URL = "http://localhost:8000"

def run_test(scenario, freq, duration):
    client = httpx.Client(base_url=BASE_URL, timeout=10.0)
    print(f"\n==================================================")
    print(f"TEST: scenario={scenario}, freq={freq}Hz, duration={duration}s")
    print(f"==================================================")
    
    # Stop any existing
    try:
        client.post("/api/simulation/stop")
    except:
        pass
        
    try:
        res = client.post("/api/simulation/start", json={"scenario": scenario, "frequency_hz": freq})
        res.raise_for_status()
        session_id = res.json()["session_id"]
    except Exception as e:
        print(f"Failed to start simulation: {e}")
        return
    
    time.sleep(duration)
    
    try:
        res = client.post("/api/simulation/stop")
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to stop simulation: {e}")
        return
    
    try:
        res = client.get(f"/api/sessions/{session_id}")
        res.raise_for_status()
        session_data = res.json()
    except Exception as e:
        print(f"Failed to get session data: {e}")
        return
        
    samples = session_data.get("samples", [])
    
    total = len(samples)
    print(f"Total samples recorded: {total}")
    if total == 0:
        print("No samples!")
        return
        
    predictions = [s["prediction"]["predicted_movement"] for s in samples]
    confs = [s["prediction"]["confidence"] for s in samples]
    
    dist = Counter(predictions)
    print(f"Distribution: {dist}")
    print(f"Avg confidence: {statistics.mean(confs):.4f}")
    print(f"Min confidence: {min(confs):.4f}")
    print(f"Max confidence: {max(confs):.4f}")
    
    if dist:
        dom = dist.most_common(1)[0]
        print(f"Dominant prediction: {dom[0]} ({dom[1]/total*100:.1f}%)")

if __name__ == "__main__":
    run_test("normal_walking", 1, 10)
    run_test("normal_walking", 5, 15)
    run_test("rest", 5, 10)
    run_test("sit_to_stand", 5, 10)
    run_test("stand_to_sit", 5, 10)
    run_test("knee_flexion", 5, 10)
    run_test("knee_extension", 5, 10)

import httpx
import time
import json
import statistics
from collections import Counter

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
        
    res = client.post("/api/simulation/start", json={"scenario": scenario, "frequency_hz": freq})
    session_id = res.json()["session_id"]
    
    time.sleep(duration)
    
    res = client.post("/api/simulation/stop")
    stop_data = res.json()
    
    res = client.get(f"/api/sessions/{session_id}")
    session_data = res.json()
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
    
    # Print the first 10 predictions and every 10th prediction to see the drift
    print("\nSample sequence:")
    for i, s in enumerate(samples):
        if i < 15 or i % 5 == 0:
            pred = s["prediction"]["predicted_movement"]
            conf = s["prediction"]["confidence"]
            t = s["t"] - session_data["started_at"]
            print(f"  Sample {i+1:3d} (t={t:.2f}s): {pred:<15} {conf:.4f}")

if __name__ == "__main__":
    run_test("normal_walking", 5, 10)
    run_test("sit_to_stand", 5, 10)

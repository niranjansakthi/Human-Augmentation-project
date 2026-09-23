import httpx
import time

client = httpx.Client(base_url='http://localhost:8000', timeout=10)
client.post('/api/simulation/start', json={'scenario': 'normal_walking', 'frequency_hz': 5})
time.sleep(2)
res = client.post('/api/simulation/stop').json()
session_id = res['session_id']
samples = client.get(f'/api/sessions/{session_id}').json()['samples']
for i, s in enumerate(samples[:15]):
    sensor = s['sensor']
    pred = s['prediction']
    print(f"{i+1}: ang={sensor['knee_angle']:.1f}, vel={sensor['knee_angular_velocity']:.1f}, acc={sensor['knee_angular_acceleration']:.1f} -> {pred['predicted_movement']}")

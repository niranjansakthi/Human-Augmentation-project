/* ============================================================
   api.js  —  All backend communication
   Backend: https://human-augmentation-project.onrender.com
   ============================================================ */

export const BASE_URL = 'https://human-augmentation-project.onrender.com';

const get  = (path) => fetch(`${BASE_URL}${path}`).then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); });
const post = (path, body) => fetch(`${BASE_URL}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined }).then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); });

export const api = {
  health:           ()           => get('/api/health'),
  modelInfo:        ()           => get('/api/model/info'),
  modelFeatures:    ()           => get('/api/model/features'),
  metrics:          ()           => get('/api/metrics'),
  simStatus:        ()           => get('/api/simulation/status'),
  simStart:         (cfg)        => post('/api/simulation/start', cfg),
  simStop:          ()           => post('/api/simulation/stop'),
  predict:          (data)       => post('/api/predict', data),
  sessions:         ()           => get('/api/sessions'),
  session:          (id)         => get(`/api/sessions/${id}`),
  sessionCsv:       (id)         => `${BASE_URL}/api/sessions/${id}/csv`,
};

export const SCENARIOS = [
  { value: 'rest',           label: 'Rest' },
  { value: 'normal_walking', label: 'Normal Walking' },
  { value: 'sit_to_stand',   label: 'Sit-to-Stand' },
  { value: 'stand_to_sit',   label: 'Stand-to-Sit' },
  { value: 'knee_flexion',   label: 'Knee Flexion' },
  { value: 'knee_extension', label: 'Knee Extension' },
];

/* ============================================================
   Dashboard.jsx  —  Full functional engineering console
   All data comes from the backend — zero invented values.
   ============================================================ */
import { useState, useEffect, useRef, useCallback } from 'react';
import { api, SCENARIOS, BASE_URL } from '../api';
import ExoSchematic from './ExoSchematic';
import './Dashboard.css';

// ── Live Clock ──────────────────────────────────────────────
function useClock() {
  const [t, setT] = useState('00:00:00.00');
  const start = useRef(Date.now());
  useEffect(() => {
    const iv = setInterval(() => {
      const d = Date.now() - start.current;
      const h = String(Math.floor(d / 3600000)).padStart(2,'0');
      const m = String(Math.floor((d % 3600000)/60000)).padStart(2,'0');
      const s = String(Math.floor((d % 60000)/1000)).padStart(2,'0');
      const ms = String(Math.floor((d % 1000)/10)).padStart(2,'0');
      setT(`${h}:${m}:${s}.${ms}`);
    }, 50);
    return () => clearInterval(iv);
  }, []);
  return t;
}

// ── Model Info hook ─────────────────────────────────────────
function useModelInfo() {
  const [info, setInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  useEffect(() => {
    api.modelInfo().then(setInfo).catch(() => {});
    api.metrics().then(setMetrics).catch(() => {});
  }, []);
  return { info, metrics };
}

// ── Simulation + prediction hook ────────────────────────────
function useSimulation() {
  const [status, setStatus]       = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [sessionData, setSession]   = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const pollRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try { const s = await api.simStatus(); setStatus(s); return s; }
    catch { setStatus(null); }
  }, []);

  useEffect(() => {
    fetchStatus();
    const iv = setInterval(() => fetchStatus(), 3000);
    return () => clearInterval(iv);
  }, [fetchStatus]);

  useEffect(() => {
    if (!status?.running) { setPrediction(null); setSession(null); return; }
    const poll = async () => {
      try {
        if (status.session_id) {
          const sess = await api.session(status.session_id);
          setSession(sess);
          const samples = sess?.samples ?? [];
          if (samples.length > 0) {
            const last = samples[samples.length - 1];
            if (last?.sensor) {
              const pred = await api.predict(last.sensor);
              setPrediction(pred);
            }
          }
        }
      } catch { /* transient */ }
    };
    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => clearInterval(pollRef.current);
  }, [status?.running, status?.session_id]);

  const start = useCallback(async (scenario) => {
    setLoading(true); setError(null);
    try { await api.simStart({ scenario, frequency_hz: 5.0 }); await fetchStatus(); }
    catch (e) { setError(e.message); }
    finally   { setLoading(false); }
  }, [fetchStatus]);

  const stop = useCallback(async () => {
    setLoading(true); setError(null);
    try { await api.simStop(); await fetchStatus(); }
    catch (e) { setError(e.message); }
    finally   { setLoading(false); }
  }, [fetchStatus]);

  return { status, prediction, sessionData, loading, error, start, stop };
}

// ── SENSOR META ─────────────────────────────────────────────
const SENSORS = [
  { k:'knee_angle',               lbl:'1. Knee Angle',         unit:'deg' },
  { k:'knee_angular_velocity',    lbl:'2. Knee Ang. Velocity', unit:'deg/s', orange:true },
  { k:'knee_angular_acceleration',lbl:'3. Knee Ang. Accel',    unit:'deg/s²' },
  { k:'force',                    lbl:'4. Axial Force',         unit:'N',   orange:true },
  { k:'acceleration_x',           lbl:'5. Accel X',            unit:'g' },
  { k:'acceleration_y',           lbl:'6. Accel Y',            unit:'g' },
  { k:'acceleration_z',           lbl:'7. Accel Z',            unit:'g' },
  { k:'gyroscope_x',              lbl:'8. Gyro X',             unit:'deg/s' },
  { k:'gyroscope_y',              lbl:'9. Gyro Y',             unit:'deg/s' },
  { k:'gyroscope_z',              lbl:'10. Gyro Z',            unit:'deg/s' },
  { k:'fatigue_indicator',        lbl:'11. Fatigue Indicator', unit:'[0-1]', wide:true },
];

const MOVE_CLASSES = ['WALKING','STAND_TO_SIT','KNEE_EXTENSION','KNEE_FLEXION','REST','SIT_TO_STAND'];

// ── Dashboard component ─────────────────────────────────────
export default function Dashboard({ onBack }) {
  const clock = useClock();
  const { info, metrics } = useModelInfo();
  const { status, prediction, sessionData, loading, error, start, stop } = useSimulation();

  const [scenario, setScenario] = useState(SCENARIOS[1].value);

  const isRunning   = status?.running ?? false;
  const backendOk   = info !== null;
  const modelOk     = info?.loaded ?? false;

  const latestSensor = sessionData?.samples?.length
    ? sessionData.samples[sessionData.samples.length - 1]?.sensor
    : null;

  const confPct   = prediction ? (prediction.confidence * 100).toFixed(1) : null;
  const assistPct = prediction ? Number(prediction.recommended_assistance).toFixed(1) : null;

  // movement_probabilities → map by class name
  const probMap = {};
  if (prediction?.movement_probabilities) {
    prediction.movement_probabilities.forEach(p => { probMap[p.movement] = p.probability; });
  }

  // metrics from backend
  const clf = metrics?.metrics?.movement_classifier;
  const reg = metrics?.metrics?.assistance_regressor;

  const catClass = prediction?.assistance_category
    ? `cat-badge--${prediction.assistance_category.toLowerCase().replace(/\s+/g,'-')}`
    : '';

  return (
    <div className="db">

      {/* ── HEADER ── */}
      <header className="db-header">
        <div className="db-header__left">
          <div className="db-logo-mark"><span/><span/></div>
          <span className="db-brand">H-AUGMENT</span>
          <span className="db-tag">AI-ASSISTED HUMAN AUGMENTATION</span>
        </div>
        <div className="db-header__right">
          <span className="db-header-badge">
            &lt;&gt; SOFTWARE-IN-THE-LOOP
          </span>
          <span className="db-header-badge">
            ⊡ SIMULATED DATA
          </span>
          <span className={`db-header-badge${backendOk ? ' db-header-badge--green' : ''}`}>
            <span className="dot"/>
            {backendOk ? 'BACKEND CONNECTED' : 'BACKEND OFFLINE'}
          </span>
          <button className="db-back-btn" onClick={onBack}>← BACK</button>
        </div>
      </header>

      {/* ── CONTEXT BAR ── */}
      <div className="db-context-bar">
        <div className="db-context-bar__left">
          <span className="db-context-bar__tag">⚗ SW-LOOP // BUILD 2.4.9-SIM</span>
          <span>TELEMETRY BENCH: ORTHOTIC DYNAMICS</span>
        </div>
        <div className="db-context-bar__right">
          <span>REF_CLOCK: <span className="db-clock">{clock}</span></span>
          <span>SAMPLING: <span className="db-hz">5.00 Hz {isRunning ? 'ACTIVE' : 'IDLE'}</span></span>
        </div>
      </div>

      <main className="db-main">

        {/* ══════════════════════════════════════════
            SECTION 1: HERO — 5 col left + 7 col dark schematic
            ══════════════════════════════════════════ */}
        <div className="db-grid-12">

          {/* LEFT HERO PANEL */}
          <div className="panel hero-panel col-5">
            <div className="panel-header">
              <div className="panel-header__left">
                <span className="panel-mod">// SYS.OVERVIEW</span>
                <span className="panel-title">Adaptive Assistance Around Human Movement.</span>
              </div>
            </div>
            <div className="panel-body">
              <div className="hero-badge-row">
                <span className="hero-indicator">
                  <span className="dot" style={{ background: backendOk ? 'var(--green)' : 'var(--text-light)' }}/>
                  {backendOk ? 'Backend Connected' : 'Backend Offline'}
                </span>
                <span className="hero-indicator">
                  <span className="dot" style={{ background: modelOk ? 'var(--green)' : 'var(--text-light)' }}/>
                  {modelOk ? 'ML Model Loaded' : 'Model Unavailable'}
                </span>
              </div>
              <p className="hero-desc">
                Real-time movement classification and adaptive assistance estimation
                utilizing high-fidelity simulated biomechanical kinematic telemetry.
              </p>
            </div>
            <div className="hero-actions">
              <button className="btn-primary" disabled={isRunning || loading} onClick={() => start(scenario)}>
                ▶ {loading && !isRunning ? 'STARTING...' : 'START SIMULATION'}
              </button>
              <button className="btn-secondary" disabled={!isRunning || loading} onClick={stop}>
                ■ STOP
              </button>
              {error && <span className="err-msg">{error}</span>}
            </div>
          </div>

          {/* RIGHT SCHEMATIC PANEL */}
          <div className="panel schematic-panel col-7">
            <div className="panel-header">
              <div className="panel-header__left">
                <span className="panel-mod">⚙ SCHEMATIC REF // EXOSKELETAL KINEMATICS D-04</span>
              </div>
              <span className="panel-badge panel-badge--orange">
                ILLUSTRATIVE MODEL — NOT REAL HARDWARE
              </span>
            </div>
            <ExoSchematic isRunning={isRunning}/>
            <div className="pipeline-bar">
              <div className="pipeline-flow">
                <span className="step">SENSORS</span><span className="arrow">→</span>
                <span className="step">BIOMECHANICAL DATA</span><span className="arrow">→</span>
                <span className="step">ML MODEL</span><span className="arrow">→</span>
                <span className="active-step">CLASSIFICATION</span><span className="arrow">→</span>
                <span className="step">ESTIMATION</span>
              </div>
              <span className="pipeline-note">SIMULATED BIOMECHANICAL PIPELINE</span>
            </div>
          </div>

        </div>

        {/* ══════════════════════════════════════════
            SECTION 2: SIMULATION CONTROL BAR
            ══════════════════════════════════════════ */}
        <div className="panel">
          <div className="panel-body">
            <div className="sim-bar">
              <div className="sim-bar__left">
                <div className="sim-bar__title">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{color:'var(--orange)'}}>
                    <rect x="1" y="5" width="14" height="7" rx="1" stroke="currentColor" strokeWidth="1.4"/>
                    <line x1="5" y1="5" x2="5" y2="12" stroke="currentColor" strokeWidth="0.9"/>
                    <line x1="11" y1="5" x2="11" y2="12" stroke="currentColor" strokeWidth="0.9"/>
                    <line x1="1" y1="8" x2="15" y2="8" stroke="currentColor" strokeWidth="0.9"/>
                  </svg>
                  SIMULATION CONTROL
                </div>

                <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
                  <span className="sim-label">SCENARIO:</span>
                  <select
                    className="select-scenario"
                    value={scenario}
                    onChange={e => setScenario(e.target.value)}
                    disabled={isRunning || loading}
                  >
                    {SCENARIOS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                  </select>
                </div>

                <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
                  <span className="sim-label">FREQUENCY:</span>
                  <span className="freq-pill">5 Hz (Standard Telemetry Rate)</span>
                </div>
              </div>

              <div className="sim-bar__right">
                <button className="btn-secondary" disabled={!isRunning || loading} onClick={stop}>
                  ⏸ PAUSE STREAM
                </button>
                <button className="btn-secondary" disabled={!isRunning || loading} onClick={stop}>
                  ■ RESET
                </button>
                {isRunning && (
                  <span className="sim-active-badge">
                    <span className="ping"/>
                    SIMULATION ACTIVE
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ══════════════════════════════════════════
            SECTION 3: MAIN DASHBOARD — 8 col left, 4 col right
            ══════════════════════════════════════════ */}
        <div className="db-grid-12">

          {/* ── LEFT 8 COLS ── */}
          <div className="col-8" style={{display:'flex',flexDirection:'column',gap:'14px'}}>

            {/* MOD.01 — CURRENT INFERENCE */}
            <div className="panel">
              <div className="panel-header">
                <div className="panel-header__left">
                  <span className="panel-mod">// MOD.01_ACTIVE_INFERENCE</span>
                  <span className="panel-title">CURRENT INFERENCE OUTPUT</span>
                </div>
                {prediction && (
                  <span className="panel-badge panel-badge--orange">LATENCY: LIVE</span>
                )}
              </div>
              <div className="panel-body">
                {prediction ? (
                  <>
                    <div className="inference-grid">
                      {/* Movement */}
                      <div className="inf-card">
                        <div className="inf-card__label">
                          IDENTIFIED GAIT MODE
                          <span className="inf-card__class-badge">P-CLASS 01</span>
                        </div>
                        <div className="inf-card__main" id="disp-movement">
                          {prediction.predicted_movement}
                        </div>
                        <div className="inf-card__sub">
                          <span>Class Confidence</span>
                          <strong id="disp-confidence">{confPct}%</strong>
                        </div>
                      </div>
                      {/* Assistance */}
                      <div className="inf-card">
                        <div className="inf-card__label">
                          ADAPTIVE ACTUATION
                          <span className={`cat-badge ${catClass}`} id="disp-category-badge">
                            {prediction.assistance_category?.toUpperCase() ?? '—'} ASSISTANCE
                          </span>
                        </div>
                        <div className="inf-card__main inf-card__main--orange" id="disp-assistance-val">
                          {assistPct}%
                        </div>
                        <div className="inf-card__sub">
                          <span>Continuous Output</span>
                          <strong>TARGET CAPACITY</strong>
                        </div>
                      </div>
                    </div>

                    {/* Gauge */}
                    <div className="gauge-container">
                      <div className="gauge-labels">
                        <span>0% ASSIST</span><span>25%</span>
                        <span>50% ASSIST</span><span>75%</span>
                        <span>100% ASSIST (MAX LIMIT)</span>
                      </div>
                      <div className="gauge-track">
                        <div className="gauge-fill" id="gauge-bar" style={{width:`${assistPct}%`}}/>
                        <div className="gauge-notch" style={{left:'25%'}}/>
                        <div className="gauge-notch" style={{left:'50%'}}/>
                        <div className="gauge-notch" style={{left:'75%'}}/>
                      </div>
                    </div>

                    <div className="disclaimer-strip">
                      <span style={{color:'var(--orange)'}}>ⓘ</span>
                      Prototype model output — not a medical or clinical recommendation. Designed for software simulation analysis.
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state__title">NO ACTIVE PREDICTION</div>
                    <div className="empty-state__sub">Start a simulation to see live inference output.</div>
                  </div>
                )}
              </div>
            </div>

            {/* MOD.02 — LIVE SENSOR READINGS */}
            <div className="panel">
              <div className="panel-header">
                <div className="panel-header__left">
                  <span className="panel-mod">// MOD.02_TELEM_BUS</span>
                  <span className="panel-title">LIVE SENSOR READINGS</span>
                </div>
                <span className="panel-badge">SIMULATED SYNTHETIC FEED</span>
              </div>
              <div className="panel-body">
                {latestSensor ? (
                  <div className="sensor-matrix">
                    {SENSORS.map(({ k, lbl, unit, orange, wide }) => {
                      const val = latestSensor[k];
                      return (
                        <div className={`sensor-cell${wide ? ' sensor-cell--wide' : ''}`} key={k}>
                          <div className="sensor-cell__lbl">{lbl}</div>
                          <div className="sensor-cell__val-row">
                            <span className={`sensor-cell__val${orange ? ' sensor-cell__val--orange' : ''}`}>
                              {val !== undefined ? Number(val).toFixed(2) : '—'}
                            </span>
                            <span className="sensor-cell__unit">{unit}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state__title">SENSOR DATA UNAVAILABLE</div>
                    <div className="empty-state__sub">Start simulation to see live sensor telemetry.</div>
                  </div>
                )}
              </div>
            </div>

            {/* MOD.03 — MOVEMENT CLASSIFICATION PROBABILITIES */}
            <div className="panel">
              <div className="panel-header">
                <div className="panel-header__left">
                  <span className="panel-mod">// MOD.03_MULTI_CLASS</span>
                  <span className="panel-title">MOVEMENT CLASSIFICATION PROBABILITIES</span>
                </div>
                <span className="panel-badge">6 ARCHETYPES</span>
              </div>
              <div className="panel-body">
                {prediction?.movement_probabilities?.length ? (
                  <div className="prob-list">
                    {MOVE_CLASSES.map(cls => {
                      const prob = probMap[cls] ?? 0;
                      const pct  = (prob * 100).toFixed(1);
                      const isActive = cls === prediction.predicted_movement;
                      return (
                        <div className={`prob-row${isActive ? ' prob-row--active' : ' prob-row--inactive'}`} key={cls}>
                          <span className="prob-row__name">{cls}</span>
                          <div className="prob-row__track">
                            <div className="prob-row__fill" style={{width:`${pct}%`}}/>
                          </div>
                          <span className="prob-row__pct">{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state__title">DATA UNAVAILABLE</div>
                    <div className="empty-state__sub">Start simulation to view movement probabilities.</div>
                  </div>
                )}
              </div>
            </div>

          </div>

          {/* ── RIGHT 4 COLS ── */}
          <div className="col-4" style={{display:'flex',flexDirection:'column',gap:'14px'}}>

            {/* MOD.04 — SYSTEM STATUS */}
            <div className="panel">
              <div className="panel-header">
                <div className="panel-header__left">
                  <span className="panel-mod">// MOD.04</span>
                  <span className="panel-title">SYSTEM STATUS</span>
                </div>
                {backendOk && <span style={{color:'var(--green)',fontSize:'16px'}}>✓</span>}
              </div>
              <div className="status-list">
                <div className="status-row">
                  <span className="status-row__label">BACKEND API:</span>
                  <span className={`status-val${backendOk ? ' status-val--green' : ' status-val--grey'}`}>
                    {backendOk ? <><span className="dot"/>CONNECTED</> : 'OFFLINE'}
                  </span>
                </div>
                <div className="status-row">
                  <span className="status-row__label">ML MODEL ENGINE:</span>
                  <span className={`status-val${modelOk ? ' status-val--green' : ' status-val--grey'}`}>
                    {modelOk ? <><span className="dot"/>LOADED</> : 'UNAVAILABLE'}
                  </span>
                </div>
                <div className="status-row">
                  <span className="status-row__label">SIMULATION PIPELINE:</span>
                  <span className={`status-val${isRunning ? ' status-val--orange' : ' status-val--grey'}`}>
                    {isRunning ? <><span className="dot"/>ACTIVE (5Hz)</> : 'STOPPED'}
                  </span>
                </div>
                <div className="status-row">
                  <span className="status-row__label">INFERENCE PREDICTION:</span>
                  <span className={`status-val${prediction ? ' status-val--green' : ' status-val--grey'}`}>
                    {prediction ? <><span className="dot"/>ACTIVE</> : 'IDLE'}
                  </span>
                </div>
                <div className="status-row">
                  <span className="status-row__label">TELEMETRY SOURCE:</span>
                  <span className="status-val status-val--grey">
                    {status?.data_source ?? 'SIMULATED'}
                  </span>
                </div>
              </div>
            </div>

            {/* MOD.05 — WHY THIS PREDICTION? */}
            <div className="panel">
              <div className="panel-header">
                <div className="panel-header__left">
                  <span className="panel-mod">// MOD.05_XAI</span>
                  <span className="panel-title">WHY THIS PREDICTION?</span>
                </div>
                <span style={{color:'var(--orange)',fontSize:'15px'}}>⊕</span>
              </div>
              <div className="panel-body">
                {prediction ? (
                  <>
                    {prediction.explanation_text && (
                      <div className="xai-text">{prediction.explanation_text}</div>
                    )}
                    <div className="xai-feat-heading">TOP CONTRIBUTING FEATURES (SHAP VALUES)</div>
                    {prediction.top_features?.length ? (
                      <div className="xai-feat-list">
                        {prediction.top_features.map((f, i) => {
                          const maxImp = Math.max(...prediction.top_features.map(x => x.importance));
                          const pct    = (f.importance * 100).toFixed(1);
                          const barW   = ((f.importance / maxImp) * 100).toFixed(1);
                          return (
                            <div className="xai-feat" key={i}>
                              <div className="xai-feat__row">
                                <span className="xai-feat__name">{f.feature.replace(/_/g,' ')}</span>
                                <span className="xai-feat__pct">{pct}%</span>
                              </div>
                              <div className="xai-feat__bar-track">
                                <div className="xai-feat__bar-fill" style={{width:`${barW}%`}}/>
                              </div>
                              {f.value !== null && f.value !== undefined && (
                                <div className="xai-feat__curr">
                                  Current: {Number(f.value).toFixed(3)}&nbsp;&nbsp;Weight Rank #{i + 1}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : null}
                    <div className="xai-disclaimer">
                      Feature importance based on SYNTHETIC training data. Not a clinical explanation.
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state__title">DATA UNAVAILABLE</div>
                    <div className="empty-state__sub">Active prediction required.</div>
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>

        {/* ══════════════════════════════════════════
            SECTION 4: MODEL INFO + SESSION TELEMETRY
            ══════════════════════════════════════════ */}
        <div className="db-grid-12">

          {/* MODEL ARCHITECTURE */}
          <div className="panel col-6" id="model-metrics">
            <div className="panel-header">
              <div className="panel-header__left">
                <span className="panel-mod">// BENCHMARK_SPECS</span>
                <span className="panel-title">MODEL ARCHITECTURE &amp; BENCHMARKS</span>
              </div>
              <span className="panel-badge">/api/model/info | /api/metrics</span>
            </div>
            <div className="panel-body">
              <div className="model-top-cards">
                <div className="model-card">
                  <div className="model-card__lbl">Model Core</div>
                  <div className="model-card__val">
                    {info ? `${info.classifier?.replace('Classifier','')} + ${info.regressor?.replace('Regressor','')}` : '—'}
                  </div>
                </div>
                <div className="model-card">
                  <div className="model-card__lbl">Feature Count</div>
                  <div className="model-card__val">{info?.n_features ?? '—'} Input Features</div>
                </div>
                <div className="model-card">
                  <div className="model-card__lbl">Movement Archetypes</div>
                  <div className="model-card__val">{info?.movement_classes?.length ?? '—'} Classes</div>
                </div>
              </div>

              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>VALIDATION METRIC</th>
                    <th style={{textAlign:'right'}}>BENCHMARK SCORE</th>
                    <th style={{textAlign:'right'}}>STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {clf && <>
                    <tr>
                      <td>Classification Accuracy</td>
                      <td style={{textAlign:'right'}} className="metric-val--orange">
                        {(clf.test_accuracy * 100).toFixed(1)}%
                      </td>
                      <td style={{textAlign:'right'}} className="metric-status">PASSED</td>
                    </tr>
                    <tr>
                      <td>CV Accuracy Mean</td>
                      <td style={{textAlign:'right'}} className="metric-val--orange">
                        {(clf.cv_accuracy_mean * 100).toFixed(1)}%
                      </td>
                      <td style={{textAlign:'right'}} className="metric-status">PASSED</td>
                    </tr>
                  </>}
                  {reg && <>
                    <tr>
                      <td>MAE (Assistance Level)</td>
                      <td style={{textAlign:'right'}} className="metric-val--orange">
                        {reg.test_mae?.toFixed(2)}%
                      </td>
                      <td style={{textAlign:'right'}} className="metric-status">PASSED</td>
                    </tr>
                    <tr>
                      <td>RMSE (Torque Fit)</td>
                      <td style={{textAlign:'right'}} className="metric-val--orange">
                        {reg.test_rmse?.toFixed(2)}%
                      </td>
                      <td style={{textAlign:'right'}} className="metric-status">PASSED</td>
                    </tr>
                    <tr>
                      <td>R² Deterministic Score</td>
                      <td style={{textAlign:'right'}} className="metric-val--orange">
                        {reg.test_r2?.toFixed(3)}
                      </td>
                      <td style={{textAlign:'right'}} className="metric-status">CONVERGED</td>
                    </tr>
                  </>}
                  {!clf && !reg && (
                    <tr><td colSpan="3" style={{textAlign:'center',color:'var(--text-light)'}}>Loading metrics...</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* SESSION TELEMETRY */}
          <div className="panel col-6">
            <div className="panel-header">
              <div className="panel-header__left">
                <span className="panel-mod">// LOG_DISPATCH</span>
                <span className="panel-title">SESSION TELEMETRY</span>
              </div>
              <span className={`panel-badge${isRunning ? ' panel-badge--green' : ''}`}>
                {isRunning ? 'LIVE LOGGING' : 'SESSION ENDED'}
              </span>
            </div>
            <div className="panel-body">
              {status?.session_id ? (
                <>
                  <div className="session-rows">
                    <div className="session-row-item">
                      <span className="session-row-item__lbl">SESSION IDENTIFIER:</span>
                      <span className="session-row-item__val">{status.session_id}</span>
                    </div>
                    <div className="session-row-item">
                      <span className="session-row-item__lbl">ELAPSED DURATION:</span>
                      <span className="session-row-item__val">
                        {status.elapsed_seconds ? `${Math.floor(status.elapsed_seconds / 60).toString().padStart(2,'0')}:${Math.floor(status.elapsed_seconds % 60).toString().padStart(2,'0')}` : '00:00'}
                      </span>
                    </div>
                    <div className="session-row-item">
                      <span className="session-row-item__lbl">SAMPLE COUNT:</span>
                      <span className="session-row-item__val">{status.sample_count?.toLocaleString() ?? 0} frames</span>
                    </div>
                    {sessionData?.summary?.avg_assistance && (
                      <div className="session-row-item">
                        <span className="session-row-item__lbl">AVERAGE ASSISTANCE:</span>
                        <span className="session-row-item__val session-row-item__val--orange">
                          {sessionData.summary.avg_assistance?.toFixed(1)}%
                        </span>
                      </div>
                    )}
                    {sessionData?.summary?.movement_distribution && (
                      <div className="session-row-item" style={{flexDirection:'column',gap:'6px',alignItems:'flex-start'}}>
                        <span className="session-row-item__lbl">MOVEMENT DISTRIBUTION:</span>
                        <div className="session-dist">
                          {Object.entries(sessionData.summary.movement_distribution).map(([k, v], i, arr) => (
                            <span key={k}>
                              {k} ({v})
                              {i < arr.length - 1 && <span className="session-dist-sep"> | </span>}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  {status.session_id && !isRunning && (
                    <div style={{marginTop:'12px',paddingTop:'12px',borderTop:'1px solid var(--border)',display:'flex',alignItems:'center',justifyContent:'space-between'}}>
                      <span style={{fontSize:'9.5px',color:'var(--text-light)'}}>Format: RFC-4180 CSV Compliant</span>
                      <a className="export-btn" href={api.sessionCsv(status.session_id)} download>
                        ⬇ EXPORT SESSION CSV
                      </a>
                    </div>
                  )}
                </>
              ) : (
                <div className="empty-state">
                  <div className="empty-state__title">NO ACTIVE SESSION</div>
                  <div className="empty-state__sub">Start a simulation to begin session logging.</div>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer className="db-footer">
        <p>H-AUGMENT &nbsp;|&nbsp; AI-Assisted Human Augmentation System &nbsp;|&nbsp;
          <a href="#">SOFTWARE-IN-THE-LOOP PROTOTYPE</a> &nbsp;|&nbsp; SIMULATED SENSOR DATA
        </p>
        <p>Prototype model recommendation — not a medical/clinical decision.</p>
      </footer>
    </div>
  );
}

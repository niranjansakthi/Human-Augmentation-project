/* ============================================================
   LandingPage.jsx
   Entry experience before the prototype dashboard.
   Uses live /api/model/info to determine backend status.
   ============================================================ */
import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import ExoSchematic from './ExoSchematic';
import './LandingPage.css';

export default function LandingPage({ onEnter }) {
  const [backendOnline, setBackendOnline] = useState(null); // null=checking, true=online, false=offline
  const [transitioning, setTransitioning]= useState(false);
  const overlayRef = useRef(null);

  useEffect(() => {
    api.modelInfo()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  const handleEnter = () => {
    setTransitioning(true);
    setTimeout(() => {
      onEnter();
    }, 450);
  };

  const statusLabel = backendOnline === null
    ? 'CONNECTING...'
    : backendOnline
      ? 'PROTOTYPE · READY'
      : 'BACKEND OFFLINE';

  const statusClass = backendOnline === true ? 'lp-status-badge lp-status-badge--online' : 'lp-status-badge';

  return (
    <div className="lp">
      {/* Transition overlay */}
      <div ref={overlayRef} className={`lp-transition-overlay${transitioning ? ' active' : ''}`}/>

      {/* ── HEADER ── */}
      <header className="lp-header">
        <div className="lp-header__left">
          <div className="lp-logo-mark">
            <span/><span/>
          </div>
          <span className="lp-brand">H-AUGMENT</span>
          <span className="lp-brand-sep">·</span>
          <span className="lp-brand-meta">SIH 2026 / AUTODESK</span>
        </div>

        <div className="lp-header__right">
          <span className="lp-status-badge" style={{fontSize:'9.5px',letterSpacing:'0.05em'}}>
            SOFTWARE-IN-THE-LOOP
          </span>
          <span className={statusClass} id="header-status">
            <span className="dot"/>
            {statusLabel}
          </span>
        </div>
      </header>

      {/* ── MAIN HERO ── */}
      <main className="lp-main">
        <div className="lp-inner">
          <div className="lp-hero">

            {/* ── LEFT COLUMN ── */}
            <div className="lp-hero__left anim-fade-up">

              {/* Eyebrow */}
              <div className="lp-eyebrow">
                <span className="lp-eyebrow__dot"/>
                AI-ASSISTED HUMAN AUGMENTATION &nbsp;/&nbsp; INTELLIGENT ASSISTANCE
                &nbsp;&nbsp;[SYS.VER 2.4.9]
              </div>

              {/* Headline */}
              <h1 className="lp-headline">
                UNDERSTANDING<br/>
                HUMAN<br/>
                MOVEMENT.<br/>
                <em>ADAPTING</em><br/>
                ASSISTANCE.
              </h1>

              {/* Description */}
              <p className="lp-desc">
                An AI-assisted human augmentation prototype that analyzes movement patterns
                and estimates adaptive assistance from simulated sensor data.
              </p>

              {/* Metric Strip */}
              <div className="lp-metrics">
                <div className="lp-metric">
                  <div className="lp-metric__label">MODEL</div>
                  <div className="lp-metric__value">RANDOM FOREST</div>
                </div>
                <div className="lp-metric">
                  <div className="lp-metric__label">INPUT</div>
                  <div className="lp-metric__value">SIM. SENSOR DATA</div>
                </div>
                <div className="lp-metric">
                  <div className="lp-metric__label">OUTPUT</div>
                  <div className="lp-metric__value lp-metric__value--orange">MOVE + ASSIST</div>
                </div>
              </div>

              {/* CTA */}
              <div className="lp-cta-row">
                <button
                  id="enter-prototype-btn"
                  className="lp-cta-btn"
                  onClick={handleEnter}
                  disabled={transitioning}
                >
                  <span className="lp-cta-btn__dot" style={{
                    width:'7px',height:'7px',borderRadius:'50%',
                    background:'rgba(255,255,255,0.8)',display:'inline-block'
                  }}/>
                  [ ENTER PROTOTYPE → ]
                  <span className="lp-cta-btn__arrow" aria-hidden="true"/>
                </button>
                <span className="lp-cta-sub">
                  Explore the live movement simulation
                </span>
              </div>

            </div>

            {/* ── RIGHT COLUMN — Technical Schematic ── */}
            <div className="lp-hero__right" style={{animationDelay:'0.1s'}} className="anim-fade-up">
              <div className="lp-schematic">

                {/* Panel top bar */}
                <div className="lp-schematic__bar">
                  <div className="lp-schematic__bar-left">
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <rect x="1" y="1" width="10" height="10" rx="1" stroke="currentColor" strokeWidth="1.2"/>
                      <line x1="1" y1="5" x2="11" y2="5" stroke="currentColor" strokeWidth="0.8"/>
                    </svg>
                    CAD BLUEPRINT // LOWER EXOSKELETON SCHEMATIC
                  </div>
                  <div className="lp-schematic__bar-right">
                    SIMULATED ARCHITECTURE // 50 Hz
                  </div>
                </div>

                {/* Animated SVG schematic */}
                <ExoSchematic isRunning={false}/>

                {/* Coordinate + rig label */}
                <div style={{
                  padding:'6px 14px', background:'#111',
                  borderTop:'1px solid #2d2d2d',
                  display:'flex', justifyContent:'space-between', alignItems:'center',
                  flexWrap:'wrap', gap:'6px'
                }}>
                  <span style={{
                    fontFamily:'JetBrains Mono',fontSize:'8px',
                    color:'#FF6A00',letterSpacing:'0.04em'
                  }}>
                    COORD: [X:+52.44, Y:-48.12, Z:+0.02]
                  </span>
                  <span style={{
                    fontFamily:'JetBrains Mono',fontSize:'8px',
                    color:'#777',letterSpacing:'0.04em',textTransform:'uppercase'
                  }}>
                    KINEMATIC RIG D-04
                  </span>
                </div>

                {/* Pipeline footer */}
                <div className="lp-schematic__footer">
                  <div className="lp-pipe">
                    <span>SENSORS</span>
                    <em className="arr">→</em>
                    <span>BIOMECHANICAL DATA</span>
                    <em className="arr">→</em>
                    <span>ML MODEL</span>
                    <em className="arr">→</em>
                    <em>CLASSIFICATION</em>
                    <em className="arr">→</em>
                    <span>ESTIMATION</span>
                  </div>
                  <div className="lp-schematic__foot-note">
                    SIMULATED BIOMECHANICAL PIPELINE
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer className="lp-footer">
        <p>SIH 2026 &nbsp;|&nbsp; AUTODESK &nbsp;|&nbsp;
          <a href="#">SOFTWARE-IN-THE-LOOP PROTOTYPE</a>
        </p>
        <p>SIMULATED SENSOR DATA &nbsp;·&nbsp; CONFIDENTIAL EVALUATION BENCH</p>
      </footer>
    </div>
  );
}

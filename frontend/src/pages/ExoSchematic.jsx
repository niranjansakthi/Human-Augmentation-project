/* ExoSchematic.jsx — Animated lower-limb exoskeleton technical schematic */
import './ExoSchematic.css';

export default function ExoSchematic({ isRunning = false }) {
  return (
    <svg
      className="lp-schematic__svg"
      viewBox="0 0 480 360"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Illustrative exoskeleton schematic — not real hardware"
    >
      <defs>
        {/* Grid */}
        <pattern id="g" width="24" height="24" patternUnits="userSpaceOnUse">
          <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#252525" strokeWidth="0.7"/>
        </pattern>
        {/* Orange glow */}
        <radialGradient id="glow-k" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#FF6A00" stopOpacity="0.25"/>
          <stop offset="100%" stopColor="#FF6A00" stopOpacity="0"/>
        </radialGradient>
        <filter id="glow-f" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>

      {/* Background grid */}
      <rect width="480" height="360" fill="url(#g)"/>

      {/* ── Coordinate marks ── */}
      <text x="8" y="14" fill="#333" fontSize="8" fontFamily="JetBrains Mono">X:+52.44</text>
      <text x="8" y="24" fill="#333" fontSize="8" fontFamily="JetBrains Mono">Y:-48.12</text>
      <text x="8" y="34" fill="#333" fontSize="8" fontFamily="JetBrains Mono">Z:+0.02</text>

      {/* ── Hip Actuator Block ── */}
      <rect x="190" y="34" width="100" height="20" rx="2"
        stroke="#FF6A00" strokeWidth="1.8" fill="#1f1f1f"
        className="exo-label exo-label--1"/>
      <circle cx="240" cy="44" r="5" fill="#FF6A00" filter="url(#glow-f)"
        className="exo-node exo-node--1"/>
      <text x="175" y="28" fill="#8a8078" fontSize="8" fontFamily="JetBrains Mono"
        className="exo-label exo-label--1">ACTUATOR</text>
      <text x="167" y="37" fill="#666" fontSize="7" fontFamily="JetBrains Mono"
        className="exo-label exo-label--1">BRUSHLESS TORQUE DRIVE</text>

      {/* ── Left Thigh Strut ── */}
      <line x1="225" y1="54" x2="200" y2="140"
        stroke="#FF6A00" strokeWidth="2.5" strokeLinecap="round"
        className="exo-strut exo-strut--1"/>
      <line x1="232" y1="54" x2="207" y2="140"
        stroke="#FF6A00" strokeWidth="0.6" strokeDasharray="3 3"
        opacity="0.4" className="exo-strut exo-strut--1"/>

      {/* ── Right Thigh Strut ── */}
      <line x1="255" y1="54" x2="280" y2="140"
        stroke="#FF6A00" strokeWidth="2.5" strokeLinecap="round"
        className="exo-strut exo-strut--2"/>

      {/* ── Dimension annotation ── */}
      <path d="M 180 44 L 168 44 L 168 140 L 193 140"
        stroke="#444" strokeDasharray="3 3" strokeWidth="1" fill="none"
        className="exo-label exo-label--2"/>
      <text x="100" y="94" fill="#FF6A00" fontSize="9" fontWeight="600" fontFamily="JetBrains Mono"
        className="exo-label exo-label--2">Δθ: 42.8°</text>
      <text x="100" y="105" fill="#555" fontSize="8" fontFamily="JetBrains Mono"
        className="exo-label exo-label--2">L_TOTAL: 640mm</text>

      {/* ── KNEE JOINT (LEFT) — highlighted ── */}
      <ellipse cx="240" cy="36" rx="120" ry="80" fill="url(#glow-k)" opacity="0.4"
        style={{transform:'translateY(108px)'}}/>
      <circle cx="200" cy="144" r="18" stroke="#FF6A00" strokeWidth="1.5" fill="none"
        className={`exo-knee-pulse ${isRunning ? 'exo-knee-pulse--anim' : ''} exo-node exo-node--2`}/>
      <circle cx="200" cy="144" r="10" stroke="#FF6A00" strokeWidth="2" fill="#1c1410"
        className="exo-node exo-node--2"/>
      <circle cx="200" cy="144" r="4" fill="#FF6A00" filter="url(#glow-f)"
        className="exo-node exo-node--2"/>

      {/* KNEE SENSOR label */}
      <line x1="218" y1="144" x2="290" y2="130"
        stroke="#FF6A00" strokeWidth="0.8" strokeDasharray="3 3"
        className="exo-ray exo-ray--1"/>
      <circle cx="291" cy="130" r="2.5" fill="#FF6A00" className="exo-label exo-label--2"/>
      <text x="296" y="127" fill="#FF6A00" fontSize="8.5" fontWeight="600" fontFamily="JetBrains Mono"
        className="exo-label exo-label--2">KNEE SENSOR</text>
      <text x="296" y="137" fill="#666" fontSize="7.5" fontFamily="JetBrains Mono"
        className="exo-label exo-label--2">ROTARY ANGLE &amp; VELOCITY</text>

      {/* ── KNEE JOINT (RIGHT) ── */}
      <circle cx="280" cy="144" r="10" stroke="#FF6A00" strokeWidth="2" fill="#1c1410"
        className="exo-node exo-node--2"/>
      <circle cx="280" cy="144" r="4" fill="#FF6A00" className="exo-node exo-node--2"/>

      {/* ── Left Shin Strut ── */}
      <line x1="200" y1="154" x2="185" y2="250"
        stroke="#FF6A00" strokeWidth="2.5" strokeLinecap="round"
        className="exo-strut exo-strut--3"/>

      {/* ── Right Shin Strut ── */}
      <line x1="280" y1="154" x2="295" y2="250"
        stroke="#FF6A00" strokeWidth="2.5" strokeLinecap="round"
        className="exo-strut exo-strut--4"/>

      {/* ── IMU Sensor mid-shin ── */}
      <circle cx="193" cy="200" r="4" fill="#FFFFFF" className="exo-node exo-node--3"/>
      <line x1="197" y1="200" x2="290" y2="190"
        stroke="#FF6A00" strokeWidth="0.8" strokeDasharray="3 3"
        className="exo-ray exo-ray--2"/>
      <text x="294" y="188" fill="#8a8078" fontSize="8.5" fontFamily="JetBrains Mono"
        className="exo-label exo-label--3">IMU SENSOR</text>
      <text x="294" y="198" fill="#555" fontSize="7.5" fontFamily="JetBrains Mono"
        className="exo-label exo-label--3">A-IMU AXIAL / GYRO DATA</text>

      {/* ── Ankle Pivot ── */}
      <circle cx="185" cy="254" r="7" stroke="#888" strokeWidth="1.5" fill="#1f1f1f"
        className="exo-node exo-node--3"/>
      <circle cx="295" cy="254" r="7" stroke="#888" strokeWidth="1.5" fill="#1f1f1f"
        className="exo-node exo-node--3"/>

      {/* ── Foot / Force Plate ── */}
      <path d="M 162 265 L 228 265 L 235 274 L 155 274 Z"
        stroke="#FF6A00" strokeWidth="1.5" fill="#1e1108"
        className="exo-strut exo-strut--5"/>
      <path d="M 265 265 L 331 265 L 338 274 L 258 274 Z"
        stroke="#FF6A00" strokeWidth="1.5" fill="#1e1108"
        className="exo-strut exo-strut--5"/>

      {/* FORCE SENSOR label */}
      <line x1="200" y1="270" x2="290" y2="255"
        stroke="#FF6A00" strokeWidth="0.8" strokeDasharray="3 3"
        className="exo-label exo-label--4"/>
      <text x="294" y="253" fill="#8a8078" fontSize="8.5" fontFamily="JetBrains Mono"
        className="exo-label exo-label--4">FORCE SENSOR</text>
      <text x="294" y="263" fill="#555" fontSize="7.5" fontFamily="JetBrains Mono"
        className="exo-label exo-label--4">AXIAL / SHEAR CELL (GRF)</text>

      {/* ── Movement data callout (top right) ── */}
      <line x1="280" y1="144" x2="390" y2="90"
        stroke="#FF6A00" strokeWidth="0.8" strokeDasharray="3 3"
        className="exo-label exo-label--3"/>
      <circle cx="391" cy="89" r="2.5" fill="#FF6A00"/>
      <text x="396" y="87" fill="#fff" fontSize="8.5" fontFamily="JetBrains Mono"
        className="exo-label exo-label--3">MOVEMENT DATA</text>
      <text x="396" y="97" fill="#888" fontSize="7.5" fontFamily="JetBrains Mono"
        className="exo-label exo-label--3">L: 440mm / T: 42.1° / ACC: 80.4 (deg/s)</text>

      {/* ── Watermarks ── */}
      <text x="240" y="318" fill="#333" fontSize="8" textAnchor="middle" fontFamily="JetBrains Mono"
        className="exo-label exo-label--4">KINEMATIC RIG D-04</text>
      <text x="240" y="330" fill="#FF6A00" fontSize="7.5" textAnchor="middle" fontFamily="JetBrains Mono"
        opacity="0.6" className="exo-label exo-label--4">
        ILLUSTRATIVE SOFTWARE PROTOTYPE — NOT REAL MEDICAL HARDWARE
      </text>

      {/* ── Scanning line overlay (running) ── */}
      {isRunning && (
        <g className="exo-scanline">
          <clipPath id="scan-clip">
            <rect x="0" y="0" width="480" height="360"/>
          </clipPath>
          <rect x="0" y="0" width="480" height="3"
            fill="rgba(255,106,0,0.12)"
            style={{animation:'scan-line 3s linear infinite'}}
            clipPath="url(#scan-clip)"/>
        </g>
      )}
    </svg>
  );
}

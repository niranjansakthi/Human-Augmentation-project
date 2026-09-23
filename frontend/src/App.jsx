/* ============================================================
   App.jsx  —  Router between Landing and Dashboard
   ============================================================ */
import { useState } from 'react';
import LandingPage from './pages/LandingPage';
import Dashboard   from './pages/Dashboard';

export default function App() {
  const [page, setPage] = useState('landing'); // 'landing' | 'dashboard'

  return page === 'landing'
    ? <LandingPage onEnter={() => setPage('dashboard')} />
    : <Dashboard   onBack={() => setPage('landing')}   />;
}

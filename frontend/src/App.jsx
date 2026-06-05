/**
 * App.jsx — TerraVision AI main application.
 * Wires together PredictForm, SatMap, and ResultsPanel.
 * Handles API state with React hooks — no external state library needed.
 */
import { useState, useCallback } from 'react'
import SatMap       from './components/SatMap.jsx'
import PredictForm  from './components/PredictForm.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'
import { predict }  from './api/predict.js'

const STEPS = [
  '🛰️ Querying 6-month Sentinel-2 sequence…',
  '🌡️ Pulling ERA5-Land climate data…',
  '⚡ Running temporal transformer…',
  '📊 Computing MC Dropout confidence…',
]

export default function App() {
  const [lat,      setLat]      = useState(31.5204)
  const [lon,      setLon]      = useState(74.3587)
  const [crop,     setCrop]     = useState('Wheat')
  const [mcPasses, setMcPasses] = useState(20)
  const [loading,  setLoading]  = useState(false)
  const [step,     setStep]     = useState('')
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)

  // Field change handler — routes to correct setter
  const handleChange = useCallback((field, value) => {
    const setters = { lat: setLat, lon: setLon, crop: setCrop, mcPasses: setMcPasses }
    setters[field]?.(value)
  }, [])

  // Preset selection — updates all three fields at once
  const handlePreset = useCallback(({ lat: pLat, lon: pLon, crop: pCrop }) => {
    setLat(pLat); setLon(pLon); setCrop(pCrop)
    setResult(null); setError(null)
  }, [])

  // Run inference
  const handleSubmit = useCallback(async () => {
    setLoading(true); setError(null); setResult(null)

    let si = 0
    setStep(STEPS[0])
    const iv = setInterval(() => {
      si = (si + 1) % STEPS.length
      setStep(STEPS[si])
    }, 2_500)

    try {
      const data = await predict({
        lat, lon, crop,
        mc_passes:         mcPasses,
        include_report:    true,
        include_ndvi_tile: true,
      })
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      clearInterval(iv)
      setLoading(false)
      setStep('')
    }
  }, [lat, lon, crop, mcPasses])

  return (
    <>
      {/* Navigation */}
      <nav className="app-nav">
        <span className="nav-brand">🛰️ TERRAVISION AI</span>
        <div className="nav-links">
          <a className="nav-link" href="/v1/docs" target="_blank" rel="noopener noreferrer">API Docs</a>
          <a className="nav-link" href="https://github.com/ahmedabbas52233/TerraVision-AI"
             target="_blank" rel="noopener noreferrer">GitHub</a>
          <a className="nav-link" href="https://terravision-ai.streamlit.app"
             target="_blank" rel="noopener noreferrer">Streamlit</a>
        </div>
      </nav>

      {/* Main grid: panel | map */}
      <div className="app-body">

        {/* Left panel */}
        <div className="panel">
          <PredictForm
            lat={lat} lon={lon} crop={crop} mcPasses={mcPasses}
            loading={loading}
            onChange={handleChange}
            onPreset={handlePreset}
            onSubmit={handleSubmit}
          />

          <hr className="divider" />

          {/* Loading state */}
          {loading && (
            <div className="spinner-wrap">
              <div className="spinner" />
              <div className="spinner-text">{step}</div>
            </div>
          )}

          {/* Error state */}
          {error && !loading && (
            <div className="error-box">⛔ {error}</div>
          )}

          {/* Results */}
          {result && !loading && (
            <ResultsPanel data={result} />
          )}
        </div>

        {/* Right — satellite map */}
        <SatMap lat={lat} lon={lon} result={result} />
      </div>

      {/* Footer */}
      <footer className="app-footer">
        © 2026 TerraVision AI ·{' '}
        <a href="mailto:ahmedabbas52233@gmail.com">Ahmad Abbas Hussain</a> ·{' '}
        <a href="https://github.com/ahmedabbas52233/TerraVision-AI"
           target="_blank" rel="noopener noreferrer">GitHub</a>
      </footer>
    </>
  )
}

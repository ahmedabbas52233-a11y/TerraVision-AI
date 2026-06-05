/**
 * PredictForm.jsx
 * Coordinate inputs, crop selector, MC passes slider, and regional presets.
 */

const CROPS = ['Wheat', 'Rice', 'Maize', 'Soybean']

const PRESETS = [
  { label: 'Punjab, Pakistan 🌾',     lat: 31.5204, lon:  74.3587, crop: 'Wheat'   },
  { label: 'Kansas, USA 🌾',          lat: 38.5000, lon: -98.0000, crop: 'Wheat'   },
  { label: 'Mato Grosso, Brazil 🌽',  lat: -12.500, lon: -55.5000, crop: 'Maize'   },
  { label: 'Rift Valley, Kenya 🌽',   lat:  1.0189, lon:  34.9542, crop: 'Maize'   },
  { label: 'Hunan, China 🍚',         lat: 27.6104, lon: 111.7088, crop: 'Rice'    },
  { label: 'Iowa, USA 🌱',            lat: 42.0000, lon: -93.5000, crop: 'Soybean' },
]

export default function PredictForm({ lat, lon, crop, mcPasses, loading, onChange, onPreset, onSubmit }) {
  return (
    <div>
      <div className="panel-title">📍 Field Parameters</div>

      <div className="coord-row">
        <div className="field">
          <label>Latitude</label>
          <input type="number" step="0.0001" min="-90" max="90"
                 value={lat}
                 onChange={e => onChange('lat', parseFloat(e.target.value) || 0)} />
        </div>
        <div className="field">
          <label>Longitude</label>
          <input type="number" step="0.0001" min="-180" max="180"
                 value={lon}
                 onChange={e => onChange('lon', parseFloat(e.target.value) || 0)} />
        </div>
      </div>

      <div className="field">
        <label>Crop Type</label>
        <select value={crop} onChange={e => onChange('crop', e.target.value)}>
          {CROPS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      <div className="field">
        <label>MC Dropout Passes ({mcPasses})</label>
        <input type="range" min="5" max="50" step="5"
               value={mcPasses}
               onChange={e => onChange('mcPasses', parseInt(e.target.value))}
               style={{ padding: 0, background: 'transparent', border: 'none',
                        accentColor: 'var(--accent)' }} />
      </div>

      <button className="btn-primary" onClick={onSubmit} disabled={loading}>
        {loading ? 'Running…' : '🚀 Run Live Inference'}
      </button>

      <hr className="divider" />

      <div className="panel-title" style={{ marginTop: '.2rem' }}>Quick Presets</div>
      {PRESETS.map(p => (
        <button key={p.label} className="preset-btn" onClick={() => onPreset(p)}>
          {p.label}
        </button>
      ))}
    </div>
  )
}

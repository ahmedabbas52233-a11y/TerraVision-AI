/**
 * ResultsPanel.jsx
 * Displays inference results: yield metrics, NDVI alert, ERA5 climate,
 * real MC Dropout confidence bar, and report download.
 */
import ConfidenceBar from './ConfidenceBar.jsx'

const ALERT_CLASS = {
  success: 'alert-success',
  info:    'alert-info',
  warning: 'alert-warning',
  error:   'alert-error',
}

export default function ResultsPanel({ data }) {
  const ns     = data.ndvi_status ?? {}
  const e5     = data.era5 ?? {}
  const delta  = data.yield_delta_t_ha ?? 0
  const sign   = delta >= 0 ? '+' : ''

  // Downloadable plain-text report
  const reportHref = data.report
    ? URL.createObjectURL(new Blob([data.report], { type: 'text/plain' }))
    : null
  const reportName = `TerraVision_${data.crop}_${data.lat?.toFixed(4)}_${data.lon?.toFixed(4)}.txt`

  return (
    <div>
      {/* NDVI status badge */}
      <div style={{ marginBottom: '.75rem' }}>
        <span className={`alert ${ALERT_CLASS[ns.alert_type] ?? 'alert-info'}`}
              style={{ display: 'block' }}>
          <strong>{ns.label}</strong>
          <br />
          {ns.action}
        </span>
      </div>

      {/* Primary yield + NDVI metrics */}
      <div className="metric-grid">
        <div className="metric-card">
          <span className="metric-val">{data.yield_adj_t_ha?.toFixed(2)}</span>
          <span className="metric-lbl">ERA5-Adj Yield (t/ha)</span>
        </div>
        <div className="metric-card">
          <span className="metric-val">{data.ndvi?.toFixed(4)}</span>
          <span className="metric-lbl">NDVI Index</span>
        </div>
      </div>

      {/* Secondary: base yield + delta */}
      <div className="metric-grid">
        <div className="metric-card">
          <span className="metric-val" style={{ fontSize: '1.05rem' }}>
            {data.yield_base_t_ha?.toFixed(2)}
          </span>
          <span className="metric-lbl">Base Yield (t/ha)</span>
        </div>
        <div className="metric-card">
          <span className="metric-val"
                style={{ fontSize: '1.05rem',
                         color: delta >= 0 ? 'var(--accent)' : 'var(--danger)' }}>
            {sign}{delta.toFixed(2)}
          </span>
          <span className="metric-lbl">ERA5 Delta (t/ha)</span>
        </div>
      </div>

      {/* Carbon + model name */}
      <div className="metric-grid">
        <div className="metric-card">
          <span className="metric-val" style={{ fontSize: '1.1rem' }}>
            {data.carbon_mg_c_ha?.toFixed(2)}
          </span>
          <span className="metric-lbl">Carbon (Mg C/ha)</span>
        </div>
        <div className="metric-card">
          <span className="metric-val"
                style={{ fontSize: '.75rem', letterSpacing: '1px', lineHeight: 1.3 }}>
            {data.model_name ?? '—'}
          </span>
          <span className="metric-lbl">Architecture</span>
        </div>
      </div>

      {/* Real MC Dropout confidence — Gap 1 fix */}
      <ConfidenceBar
        confidencePct={data.confidence_pct}
        stdYield={data.yield_std_t_ha}
        ciLower={data.ci_95_lower}
        ciUpper={data.ci_95_upper}
      />

      <hr className="divider" />

      {/* ERA5 climate detail */}
      {e5.source === 'era5-land' && (
        <div className="era5-block">
          <div className="era5-title">🌡️ ERA5-Land Climate (30-day)</div>
          <div className="era5-row"><span>Air Temp (2 m)</span>{e5.temp_c?.toFixed(1)} °C</div>
          <div className="era5-row"><span>Monthly Precip.</span>{e5.precip_mm_month?.toFixed(1)} mm</div>
          <div className="era5-row"><span>Source</span>GEE · ERA5-Land</div>
        </div>
      )}

      {/* Metadata */}
      <div style={{ fontSize: '.68rem', color: 'var(--muted)', marginBottom: '.6rem' }}>
        v{data.model_version} · {data.inference_ms?.toFixed(0)} ms · {data.generated_utc}
      </div>

      {/* Report download */}
      {reportHref && (
        <a className="btn-outline" href={reportHref} download={reportName}
           style={{ display: 'block', textAlign: 'center', textDecoration: 'none' }}>
          📥 Download Intelligence Report
        </a>
      )}
    </div>
  )
}

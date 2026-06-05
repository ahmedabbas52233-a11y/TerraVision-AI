/**
 * ConfidenceBar.jsx
 * Visualises real MC Dropout confidence with a progress bar and 95 % CI.
 * Replaces the old hardcoded 94.2 % displayed as plain text.
 */
export default function ConfidenceBar({ confidencePct, stdYield, ciLower, ciUpper }) {
  const pct = Math.min(100, Math.max(0, confidencePct))

  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-label">
        <span>Confidence (MC Dropout)</span>
        <span style={{ color: 'var(--accent)', fontFamily: "'Orbitron', monospace" }}>
          {pct.toFixed(1)} %
        </span>
      </div>

      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${pct}%` }} />
      </div>

      <div className="conf-ci">
        ±{stdYield?.toFixed(3)} t/ha &nbsp;·&nbsp;
        95 % CI [{ciLower?.toFixed(2)}, {ciUpper?.toFixed(2)}] t/ha
      </div>
    </div>
  )
}

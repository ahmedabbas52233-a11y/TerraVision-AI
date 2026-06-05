/**
 * SatMap.jsx — Leaflet satellite map component.
 *
 * Shows:
 *   · Google Hybrid satellite base layer
 *   · Analysis target marker with popup
 *   · 500 m Sentinel-2 buffer circle
 *   · 10 km ERA5 buffer circle (dashed)
 *   · NDVI heatmap TileLayer when tile URL available
 *   · Layer control toggle
 */
import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix Leaflet default marker icon paths broken by Vite bundling
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl:       new URL('leaflet/dist/images/marker-icon.png',    import.meta.url).href,
  iconRetinaUrl: new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href,
  shadowUrl:     new URL('leaflet/dist/images/marker-shadow.png',  import.meta.url).href,
})

const GOOGLE_HYBRID = L.tileLayer(
  'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
  { attribution: '© Google Maps', maxZoom: 20 }
)

export default function SatMap({ lat, lon, result }) {
  const containerRef = useRef(null)
  const mapRef       = useRef(null)
  const layersRef    = useRef({ marker: null, buf500: null, buf10k: null, ndvi: null, ctrl: null })

  // ── Init map once ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (mapRef.current) return
    const map = L.map(containerRef.current, { zoomControl: true })
      .setView([lat, lon], 13)
    GOOGLE_HYBRID.addTo(map)
    mapRef.current = map
    return () => { map.remove(); mapRef.current = null }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Update layers when lat/lon/result changes ────────────────────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    map.setView([lat, lon], 13)
    const L2 = layersRef.current

    // Clean previous layers
    L2.marker?.remove()
    L2.buf500?.remove()
    L2.buf10k?.remove()
    L2.ndvi?.remove()
    L2.ctrl?.remove()

    // Analysis target marker
    L2.marker = L.circleMarker([lat, lon], {
      radius: 8, color: '#00ffaa', fillColor: '#00ffaa', fillOpacity: 1, weight: 2,
    })
      .addTo(map)
      .bindPopup(
        `<b>Analysis Target</b><br>${lat.toFixed(4)}°, ${lon.toFixed(4)}°`
        + (result ? `<br>Crop: ${result.crop}<br>NDVI: ${result.ndvi?.toFixed(4)}` : '')
      )

    // 500 m Sentinel-2 buffer
    L2.buf500 = L.circle([lat, lon], {
      radius: 500, color: '#00ffaa', weight: 1.5,
      fillColor: '#00ffaa', fillOpacity: 0.10,
    }).addTo(map).bindTooltip('500 m Sentinel-2 analysis buffer')

    // 10 km ERA5 buffer (dashed)
    L2.buf10k = L.circle([lat, lon], {
      radius: 10_000, color: '#00c8ff', weight: 1,
      dashArray: '6 4', fillColor: '#00c8ff', fillOpacity: 0.03,
    }).addTo(map).bindTooltip('10 km ERA5-Land sampling buffer')

    // NDVI heatmap overlay + layer control
    if (result?.ndvi_tile_url) {
      const ndviLayer = L.tileLayer(result.ndvi_tile_url, {
        opacity: 0.72, attribution: 'GEE · Sentinel-2 NDVI',
      })
      const baseLayers  = { 'Google Hybrid': GOOGLE_HYBRID }
      const overlays    = { 'NDVI Heatmap': ndviLayer }
      L2.ndvi = ndviLayer.addTo(map)
      L2.ctrl = L.control.layers(baseLayers, overlays, { collapsed: false }).addTo(map)
    }

    layersRef.current = L2
  }, [lat, lon, result])

  return (
    <div className="map-wrapper">
      <div ref={containerRef} style={{ height: '100%', width: '100%' }} />
    </div>
  )
}

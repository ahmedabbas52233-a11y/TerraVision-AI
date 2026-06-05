/**
 * TerraVision AI — FastAPI client
 * Centralises all /v1/* calls; handles errors and auth headers.
 */

const BASE = import.meta.env.VITE_API_URL ?? ''
const KEY  = import.meta.env.VITE_API_KEY  ?? 'dev-insecure-key'

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key':    KEY,
      ...(options.headers ?? {}),
    },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `HTTP ${res.status}`)
  }
  return res.json()
}

/**
 * GET /v1/health — no auth required
 * @returns {Promise<{status: string, model_ready: boolean, gee_ready: boolean, model_name: string}>}
 */
export async function getHealth() {
  return apiFetch('/v1/health', { headers: {} })  // public endpoint
}

/**
 * GET /v1/crops — returns supported crop types with agronomic priors
 * @returns {Promise<Array<{name: string, temp_K: number, moisture: number}>>}
 */
export async function getCrops() {
  return apiFetch('/v1/crops')
}

/**
 * POST /v1/predict — full crop yield inference
 *
 * @param {{lat: number, lon: number, crop: string, mc_passes?: number,
 *          include_report?: boolean, include_ndvi_tile?: boolean}} params
 * @returns {Promise<PredictResponse>}
 */
export async function predict(params) {
  return apiFetch('/v1/predict', {
    method: 'POST',
    body:   JSON.stringify({
      lat:               params.lat,
      lon:               params.lon,
      crop:              params.crop,
      mc_passes:         params.mc_passes         ?? 20,
      include_report:    params.include_report    ?? true,
      include_ndvi_tile: params.include_ndvi_tile ?? true,
    }),
  })
}

/**
 * @typedef {Object} PredictResponse
 * @property {number}  ndvi
 * @property {Object}  ndvi_status
 * @property {number}  yield_base_t_ha
 * @property {number}  yield_adj_t_ha
 * @property {number}  yield_delta_t_ha
 * @property {number}  confidence_pct     — real MC Dropout confidence
 * @property {number}  yield_std_t_ha     — ±uncertainty (t/ha)
 * @property {number}  ci_95_lower
 * @property {number}  ci_95_upper
 * @property {number}  carbon_mg_c_ha
 * @property {Object}  era5
 * @property {string}  model_name
 * @property {number}  inference_ms
 * @property {?string} report
 * @property {?string} ndvi_tile_url
 */

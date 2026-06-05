# ════════════════════════════════════════════════════════════════════════════
# TerraVision AI — Multi-Stage Production Dockerfile
#
# Stages
# ──────
#   builder  : installs Python deps into /root/.local (no dev tools in prod)
#   runtime  : minimal image, non-root user, HEALTHCHECK
#
# Usage
# ─────
#   docker build -t terravision-ai .
#
#   # FastAPI REST (port 8000)
#   docker run -p 8000:8000 \
#     -e TERRAVISION_API_KEY=your-secret \
#     -e GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}' \
#     terravision-ai
#
#   # Streamlit app (port 8501)
#   docker run -p 8501:8501 \
#     -e GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}' \
#     terravision-ai \
#     streamlit run app.py --server.port=8501 --server.address=0.0.0.0
# ════════════════════════════════════════════════════════════════════════════

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into user site (no --system, easy to copy to runtime stage)
RUN pip install --upgrade pip \
 && pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: production runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Minimal runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash terravision

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/terravision/.local

# Copy application source
COPY --chown=terravision:terravision . .

# Ensure .local/bin is on PATH for the non-root user
ENV PATH="/home/terravision/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TERRAVISION_ENV=production

USER terravision

# Default: FastAPI REST on port 8000
EXPOSE 8000

# Health check against the public /v1/health endpoint
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=20s \
    --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

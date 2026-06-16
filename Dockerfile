# ─────────────────────────────────────────────────────────────────────────────
# TerraVision AI — Dockerfile
# Target : Railway (uvicorn api:app --host 0.0.0.0 --port 8000)
#
# FIX: previous build timed out at 18+ min because `chown -R` was applied to
# the pip install directory containing PyTorch and 90+ heavy packages.
#
# Solution: install packages as root (fine — they only need to be readable),
# then chown ONLY the /app source directory which is tiny (~1 MB).
# Site-packages stay owned by root; the app user can read but not write them,
# which is exactly what production requires.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
# libgomp1  : OpenMP runtime required by PyTorch CPU kernels
# curl      : healthcheck probe
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps  (installed as root — no chown needed) ───────────────────────
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Non-root user ─────────────────────────────────────────────────────────────
# Created AFTER pip install so site-packages stay root-owned (readable by all).
# Only /app is chowned — fast because it is just source files.
RUN groupadd -r terravision \
    && useradd -r -g terravision -s /bin/false -d /app terravision

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# chown only source files — NOT site-packages (avoids the 18-min timeout)
RUN chown -R terravision:terravision /app

# ── Runtime ───────────────────────────────────────────────────────────────────
USER terravision
EXPOSE 8000

# Healthcheck — Railway uses this to confirm the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
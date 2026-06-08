# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first, then everything else from requirements
RUN pip install --upgrade pip \
 && pip install --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash terravision
WORKDIR /app

COPY --from=builder /root/.local /home/terravision/.local
RUN chown -R terravision:terravision /home/terravision/.local

COPY --chown=terravision:terravision . .

ENV PATH="/home/terravision/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TERRAVISION_ENV=production

USER terravision
EXPOSE 8000 8501

# Default to API; compose overrides for Streamlit
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
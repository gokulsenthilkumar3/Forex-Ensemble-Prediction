# ───────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder: install all dependencies into a virtual environment
# Using a slim base keeps the final image small
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to maximise layer caching
COPY requirements.txt .

# Create venv and install deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ───────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime: copy only the venv and project source
# No build tools in the final image → smaller + more secure
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source (excludes anything in .dockerignore)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Environment defaults (override at runtime)
ENV DATA_PATH="Forex_Data.csv" \
    OUTPUT_DIR="outputs" \
    FEAT_CONFIG="config/features.yaml" \
    MLFLOW_TRACKING_URI="mlruns" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose MLflow UI port
EXPOSE 5000

# Default: run training pipeline
CMD ["python", "main.py"]

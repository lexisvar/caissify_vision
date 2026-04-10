# =============================================================================
# ChessVision — Multi-Stage Dockerfile
# =============================================================================
# Stage 1: builder  — install all heavy deps into a venv
# Stage 2: runtime  — lean image, copy only the venv
#
# CPU build (default — no CUDA deps, fast build):
#   docker build -t chessvision .
#
# GPU build (NVIDIA CUDA 12.1):
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
#                --build-arg TORCH_INDEX="https://download.pytorch.org/whl/cu121" \
#                -t chessvision:gpu .
# =============================================================================

ARG BASE_IMAGE=python:3.11-slim-bookworm

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder

# Install system-level build tools & OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (layer-caching)
COPY requirements.txt .

# Create venv and install all Python deps.
# torch/torchvision are installed first from the CPU-only wheel index so that
# ultralytics (and other packages) never pull in the heavy NVIDIA CUDA wheels.
# For GPU builds pass:  --build-arg TORCH_INDEX="https://download.pytorch.org/whl/cu121"
ARG TORCH_INDEX="https://download.pytorch.org/whl/cpu"
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        --index-url "${TORCH_INDEX}" \
        torch torchvision && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

# Only runtime .so libs needed by OpenCV / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 chessvision
WORKDIR /app

# Copy application source
COPY --chown=chessvision:chessvision . .

# Create directories for model weights and calibration data
RUN mkdir -p /app/models/corner_detector \
             /app/models/piece_detector \
             /app/models/square_classifier \
             /app/data \
    && chown -R chessvision:chessvision /app/models /app/data

USER chessvision

# Expose FastAPI port
EXPOSE 8006

# Health check — hits the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]

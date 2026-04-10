#!/usr/bin/env bash
# =============================================================================
# ChessVision — Container Entrypoint
# =============================================================================
# Runs pre-flight checks then starts the FastAPI server.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " ChessVision API — starting up"
echo "============================================================"

# ── 1. Validate required environment variables ────────────────────────────────
required_vars=(
    "CHESSVISION_ENV"
    "LOG_LEVEL"
)
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required environment variable '$var' is not set."
        exit 1
    fi
done

# ── 2. Check model weights are present ───────────────────────────────────────
corner_model="${MODEL_CORNER_PATH:-/app/models/corner_detector/best.pt}"
piece_model="${MODEL_PIECE_PATH:-/app/models/piece_detector/best.pt}"

if [[ ! -f "$corner_model" ]]; then
    echo "WARNING: Corner detector weights not found at $corner_model"
    echo "         The API will start but board detection will be unavailable."
    echo "         Mount the model weights volume or set MODEL_CORNER_PATH."
fi

if [[ ! -f "$piece_model" ]]; then
    echo "WARNING: Piece detector weights not found at $piece_model"
    echo "         The API will start but piece recognition will be unavailable."
    echo "         Mount the model weights volume or set MODEL_PIECE_PATH."
fi

# ── 3. Decide server mode ─────────────────────────────────────────────────────
if [[ "${CHESSVISION_ENV}" == "development" ]]; then
    echo "Mode: DEVELOPMENT — uvicorn with --reload"
    exec uvicorn api.main:app \
        --host 0.0.0.0 \
        --port "${API_PORT:-8006}" \
        --reload \
        --log-level "${LOG_LEVEL:-debug}"
else
    echo "Mode: PRODUCTION — uvicorn with ${WORKERS:-2} workers"
    exec uvicorn api.main:app \
        --host 0.0.0.0 \
        --port "${API_PORT:-8006}" \
        --workers "${WORKERS:-2}" \
        --log-level "${LOG_LEVEL:-info}" \
        --access-log
fi

================================================================================
CHESSVISION API - LOCAL DEVELOPMENT CONTEXT
================================================================================
Last Updated: April 10, 2026
Purpose: Quick reference for AI assistants and developers working on this project

================================================================================
PROJECT OVERVIEW
================================================================================

ChessVision is a FastAPI inference server that detects chess positions and
moves from a camera feed. It uses computer vision (OpenCV) and deep learning
(PyTorch / YOLO11) to output FEN/PGN game notation in real time.

Tech Stack:
- FastAPI + Uvicorn (async inference API)
- PyTorch + Ultralytics YOLO11/YOLOv8 (board corner + piece detection)
- OpenCV (homography, perspective warp, image preprocessing)
- python-chess (legal move validation, FEN/PGN generation)
- Stockfish (optional engine analysis via subprocess)
- Docker for containerization (CPU default, NVIDIA GPU optional)

================================================================================
DOCKER SETUP
================================================================================

Docker Services (defined in docker-compose.yml):
  api      - Main FastAPI inference service (port 8006)
2. nginx    - Reverse proxy (optional, enable with --profile proxy)

Container Names:
- chessvision_api
- chessvision_nginx  (optional)

Starting the Application:
  docker-compose up -d                        # Start API (detached)
  docker-compose up                           # Start with logs visible
  docker-compose up --build                   # Rebuild image and start
  docker-compose --profile proxy up -d        # Include nginx reverse proxy

Stopping the Application:
  docker-compose down                         # Stop all services
  docker-compose down -v                      # Stop and remove volumes (clears weights volume)

View Logs:
  docker-compose logs -f api                  # Follow API logs
  docker-compose logs --tail=100 api          # Last 100 lines

GPU Mode (NVIDIA):
  # Edit docker-compose.yml: uncomment the 'deploy.resources' section
  # Then rebuild specifying GPU base image:
  docker build \
    --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
    --build-arg TORCH_EXTRA=https://download.pytorch.org/whl/cu121 \
    -t chessvision:gpu .
  # Then set INFERENCE_DEVICE=cuda in .env

================================================================================
RUNNING COMMANDS INSIDE THE CONTAINER
================================================================================

General Pattern:
  docker-compose exec api [command]
  OR
  docker exec chessvision_api [command]

Common Examples:

# Interactive shell:
  docker-compose exec -it api bash

# Check which models are loaded:
  docker-compose exec api python -c "from src.piece_detector import PieceDetector; print(PieceDetector().info())"

# Run inference on a single image:
  docker-compose exec api python -c "
from src.pipeline import ChessVisionPipeline
p = ChessVisionPipeline()
result = p.run_image('/app/data/raw/test_board.jpg')
print(result.fen)
"

# Calibrate camera (interactive tool):
  docker-compose exec -it api python src/calibration.py --image /app/data/raw/calibration.jpg

# Run tests:
  docker-compose exec api pytest tests/ -v
  docker-compose exec api pytest tests/test_board_detector.py -v
  docker-compose exec api pytest tests/test_move_detector.py -v -k "test_castling"

# Start a training run (from inside container):
  docker-compose exec api python train/train_piece_detector.py

# Export PGN for a recorded game:
  docker-compose exec api python -c "
from src.game_recorder import GameRecorder
r = GameRecorder.load('/app/data/games/game_001.json')
print(r.export_pgn())
"

================================================================================
API ENDPOINTS
================================================================================

Base URL: http://localhost:8006

Health:
  GET  /health                       - Liveness check (returns {"status": "ok"})
  GET  /health/models                - Model load status + GPU/CPU info

Single Image Inference:
  POST /analyze/image                - Upload image → returns FEN + piece positions
  POST /analyze/board-only           - Upload image → returns board corners + warped image

Video / Live Stream:
  POST /analyze/video                - Upload video → returns move list + PGN
  WS   /ws/live                      - WebSocket: stream frames → receive move events

Move Detection:
  POST /moves/detect                 - Body: {prev_fen, curr_fen} → detected move
  POST /moves/validate               - Body: {fen, move_uci} → is move legal?

Game Export:
  GET  /game/{session_id}/fen        - Current FEN of a live session
  GET  /game/{session_id}/pgn        - Full PGN of a session
  GET  /game/{session_id}/lichess    - Lichess analysis URL

Calibration:
  POST /calibration/compute          - Upload calibration image → returns homography matrix
  GET  /calibration/current          - Returns current calibration.json

================================================================================
ENVIRONMENT VARIABLES
================================================================================

Location: .env (copy from .env.example)

Key Variables:
  CHESSVISION_ENV=development          # development | production
  LOG_LEVEL=debug                      # debug | info | warning | error
  API_PORT=8006
  WORKERS=2                            # uvicorn workers (production)

  INFERENCE_DEVICE=cpu                 # cpu | cuda | mps
  PIECE_CONF_THRESHOLD=0.6
  CORNER_CONF_THRESHOLD=0.4
  FRAME_CONSENSUS_COUNT=3

  MODEL_CORNER_PATH=/app/models/corner_detector/best.pt
  MODEL_PIECE_PATH=/app/models/piece_detector/best.pt
  MODEL_SQUARE_CLASSIFIER_PATH=/app/models/square_classifier/best.pt

  STOCKFISH_PATH=/usr/local/bin/stockfish   # optional
  CORS_ORIGINS=http://localhost:3000

================================================================================
PROJECT STRUCTURE
================================================================================

Root Files:
  PLAN.md                - Full system design and roadmap
  CONTEXT.md             - This file (quick reference for devs / AI)
  Dockerfile             - Python 3.11 multi-stage CPU/GPU build
  docker-compose.yml     - Single api service + optional nginx
  docker-entrypoint.sh   - Startup: validates env → starts uvicorn
  requirements.txt       - Python dependencies
  .env                   - Local environment variables (NOT in git)
  .env.example           - Template for environment variables

Source Code:
  src/
    __init__.py
    calibration.py         - Camera calibration tool (compute + save homography)
    board_detector.py      - Phase 1: YOLO corner detect → warp → 8×8 grid
    piece_detector.py      - Phase 2: YOLO piece detect on full warped board
    square_mapper.py       - IoU matching: bounding boxes → algebraic squares
    board_state.py         - FEN builder + python-chess board state
    move_detector.py       - Phase 3: state delta → legal move detection
    game_recorder.py       - PGN builder + session management + export
    pipeline.py            - Full end-to-end inference pipeline

API:
  api/
    main.py                - FastAPI app factory (routes, middleware, CORS)
    websocket_handler.py   - WebSocket live-stream endpoint
    routers/
      analyze.py           - /analyze/* endpoints
      game.py              - /game/* endpoints
      calibration.py       - /calibration/* endpoints
    schemas.py             - Pydantic request/response models

Training:
  train/
    train_corner_detector.py
    train_piece_detector.py
    train_square_classifier.py
    configs/
      corners.yaml
      pieces.yaml

Models & Data:
  models/                  - Weights (gitignored, mounted as Docker volume)
    corner_detector/best.pt
    piece_detector/best.pt
    square_classifier/best.pt
  data/
    raw/                   - Raw captured images/videos
    calibration.json        - Saved homography matrix
    games/                 - Recorded game sessions (JSON)

Tests:
  tests/
    test_board_detector.py
    test_piece_detector.py
    test_move_detector.py
    test_pipeline.py        - End-to-end integration tests
    fixtures/              - Sample images + expected FEN ground truth

================================================================================
MODEL WEIGHTS MANAGEMENT
================================================================================

Weights are stored in a named Docker volume (chessvision_model_weights) so they
survive container rebuilds without being baked into the image.

First-time setup (copy local weights into volume):
  docker run --rm \
    -v chessvision_model_weights:/models \
    -v $(pwd)/models:/src \
    alpine cp -r /src/. /models/

Check what weights are in the volume:
  docker run --rm -v chessvision_model_weights:/models alpine ls -la /models/

Download trained weights from Roboflow:
  docker-compose exec api python -c "
import roboflow
rf = roboflow.Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace().project('chess-pieces-detection')
project.version(1).download('yolov8', location='/app/models/piece_detector')
"

================================================================================
TESTING
================================================================================

Run all tests:
  docker-compose exec api pytest tests/ -v

Run specific test file:
  docker-compose exec api pytest tests/test_move_detector.py -v

Run with coverage:
  docker-compose exec api pytest tests/ --cov=src --cov-report=term-missing

Run a specific test case:
  docker-compose exec api pytest tests/test_move_detector.py::test_castling_kingside -v

Benchmark inference speed:
  docker-compose exec api python -c "
from src.pipeline import ChessVisionPipeline
import time, glob
p = ChessVisionPipeline()
imgs = glob.glob('/app/data/raw/*.jpg')[:20]
t0 = time.time()
for img in imgs: p.run_image(img)
print(f'{(time.time()-t0)/len(imgs)*1000:.1f} ms/frame')
"

================================================================================
NGINX CONFIGURATION (optional proxy profile)
================================================================================

Config location: nginx/chessvision.conf
Enable with:     docker-compose --profile proxy up -d

The nginx config proxies / → http://api:8000 and adds:
- Request size limit: 20MB (for video uploads)
- Timeouts: 120s (inference can be slow on CPU)
- WebSocket upgrade headers for /ws/

================================================================================
COMMON ISSUES
================================================================================

Models not loading:
  → Check volume mount: docker-compose exec api ls /app/models/
  → Verify MODEL_PIECE_PATH in .env matches actual file location

Slow inference (>500ms):
  → Set INFERENCE_DEVICE=cuda (requires GPU build)
  → On Apple Silicon: set INFERENCE_DEVICE=mps
  → Reduce YOLO image size: export YOLO_IMGSZ=320 (less accurate)

Board not detected:
  → Recalibrate: docker-compose exec -it api python src/calibration.py
  → Check CORNER_CONF_THRESHOLD (try lowering to 0.25)

WebSocket disconnects:
  → Increase nginx timeout: proxy_read_timeout 300s;
  → Check WORKERS count — WebSocket sessions require sticky workers (use 1 in dev)

Port already in use:
  → Change API_PORT in .env (e.g., API_PORT=8001)
  → Or: lsof -i :8000 to find and kill the blocking process

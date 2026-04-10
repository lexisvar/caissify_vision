# ChessVision — Competitive Chess Moves Detection System
> **Date:** April 2026  
> **Sources:** Roboflow Blog (chess-boards, automated-chess-game-recording), original research synthesis

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Phase 0 — Hardware & Camera Setup](#3-phase-0--hardware--camera-setup)
4. [Phase 1 — Chessboard Detection](#4-phase-1--chessboard-detection)
5. [Phase 2 — Chess Piece Recognition](#5-phase-2--chess-piece-recognition)
6. [Phase 3 — Move Detection & Game State Tracking](#6-phase-3--move-detection--game-state-tracking)
7. [Phase 4 — Output & Integration](#7-phase-4--output--integration)
8. [Dataset Strategy](#8-dataset-strategy)
9. [Model Selection & Training](#9-model-selection--training)
10. [Tech Stack](#10-tech-stack)
11. [Docker & Deployment](#11-docker--deployment)
12. [Project Structure](#12-project-structure)
13. [Evaluation Metrics & Benchmarks](#13-evaluation-metrics--benchmarks)
14. [Robustness & Edge Cases](#14-robustness--edge-cases)
15. [Roadmap & Milestones](#15-roadmap--milestones)

---

## 1. System Overview

### Goal
Build a **reliable, real-time chess game recorder** that:
- Detects the chessboard from a standard camera feed
- Recognizes all 12 piece types (6 per color) on every square
- Tracks move sequences and outputs legal game notation (FEN + PGN)
- Works across diverse boards, lighting conditions, and camera angles
- Runs on commodity hardware (laptop, Raspberry Pi, smartphone)

### Core Problems (from Roboflow research)
| Sub-problem | Challenge | Proposed Solution |
|---|---|---|
| Board detection | Perspective distortion, partial occlusion | YOLOv8 corner detection + homography |
| Piece recognition | Piece similarity, cast shadows, occlusion | YOLOv8/YOLO11 fine-tuned per class |
| Move tracking | Detecting *which* piece moved, not just current state | Frame-diff + board state delta |
| Robustness | Lighting, board styles, piece styles | Aggressive data augmentation |

---

## 2. Architecture Diagram

```
Camera Feed (video/image)
        │
        ▼
┌─────────────────────┐
│  Pre-processing      │  ← resize, denoise, normalize
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Board Detector      │  ← YOLO corner detection → homography warp
│  (Localization)      │    outputs: 640×640 bird's-eye board image
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Square Grid Builder │  ← divide warped board into 8×8 squares
│                      │    outputs: 64 square ROIs + coordinates
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Piece Detector      │  ← YOLO piece detection on full board
│  + Classifier        │    12 classes: K,Q,R,B,N,P × white/black
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Square Occupancy    │  ← IoU matching: bounding box → square
│  Mapper              │    fallback: square crop classifier (CNN)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Board State Engine  │  ← FEN generator per frame
│                      │    validates state with python-chess
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Move Detector       │  ← diff(prev_FEN, curr_FEN) → detect move
│                      │    special: castling, en passant, promotion
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  PGN Exporter        │  ← Lichess / Stockfish integration
└─────────────────────┘
```

---

## 3. Phase 0 — Hardware & Camera Setup

### Camera Options (choose one)
| Option | Resolution | FPS | Notes |
|---|---|---|---|
| Webcam (Logitech C920) | 1080p | 30 | Good starting point, cheap |
| Smartphone (top-down mount) | 4K | 60 | Best quality, use phone tripod |
| Raspberry Pi Camera v3 | 12MP | 30 | Embedded/portable deployment |
| Custom rig (Clio-style) | VGA–1080p | 30 | Smallest form factor |

### Camera Placement Rules
- **Preferred:** Directly overhead (top-down / nadir) — eliminates perspective issues for pieces
- **Acceptable:** 45–60° angle from a fixed position — requires robust homography calibration
- **Minimum height:** 40 cm above the board (avoids piece occlusion)
- **Fixed mount:** Always preferred; eliminates per-frame board re-detection overhead

### Calibration (one-time setup)
1. Place a calibration checkerboard or use the actual chess board corners as calibration targets
2. Detect 4 outer corners manually or via YOLO once
3. Compute homography matrix `H` using `cv2.findHomography()`
4. Save `H` to `calibration.json`
5. All subsequent frames are pre-warped using `H` before any ML model runs

```python
# calibration.py
import cv2, json, numpy as np

src_pts = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])  # picked manually
dst_pts = np.float32([[0,0],[640,0],[640,640],[0,640]])    # ideal square

H, _ = cv2.findHomography(src_pts, dst_pts)
json.dump(H.tolist(), open("calibration.json", "w"))
```

---

## 4. Phase 1 — Chessboard Detection

This phase outputs a **warped bird's-eye 640×640 image** of only the chessboard.

### Step 1.1 — Corner Detection Model
- **Model:** YOLOv8n or YOLO11n (fast inference, small footprint)
- **Classes:** 4 corners (top-left, top-right, bottom-right, bottom-left) **or** a single `corner` class with 4 detections sorted by position
- **Dataset:** ~200–500 images, diverse boards and lighting
- **Reported accuracy:** >99.5% on corner detection (Roboflow paper)
- **Alternative:** x-corner (inner grid intersection) detection for higher precision alignment (A1H1/Clio approach)

### Step 1.2 — Corner Ordering
```python
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]   # bottom-right (largest sum)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect
```

### Step 1.3 — Perspective Transform
```python
def warp_board(image, corners):
    ordered = order_points(corners)
    dst = np.float32([[0,0],[640,0],[640,640],[0,640]])
    H = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, H, (640, 640))
    return warped, H
```

### Step 1.4 — Square Grid Extraction
After warping, the board is a perfect 640×640 image with each square being 80×80 px.
```python
def extract_squares(warped_board):
    squares = {}
    sq_size = 640 // 8
    files = "abcdefgh"
    ranks = "87654321"
    for r_idx, rank in enumerate(ranks):
        for f_idx, file in enumerate(files):
            x1 = f_idx * sq_size
            y1 = r_idx * sq_size
            squares[f"{file}{rank}"] = warped_board[y1:y1+sq_size, x1:x1+sq_size]
    return squares
```

### Step 1.5 — Rotation Disambiguation
The board orientation (which side is white/black) must be detected once at game start:
- Use a known-piece pattern at game start (initial position is deterministic)
- Train a small classifier: `white_bottom` vs `black_bottom`
- Or allow the user to specify orientation via the UI

---

## 5. Phase 2 — Chess Piece Recognition

### Approach A: Direct YOLO Detection (Recommended for top-down camera)
- Run YOLO on the **full warped board** in one pass
- 13 classes: `empty`, `wK`, `wQ`, `wR`, `wB`, `wN`, `wP`, `bK`, `bQ`, `bR`, `bB`, `bN`, `bP`
- Use IoU matching (via `shapely`) between bounding boxes and square polygons
- For tall pieces (King, Queen), use **lower 50% of bbox** for square matching

```python
from shapely.geometry import Polygon

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou
```

### Approach B: Square Occupancy + Crop Classifier (Robust for angled cameras)
- First, classify each of the 64 squares as **occupied** or **empty** (YOLO occupancy model)
- Then, for each occupied square, run a crop classifier (ResNet18 or EfficientNet-B0) on the ROI
- More robust at low camera angles where direct detection confuses adjacent pieces

### Approach C: Hybrid (Best reliability)
1. YOLO detects pieces on full board (Approach A)
2. For low-confidence detections (<0.6), fall back to crop classifier (Approach B)
3. Apply `python-chess` legality check to filter out impossible states

### Reported performance target
| Metric | Target |
|---|---|
| Piece detection mAP@0.5 | >95% |
| Per-square classification accuracy | >99% |
| Full-board FEN accuracy | >98% |

---

## 6. Phase 3 — Move Detection & Game State Tracking

This is the most critical phase for **competitive move detection**.

### Step 3.1 — Frame-Level FEN Generation
Run the full pipeline on every N-th frame (e.g., every 5 frames at 30fps = 6 Hz):
```
FEN_sequence = [FEN_t0, FEN_t1, FEN_t2, ...]
```

### Step 3.2 — State Delta (Move Detection)
```python
import chess

def detect_move(prev_fen: str, curr_fen: str) -> chess.Move | None:
    prev_board = chess.Board(prev_fen)
    curr_board = chess.Board(curr_fen)
    
    # Find squares that changed
    changed = []
    for sq in chess.SQUARES:
        if prev_board.piece_at(sq) != curr_board.piece_at(sq):
            changed.append(sq)
    
    # Generate all legal moves from prev state
    for move in prev_board.legal_moves:
        test_board = prev_board.copy()
        test_board.push(move)
        # Normalize FEN for comparison (ignore half-move clock, en passant)
        if test_board.board_fen() == curr_board.board_fen():
            return move
    return None  # ambiguous or detection error
```

### Step 3.3 — Legality Validation
All detected moves are validated against `python-chess` legal move generation:
- Reject board states that are physically illegal
- If a transition is ambiguous (multiple legal moves match), hold state and wait for more frames
- If a transition is impossible (e.g., piece teleported), flag as detection error → use interpolation or last known good state

### Step 3.4 — Special Move Handling
| Move Type | Detection Strategy |
|---|---|
| Castling | King moves 2 squares + rook teleports → matched by `python-chess` legal moves |
| En Passant | Captured pawn disappears from a different square → delta analysis |
| Promotion | Pawn on rank 8 becomes another piece → infer from piece class change |
| Capture | One piece disappears, another moves to its square |

### Step 3.5 — Temporal Smoothing
To avoid spurious move detections from transient frames (hand moving a piece):
- Require **3 consecutive identical FEN readings** before committing a new move
- During "hand in frame" period, suppress detection (detect hand presence with simple skin-color or silhouette detector)
- Use a confidence threshold: only accept moves if piece confidence > 0.7

---

## 7. Phase 4 — Output & Integration

### FEN Output
- Standard FEN string generated after every committed move
- Example: `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1`

### PGN Export
```python
import chess.pgn, datetime

def export_pgn(moves: list[chess.Move], event="ChessVision Game") -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["Date"] = datetime.date.today().isoformat()
    node = game
    board = chess.Board()
    for move in moves:
        node = node.add_variation(move)
        board.push(move)
    return str(game)
```

### External Integrations
| Integration | Purpose | Library/API |
|---|---|---|
| Lichess | Analysis, share, explore opening | `https://lichess.org/analysis/<FEN>` |
| Stockfish | Engine analysis, best move suggestions | `python-chess` + `stockfish` binary |
| Chess.com API | Import game | `chessdotcom` Python SDK |

---

## 8. Dataset Strategy

### A) Chessboard Corner Dataset
- **Size:** 500–1000 images (with augmentation → 3000+)
- **Diversity:** Wood, silicone, digital-print boards; indoor/outdoor lighting; partial occlusion
- **Annotation:** 4 corner keypoints per image (or bounding boxes for corner regions)
- **Sources:** 
  - Roboflow Universe: `chess-board-detection` datasets
  - Self-captured: 100+ photos from multiple angles/boards
  - Synthetic: render boards in Blender with random textures/lighting

### B) Chess Piece Dataset
- **Size:** 3000–5000 annotated pieces (raw), augmented to 15000+
- **Classes:** 12 piece types + empty square
- **Diversity:** Wood, plastic, Staunton, non-standard sets; shadows, hand occlusion
- **Sources:**
  - [Roboflow Universe Chess datasets](https://universe.roboflow.com/search?q=chess%20pieces)
  - [Chess dataset by Nikzayn](https://public.roboflow.com/object-detection/chess-pieces) (publicly available)
  - Self-capture during real games
  - Synthetic augmentation (random backgrounds, lighting, rotation ±15°)

### C) Augmentation Pipeline
```python
# Using albumentations
import albumentations as A

train_transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, p=0.4),
    A.GaussNoise(p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.RandomShadow(p=0.3),
    A.Perspective(scale=(0.02, 0.08), p=0.3),  # simulate slight angle changes
])
```

---

## 9. Model Selection & Training

### Model Comparison
| Model | Speed (ms/frame) | mAP@0.5 | Size | Use Case |
|---|---|---|---|---|
| YOLOv8n | ~5ms | ~90% | 6MB | Edge / Raspberry Pi |
| YOLOv8s | ~10ms | ~93% | 22MB | Laptop real-time |
| YOLO11m | ~15ms | ~95% | 40MB | High accuracy |
| RF-DETR | ~30ms | ~97% | 120MB | Best accuracy, GPU |

### Recommended Training Recipe
```yaml
# ultralytics training config
model: yolo11s.pt       # pretrained on COCO
data: chess_pieces.yaml
imgsz: 640
epochs: 150
batch: 16
lr0: 0.01
lrf: 0.01
mosaic: 1.0
mixup: 0.15
copy_paste: 0.3         # helps with crowded boards
close_mosaic: 10
augment: true
```

### Two-Stage Training
1. **Stage 1:** Fine-tune on large public chess dataset (~5k images)
2. **Stage 2:** Fine-tune further on self-collected data specific to your board/camera setup

### Occupancy Model (separate, lightweight)
- MobileNetV3-Small or EfficientNet-B0 as crop classifier
- Input: 80×80 RGB square crop
- Output: 13 classes (empty + 12 piece types)
- Training: ~50k square crops extracted from annotated full-board images
- This acts as a **verification layer** for YOLO detections

---

## 10. Tech Stack

```
chessvision/
├── Core ML
│   ├── ultralytics        # YOLOv8 / YOLO11
│   ├── torch              # PyTorch backend
│   ├── torchvision        # crop classifier
│   └── albumentations     # augmentation
│
├── Computer Vision
│   ├── opencv-python      # image processing, homography, warping
│   ├── numpy              # array ops
│   └── shapely            # IoU geometry calculations
│
├── Chess Logic
│   ├── python-chess       # board state, legal moves, PGN/FEN
│   └── stockfish          # optional: engine analysis
│
├── API / UI
│   ├── fastapi            # REST API server
│   ├── streamlit          # quick demo UI
│   └── websockets         # live game streaming
│
└── Infra / Tools
    ├── roboflow            # dataset management + annotation
    ├── wandb               # experiment tracking
    └── pytest              # unit + integration tests
```

### Installation (local)
```bash
pip install ultralytics opencv-python python-chess stockfish \
            shapely albumentations fastapi streamlit wandb \
            torch torchvision roboflow
```

---

## 11. Docker & Deployment

ChessVision is packaged as a single **Docker container** running a FastAPI/Uvicorn inference server. No database or message broker is needed — the system is stateless per request and session state lives in memory.

### Files
| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage Python 3.11 build (CPU default, CUDA optional via build-arg) |
| `docker-compose.yml` | `api` service + optional `nginx` reverse proxy (profile: `proxy`) |
| `docker-entrypoint.sh` | Validates env vars, checks model weights, starts uvicorn |
| `.env.example` | Template — copy to `.env` before first run |
| `nginx/chessvision.conf` | Nginx config (20 MB upload limit, WebSocket upgrade, 120s timeout) |

### Quick Start
```bash
# 1. Copy and edit environment variables
cp .env.example .env

# 2. Build and start the API
docker-compose up --build

# 3. Verify it's running
curl http://localhost:8000/health
```

### Common Commands
```bash
# Rebuild after code changes
docker-compose up --build

# Follow logs
docker-compose logs -f api

# Run tests inside the container
docker-compose exec api pytest tests/ -v

# Interactive shell
docker-compose exec -it api bash

# Run inference on a local image
docker-compose exec api python -c "
from src.pipeline import ChessVisionPipeline
p = ChessVisionPipeline()
r = p.run_image('/app/data/raw/test_board.jpg')
print(r.fen)
"
```

### GPU Build (NVIDIA)
```bash
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
  --build-arg TORCH_EXTRA=https://download.pytorch.org/whl/cu121 \
  -t chessvision:gpu .
```
Then set `INFERENCE_DEVICE=cuda` in `.env` and uncomment the `deploy.resources` block in `docker-compose.yml`.

### Apple Silicon (MPS)
No special build needed — use the default CPU image and set `INFERENCE_DEVICE=mps` in `.env`.

### Model Weights Volume
Weights are **not baked into the image** — they live in a named Docker volume (`chessvision_model_weights`) so rebuilds are fast and weights survive container restarts.

```bash
# First-time: copy local weights into the volume
docker run --rm \
  -v chessvision_model_weights:/models \
  -v $(pwd)/models:/src \
  alpine cp -r /src/. /models/
```

---

## 12. Project Structure

```
chessvision/
├── PLAN.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                      # raw captured images/videos
│   ├── annotated/                # Roboflow export (YOLO format)
│   ├── augmented/                # post-augmentation
│   └── calibration.json          # homography matrix
│
├── models/
│   ├── corner_detector/          # YOLOv8 corner detection weights
│   │   └── best.pt
│   ├── piece_detector/           # YOLOv8 piece detection weights
│   │   └── best.pt
│   └── square_classifier/        # EfficientNet square crop classifier
│       └── best.pt
│
├── src/
│   ├── __init__.py
│   ├── calibration.py            # camera calibration tool
│   ├── board_detector.py         # Phase 1: board localization + warp
│   ├── piece_detector.py         # Phase 2: YOLO + crop classifier
│   ├── square_mapper.py          # IoU square-piece mapping
│   ├── board_state.py            # FEN generation + python-chess integration
│   ├── move_detector.py          # Phase 3: state delta + legality check
│   ├── game_recorder.py          # PGN builder + export
│   └── pipeline.py               # full end-to-end inference pipeline
│
├── train/
│   ├── train_corner_detector.py
│   ├── train_piece_detector.py
│   ├── train_square_classifier.py
│   └── configs/
│       ├── corners.yaml
│       └── pieces.yaml
│
├── api/
│   ├── main.py                   # FastAPI server
│   └── websocket_handler.py      # live stream endpoint
│
├── ui/
│   └── app.py                    # Streamlit demo
│
└── tests/
    ├── test_board_detector.py
    ├── test_piece_detector.py
    ├── test_move_detector.py
    └── fixtures/                 # sample images + expected FENs
```

---

## 13. Evaluation Metrics & Benchmarks

### Per-Component Metrics
| Component | Metric | Minimum Target | Competitive Target |
|---|---|---|---|
| Corner detection | mAP@0.5 | 95% | 99.5% |
| Piece detection | mAP@0.5 | 90% | 95% |
| Per-square classification | Accuracy | 97% | 99.5% |
| Full-board FEN accuracy | Exact match | 90% | 98% |
| Move detection | Precision/Recall | 95%/95% | 99%/99% |
| End-to-end latency | ms/frame | <200ms | <50ms |

### Test Sets
- **Standard test:** Initial position + 20 common openings (fixed camera, controlled lighting)
- **Stress test:** Endgame positions, promotions, castling, en passant
- **Robustness test:** Variable lighting, shadows, different boards, partial occlusion by hands
- **Real-game test:** Record 10 full OTB games and compare against DGT board ground truth

---

## 14. Robustness & Edge Cases

### Lighting Variations
- Training augmentation: `RandomBrightnessContrast`, `RandomShadow`, `CLAHE`
- Preprocessing: CLAHE normalization before inference
- HDR exposure bracketing if using dedicated camera

### Camera Angle Variations
- Fixed-angle systems: pre-calibrated homography (fastest, most robust)
- Variable-angle: per-frame corner detection + live homography computation
- Handle up to ±30° rotation and ±20° tilt via augmentation

### Board & Piece Style Variations
- Train on minimum 5 different board styles (wood, green plastic, tournament)
- Train on minimum 3 piece styles (Staunton classic, modern plastic, Dubrovnik)
- Use domain randomization: synthetic backgrounds, random textures

### Occlusion (Hand Covering Pieces)
- **Detection:** YOLOv8 hand detector or simple temporal frame delta
- **Strategy:** Hold last committed board state; resume detection when hand leaves
- **Timeout:** If hand in frame >5 seconds, trigger re-scan of full board state

### Clock-related Ambiguity
- When a player presses the clock, the board state should already be stable
- Use temporal majority voting (5 frames agree → commit state)

### Special Situations
| Situation | Handling |
|---|---|
| Pieces knocked over | Detected as empty square; human correction prompt |
| Pawn promotion | Detect disappearing pawn + new piece; prompt player to confirm piece type |
| Draw offer tokens | Ignore non-piece objects via confidence thresholding |
| Very similar pieces (e.g., Bishops) | Distinguish by color via HSV analysis on piece top |

---

## 15. Roadmap & Milestones

### Milestone 1 — Foundation ✅ (Weeks 1–2)
- [x] Set up project structure (`src/`, `api/`, `train/`, `tests/`, `data/`)
- [x] Configure Python environment with all dependencies (Docker CPU venv, CPU-only torch wheels)
- [x] Docker multi-stage image built and container running on port 8006 (`/health` → 200 OK)
- [x] Implement camera calibration tool (`src/calibration.py` — homography H, CLI + API endpoint)
- [x] Implement homography warp and square extraction pipeline (`src/board_detector.py`)
- [ ] Set up Roboflow account and create board + piece datasets

### Milestone 2 — Board Detection (Weeks 3–4)
- [x] Implement `board_detector.py` with full warp pipeline (YOLO corners → `warpPerspective` → 64 squares)
- [ ] Annotate 300+ board corner images in Roboflow
- [ ] Train YOLOv8n corner detector (`train/train_corner_detector.py` is ready)
- [ ] Place `best.pt` at `models/corner_detector/best.pt`
- [ ] Validate: >99% corner detection on test set

### Milestone 3 — Piece Recognition (Weeks 5–7)
- [x] Implement `piece_detector.py` (YOLO full-board) + `square_mapper.py` (IoU matching)
- [ ] Collect/annotate 3000+ piece images from Roboflow Universe + self-capture
- [ ] Apply augmentation pipeline (`albumentations` config in `train/configs/pieces.yaml`)
- [ ] Train YOLO11s piece detector (12 classes)
- [ ] Train EfficientNet-B0 square classifier (backup model)
- [ ] Place weights at `models/piece_detector/best.pt`
- [ ] Validate: >95% mAP, >99% per-square accuracy

### Milestone 4 — FEN Generation (Week 8)
- [x] Implement `board_state.py` with FEN builder (`_compress_row`, castling rights, side-to-move)
- [x] Integrate `python-chess` for legality checking (`Board.is_valid()`)
- [ ] Validate: >98% exact FEN match on 50 test positions (requires trained models)

### Milestone 5 — Move Detection (Weeks 9–10)
- [x] Implement state delta algorithm (`src/move_detector.py`)
- [x] Handle all special moves (castling, en passant, promotion) via `python-chess` legal move enumeration
- [x] Implement temporal smoothing (3-frame consensus deque, configurable `FRAME_CONSENSUS_COUNT`)
- [x] Hand occlusion handling (consensus flush on `reset()`)
- [ ] Validate: >99% move detection precision/recall on 10 test games

### Milestone 6 — Full Pipeline & API (Weeks 11–12)
- [x] Integrate all modules into `pipeline.py` (`ChessVisionPipeline.run_image()` + `process_frame()`)
- [x] Build FastAPI REST server (`api/`) + WebSocket live stream (`/ws/live`)
- [x] Stockfish analysis integration (`src/game_recorder.py` — `analyse_with_stockfish()`)
- [x] Dockerized and deployed: `docker compose up --build` → container healthy on port 8006
- [ ] Build Streamlit demo UI
- [ ] End-to-end latency benchmark: <50ms/frame on GPU, <200ms on CPU

### Milestone 7 — Hardening (Weeks 13–14)
- [x] Test suite scaffolded: `tests/test_board_detector.py`, `test_board_state.py`, `test_move_detector.py`, `test_game_recorder.py`
- [ ] Run tests with actual model weights: `docker compose exec api pytest tests/ -v`
- [ ] Stress test on 20+ different board/piece combinations
- [ ] Fix systematic errors from stress test
- [ ] Document all APIs (Swagger auto-docs live at `/docs`)
- [ ] Record README video demo

---

## Key References & Resources

| Resource | Link |
|---|---|
| Roboflow: Chess Boards | https://blog.roboflow.com/chess-boards/ |
| Roboflow: Automated Game Recording | https://blog.roboflow.com/automated-chess-game-recording-computer-vision/ |
| Real-Life Chess Vision (GitHub) | https://github.com/shainisan/real-life-chess-vision |
| ChessboardDetect (x-corners) | https://github.com/Elucidation/ChessboardDetect |
| Roboflow Chess Universe | https://universe.roboflow.com/search?q=chess |
| python-chess docs | https://python-chess.readthedocs.io/ |
| Ultralytics YOLO11 | https://docs.ultralytics.com/ |
| Stockfish binary | https://stockfishchess.org/download/ |
| Lichess FEN analysis | https://lichess.org/analysis/ |

---

*This plan synthesizes the YOLOv8 corner detection approach (Shai Nisan / Roboflow), the x-corner homography calibration system (A1H1 / Clio), and additional best practices for temporal consistency, legality validation, and robustness engineering.*

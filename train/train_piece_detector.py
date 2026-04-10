"""
train/train_piece_detector.py
==============================
Fine-tune a YOLO11s / YOLOv8s model to detect and classify chess pieces.

12 piece classes: wK, wQ, wR, wB, wN, wP, bK, bQ, bR, bB, bN, bP

Usage:
    # Download datasets + train in one command:
    python train/train_piece_detector.py --download

    # Train only (datasets already present):
    python train/train_piece_detector.py

    # Full options:
    python train/train_piece_detector.py --download --model yolo11s.pt --epochs 150 --device mps
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

# Must be set before any ultralytics import so the base model is downloaded
# to /tmp (writable by the non-root container user) instead of the cwd.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
_WEIGHTS_DIR = Path("/tmp/ultralytics_weights")
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/runs/train_pieces.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def train(
    model: str = "yolo11s.pt",
    data: str = "train/configs/pieces.yaml",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    project: str = "/app/runs/pieces",
    name: str = "piece_detector",
) -> None:
    from ultralytics import YOLO  # type: ignore
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"weights_dir": str(_WEIGHTS_DIR), "runs_dir": "/app/runs"})

    # Resolve to absolute path so ultralytics downloads into /tmp, not the
    # read-only cwd (/app owned by root).
    model_abs = str(_WEIGHTS_DIR / model)
    logger.info("Loading base model: %s", model_abs)
    m = YOLO(model_abs)

    logger.info("Starting piece detector training (%d epochs)...", epochs)
    m.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project=project,
        name=name,
        # Augmentation — critical for diverse pieces / lighting
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15.0,
        perspective=0.0005,
        fliplr=0.0,       # chess pieces are NOT left-right symmetric
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,   # helps with crowded boards
        close_mosaic=10,
        # Class weights to compensate rare classes (e.g. promotion pieces)
        # cls=1.5,
        plots=True,
        save=True,
    )

    # Copy best weights to models directory
    best = Path(project) / name / "weights" / "best.pt"
    dest = Path("models/piece_detector/best.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, dest)
        logger.info("Best weights saved to %s", dest)
    else:
        logger.warning("Training finished but best.pt not found at %s", best)


if __name__ == "__main__":
    # Load .env so ROBOFLOW_API_KEY is available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Train ChessVision piece detector")
    parser.add_argument("--download", action="store_true",
                        help="Download & merge Roboflow datasets before training")
    parser.add_argument("--dest", default="data/annotated/pieces",
                        help="Merged dataset destination (used with --download)")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    if args.download:
        from pathlib import Path as _Path
        from train.download_datasets import download_and_merge  # type: ignore
        logger.info("Downloading and merging datasets into %s …", args.dest)
        download_and_merge(_Path(args.dest))

    train(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

"""
train/train_piece_detector.py
==============================
Fine-tune a YOLO11s / YOLOv8s model to detect and classify chess pieces.

Two-stage training strategy:
  Stage 1: Pre-train on large public Roboflow Universe chess dataset
  Stage 2: Fine-tune on self-collected data from your specific board/camera

12 piece classes: wK, wQ, wR, wB, wN, wP, bK, bQ, bR, bB, bN, bP

Usage:
    python train/train_piece_detector.py
    python train/train_piece_detector.py --model yolo11s.pt --epochs 150
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(
    model: str = "yolo11s.pt",
    data: str = "train/configs/pieces.yaml",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    project: str = "runs/pieces",
    name: str = "piece_detector",
) -> None:
    from ultralytics import YOLO  # type: ignore

    logger.info("Loading base model: %s", model)
    m = YOLO(model)

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
    parser = argparse.ArgumentParser(description="Train ChessVision piece detector")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    train(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

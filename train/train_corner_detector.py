"""
train/train_corner_detector.py
===============================
Fine-tune a YOLOv8n / YOLO11n model to detect chess board corners.

Steps:
  1. Optionally download dataset from Roboflow Universe
  2. Apply augmentation
  3. Train for 150 epochs with aggressive augmentation settings
  4. Validate and export best.pt to models/corner_detector/

Usage:
    python train/train_corner_detector.py
    python train/train_corner_detector.py --epochs 200 --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(
    model: str = "yolo11n.pt",
    data: str = "train/configs/corners.yaml",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    project: str = "runs/corners",
    name: str = "corner_detector",
) -> None:
    from ultralytics import YOLO  # type: ignore

    logger.info("Loading base model: %s", model)
    m = YOLO(model)

    logger.info("Starting corner detector training (%d epochs)...", epochs)
    m.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project=project,
        name=name,
        # Augmentation — key for diverse board/lighting robustness
        degrees=30.0,         # rotation up to ±30°
        perspective=0.001,    # slight perspective distortion
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        # Logging
        plots=True,
        save=True,
    )

    # Copy best weights to models directory
    best = Path(project) / name / "weights" / "best.pt"
    dest = Path("models/corner_detector/best.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, dest)
        logger.info("Best weights saved to %s", dest)
    else:
        logger.warning("Training finished but best.pt not found at %s", best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChessVision corner detector")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model (default: yolo11n.pt)")
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

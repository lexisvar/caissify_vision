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
import os
import shutil
import tempfile
from pathlib import Path

# Must be set before any ultralytics import so the base model is downloaded
# to /tmp (writable by the non-root container user) instead of the cwd.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
_WEIGHTS_DIR = Path("/tmp/ultralytics_weights")
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


_ROOT = Path(os.environ.get("CHESSVISION_ROOT", Path(__file__).parent.parent))


def train(
    model: str = "yolo11n.pt",
    data: str = "train/configs/corners.yaml",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    project: str = "",
    name: str = "corner_detector",
) -> None:
    from ultralytics import YOLO  # type: ignore
    from ultralytics.utils import SETTINGS

    runs_dir = _ROOT / "runs"
    if not project:
        project = str(runs_dir / "corners")

    # Set up log file
    log_path = Path(project).parent / "train_corners.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(str(log_path), mode="w"))
    logger.info("Log: %s", log_path)

    SETTINGS.update({"weights_dir": str(_WEIGHTS_DIR), "runs_dir": str(runs_dir)})

    # Resolve data YAML: replace the 'path:' entry so it points to CHESSVISION_ROOT,
    # not the Docker-only /app path.
    import yaml  # noqa: PLC0415
    data_yaml_src = _ROOT / data
    with open(data_yaml_src) as fh:
        data_cfg = yaml.safe_load(fh)
    data_cfg["path"] = str(_ROOT / "data" / "annotated" / "corners")
    _tmp_yaml = tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, prefix="corners_"
    )
    yaml.dump(data_cfg, _tmp_yaml)
    _tmp_yaml.flush()
    data = _tmp_yaml.name
    logger.info("Resolved data YAML: %s (path=%s)", data, data_cfg["path"])

    # Resolve to absolute path so ultralytics downloads into /tmp, not the
    # read-only cwd (/app owned by root).
    model_abs = str(_WEIGHTS_DIR / model)
    logger.info("Loading base model: %s", model_abs)
    m = YOLO(model_abs)

    # ── progress callback: one clean line per epoch to stdout + log ──────────
    def _on_epoch_end(trainer) -> None:
        e      = trainer.epoch + 1
        total  = trainer.epochs
        pct    = e / total * 100
        bar    = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        m_loss = trainer.loss.item() if hasattr(trainer.loss, "item") else float(trainer.loss)
        lr     = trainer.optimizer.param_groups[0]["lr"]
        line   = (
            f"  [{bar}] {pct:5.1f}%  epoch {e:>3}/{total}"
            f"  loss={m_loss:.4f}  lr={lr:.2e}"
        )
        print(line, flush=True)
        logger.info(line)

    m.add_callback("on_train_epoch_end", _on_epoch_end)

    logger.info("Starting corner detector training (%d epochs)...", epochs)
    logger.info("Log file: %s", log_path)
    m.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project=project,
        name=name,
        verbose=False,        # suppress Ultralytics per-batch spam; use our callback
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
    dest = _ROOT / "models" / "corner_detector" / "best.pt"
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

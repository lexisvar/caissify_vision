"""
train/download_datasets.py
===========================
Downloads the two Roboflow Universe chess-piece datasets, remaps their class
labels to the canonical ChessVision convention, and merges everything into a
single dataset under  data/annotated/pieces/.

Datasets used
─────────────
  1. chess-lzrhd / chess-vision          (193 images)
  2. roboflow-100 / chess-pieces-mjzgj   (289 images, version 2)

Canonical class order (written to pieces.yaml)
───────────────────────────────────────────────
  0  wK   White King
  1  wQ   White Queen
  2  wR   White Rook
  3  wB   White Bishop
  4  wN   White Knight
  5  wP   White Pawn
  6  bK   Black King
  7  bQ   Black Queen
  8  bR   Black Rook
  9  bB   Black Bishop
  10 bN   Black Knight
  11 bP   Black Pawn

Usage
─────
  python train/download_datasets.py
  python train/download_datasets.py --dest data/annotated/pieces --skip-download
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml  # PyYAML, installed via ultralytics dep

# ── canonical class mapping ────────────────────────────────────────────────────
#  Every alias that may appear in a downloaded data.yaml is mapped to our index.
CANONICAL_NAMES: list[str] = [
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
]

# All known aliases in public datasets → (colour prefix, piece letter)
_ALIAS_MAP: dict[str, str] = {
    # long hyphenated  (chess-lzrhd, roboflow-100)
    "white-king":   "wK", "white-queen":   "wQ", "white-rook":   "wR",
    "white-bishop": "wB", "white-knight":  "wN", "white-pawn":   "wP",
    "black-king":   "bK", "black-queen":   "bQ", "black-rook":   "bR",
    "black-bishop": "bB", "black-knight":  "bN", "black-pawn":   "bP",
    # short canonical
    "wk": "wK", "wq": "wQ", "wr": "wR", "wb": "wB", "wn": "wN", "wp": "wP",
    "bk": "bK", "bq": "bQ", "br": "bR", "bb": "bB", "bn": "bN", "bp": "bP",
    # single-letter variants  (some datasets use "K", "Q" …)
    "k": "wK", "q": "wQ", "r": "wR", "b": "wB", "n": "wN", "p": "wP",
}
CANONICAL_INDEX: dict[str, int] = {n: i for i, n in enumerate(CANONICAL_NAMES)}


def _alias_to_canonical(raw: str) -> str | None:
    """Normalise a raw class name from a downloaded data.yaml to canonical."""
    return _ALIAS_MAP.get(raw.strip().lower())


def _build_index_remap(src_names: list[str]) -> dict[int, int] | None:
    """
    Return a dict  {src_class_index: canonical_index}  for all classes in
    src_names.  Returns None if any class cannot be mapped.
    """
    remap: dict[int, int] = {}
    for i, raw in enumerate(src_names):
        canon = _alias_to_canonical(raw)
        if canon is None:
            print(f"  [WARN] Cannot map class '{raw}' — will skip annotations with this label.")
            continue
        remap[i] = CANONICAL_INDEX[canon]
    return remap


def _remap_labels(src_labels_dir: Path, dst_labels_dir: Path,
                  remap: dict[int, int]) -> None:
    """Copy & remap class indices in every YOLO .txt label file."""
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for txt in src_labels_dir.glob("*.txt"):
        lines_out: list[str] = []
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            src_cls = int(parts[0])
            if src_cls not in remap:
                continue                    # skip unknown class
            new_cls = remap[src_cls]
            lines_out.append(f"{new_cls} {' '.join(parts[1:])}")
        (dst_labels_dir / txt.name).write_text("\n".join(lines_out) + "\n")
        count += 1
    print(f"    Remapped {count} label files → {dst_labels_dir}")


def _copy_images(src_images_dir: Path, dst_images_dir: Path, prefix: str) -> None:
    """Copy images, adding a prefix to avoid filename collisions."""
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src_images_dir.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            shutil.copy(img, dst_images_dir / f"{prefix}_{img.name}")
            count += 1
    print(f"    Copied {count} images → {dst_images_dir}")


def _merge_dataset(src_root: Path, dest_root: Path, prefix: str) -> None:
    """
    Merge one downloaded Roboflow dataset (src_root) into dest_root.
    Reads src_root/data.yaml to learn the source class list.
    """
    data_yaml_path = src_root / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {src_root}")

    with open(data_yaml_path) as f:
        meta = yaml.safe_load(f)

    src_names: list[str] = meta.get("names", [])
    print(f"  Source classes ({len(src_names)}): {src_names}")

    remap = _build_index_remap(src_names)
    if not remap:
        raise RuntimeError(f"No class mappings found for dataset {prefix}")

    for split in ("train", "valid", "test"):
        src_images = src_root / split / "images"
        src_labels = src_root / split / "labels"
        if not src_images.exists():
            print(f"  Split '{split}' not found — skipping.")
            continue

        # Roboflow uses "valid", YOLO convention uses "val"
        dst_split = "val" if split == "valid" else split
        _copy_images(src_images, dest_root / "images" / dst_split, prefix)
        _remap_labels(src_labels, dest_root / "labels" / dst_split, remap)


def _merge_corners_dataset(src_root: Path, dest_root: Path, prefix: str) -> None:
    """
    Merge one Roboflow corner dataset into dest_root.
    Only the 'corner' class is kept (class index 0 in chess-corner-pwmrm).
    The 'r' class is discarded.
    """
    data_yaml_path = src_root / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {src_root}")

    with open(data_yaml_path) as f:
        meta = yaml.safe_load(f)

    src_names: list[str] = meta.get("names", [])
    print(f"  Source classes ({len(src_names)}): {src_names}")

    # Find which source index maps to 'corner'
    corner_src_idx: int | None = None
    for i, name in enumerate(src_names):
        if name.strip().lower() == "corner":
            corner_src_idx = i
            break
    if corner_src_idx is None:
        raise RuntimeError(f"No 'corner' class found in {src_root}/data.yaml: {src_names}")

    for split in ("train", "valid", "test"):
        src_images = src_root / split / "images"
        src_labels = src_root / split / "labels"
        if not src_images.exists():
            print(f"  Split '{split}' not found — skipping.")
            continue
        dst_split = "val" if split == "valid" else split
        _copy_images(src_images, dest_root / "images" / dst_split, prefix)
        # Remap: keep only corner_src_idx, map it to class 0
        remap = {corner_src_idx: 0}
        _remap_labels(src_labels, dest_root / "labels" / dst_split, remap)


def download_and_merge(dest: Path, skip_download: bool = False,
                       mode: str = "pieces") -> None:
    """Main entry point: download datasets and merge into dest/."""
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key and not skip_download:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY is not set. "
            "Export it or add it to your .env file."
        )

    tmp = Path("data/tmp_rf_downloads")
    tmp.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    PIECE_DATASETS = [
        # (workspace, project, version, local_dir_name, merge_prefix)
        ("chess-piece-detection-lydqy", "chess-piece-detection-5ipnt", 3, "chess_piece_det", "cpd3"),
        ("roboflow-100",               "chess-pieces-mjzgj",           2, "chess_pieces_rf", "rf2"),
    ]
    CORNER_DATASETS = [
        ("chess-ai-0uukd", "chess-corner-pwmrm", 4, "chess_corners", "cc4"),
    ]

    datasets_to_download = CORNER_DATASETS if mode == "corners" else PIECE_DATASETS

    if not skip_download:
        from roboflow import Roboflow  # type: ignore
        rf = Roboflow(api_key=api_key)

        for workspace, project_name, version, dir_name, _ in datasets_to_download:
            out_dir = tmp / dir_name
            if out_dir.exists():
                print(f"[SKIP] {workspace}/{project_name} already downloaded at {out_dir}")
                continue
            print(f"\n[DOWNLOAD] {workspace}/{project_name} v{version} …")
            proj = rf.workspace(workspace).project(project_name)
            ds = proj.version(version).download("yolov11", location=str(tmp / dir_name))
            print(f"  → {ds.location}")
    else:
        print("[skip-download] Using cached data in", tmp)

    # ── merge ────────────────────────────────────────────────────────────────
    for _, _, _, dir_name, prefix in datasets_to_download:
        src = tmp / dir_name
        if not src.exists():
            print(f"[WARN] {src} not found — skipping merge for this dataset.")
            continue
        print(f"\n[MERGE] {dir_name} (prefix={prefix}) → {dest}")
        if mode == "corners":
            _merge_corners_dataset(src, dest, prefix)
        else:
            _merge_dataset(src, dest, prefix)

    # ── write unified data.yaml ───────────────────────────────────────────────
    if mode == "corners":
        yaml_path = Path("train/configs/corners.yaml")
        data = {
            "path": str(dest.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc": 1,
            "names": ["corner"],
        }
    else:
        yaml_path = Path("train/configs/pieces.yaml")
        data = {
            "path": str(dest.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc": len(CANONICAL_NAMES),
            "names": CANONICAL_NAMES,
        }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\n[DONE] Wrote {yaml_path}")
    print(f"       Merged dataset at {dest}")

    # Stats
    for split in ("train", "val", "test"):
        imgs = list((dest / "images" / split).glob("*")) if (dest / "images" / split).exists() else []
        print(f"       {split:6s}: {len(imgs)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & merge Roboflow chess datasets")
    parser.add_argument("--mode", choices=["pieces", "corners"], default="pieces",
                        help="Which dataset to download: pieces (default) or corners")
    parser.add_argument("--dest", default=None,
                        help="Output directory (default: data/annotated/<mode>)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Roboflow download, use cached data in data/tmp_rf_downloads/")
    args = parser.parse_args()

    dest = Path(args.dest) if args.dest else Path(f"data/annotated/{args.mode}")

    # Load .env if dotenv is available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    download_and_merge(dest, skip_download=args.skip_download, mode=args.mode)

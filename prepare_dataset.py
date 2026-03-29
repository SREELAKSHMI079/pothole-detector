"""
prepare_dataset.py — Download RDD2022 India subset and convert to YOLO format.

RDD2022 paper: https://arxiv.org/abs/2209.08538
Download page: https://github.com/sekilab/RoadDamageDetector

Class mapping (RDD2022 → this project):
    D00 — Alligator crack
    D10 — Longitudinal crack
    D20 — Transverse crack
    D40 — Pothole  ← primary target

Usage:
    python scripts/prepare_dataset.py --country India
"""

import os
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# All RDD2022 classes (YOLO class index must be consistent)
CLASS_MAP = {
    "D00": 0,  # Alligator / fatigue crack
    "D10": 1,  # Longitudinal crack
    "D20": 2,  # Transverse crack
    "D40": 3,  # Pothole
}

CLASS_NAMES = list(CLASS_MAP.keys())


def pascal_to_yolo(size, box):
    """Convert Pascal VOC bbox to YOLO normalised format."""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 * dw
    y = (box[2] + box[3]) / 2.0 * dh
    w = (box[1] - box[0]) * dw
    h = (box[3] - box[2]) * dh
    return x, y, w, h


def convert_annotation(xml_path: Path, label_out: Path):
    """Parse one Pascal VOC XML and write YOLO .txt label."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        xmax = float(bb.find("xmax").text)
        ymin = float(bb.find("ymin").text)
        ymax = float(bb.find("ymax").text)
        x, y, w, h = pascal_to_yolo((W, H), (xmin, xmax, ymin, ymax))
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    label_out.write_text("\n".join(lines))


def prepare(rdd_root: str, out_root: str = "data", country: str = "India", val_split: float = 0.2):
    rdd_root = Path(rdd_root)
    out_root = Path(out_root)

    country_dir = rdd_root / country
    img_dir = country_dir / "train" / "images"
    ann_dir = country_dir / "train" / "annotations" / "xmls"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Expected images at {img_dir}\n"
            "Download RDD2022 from https://github.com/sekilab/RoadDamageDetector"
        )

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))
    print(f"[INFO] Found {len(images)} images for country={country}")

    split_idx = int(len(images) * (1 - val_split))
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    for split, imgs in splits.items():
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(imgs, desc=f"Processing {split}"):
            shutil.copy(img_path, out_root / split / "images" / img_path.name)
            xml_path = ann_dir / (img_path.stem + ".xml")
            lbl_path = out_root / split / "labels" / (img_path.stem + ".txt")
            if xml_path.exists():
                convert_annotation(xml_path, lbl_path)
            else:
                lbl_path.write_text("")  # empty label for unannotated images

    # Write dataset YAML
    yaml_content = f"""# RDD2022 — Road Damage Dataset (country: {country})
path: {out_root.resolve()}
train: train/images
val:   val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = out_root / "rdd2022.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[INFO] Dataset YAML written to: {yaml_path}")
    print(f"[INFO] Train: {len(splits['train'])} images | Val: {len(splits['val'])} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd-root", default="data/RDD2022",   help="Path to downloaded RDD2022 folder")
    ap.add_argument("--out",      default="data",            help="Output directory for processed dataset")
    ap.add_argument("--country",  default="India",           help="Country subset (India/Japan/Norway/USA)")
    ap.add_argument("--val-split",type=float, default=0.2,  help="Fraction for validation")
    args = ap.parse_args()
    prepare(args.rdd_root, args.out, args.country, args.val_split)

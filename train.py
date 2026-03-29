"""
train.py — Fine-tune YOLOv8 on the RDD2022 (Road Damage Dataset) pothole subset.

Dataset: https://github.com/sekilab/RoadDamageDetector
Classes used: D10 (longitudinal crack), D20 (transverse crack), D40 (pothole), D00 (alligator crack)

Usage:
    python src/train.py
    python src/train.py --epochs 100 --batch 16 --img 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str = "data/rdd2022.yaml",
    model_size: str = "yolov8m.pt",   # n / s / m / l / x
    epochs: int = 50,
    img_size: int = 640,
    batch: int = 16,
    project: str = "runs/train",
    name: str = "pothole_v1",
    resume: bool = False,
):
    print(f"[INFO] Starting training: {model_size} | {epochs} epochs | img={img_size} | batch={batch}")

    model = YOLO(model_size)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=project,
        name=name,
        resume=resume,
        patience=15,          # early stopping
        lr0=0.01,
        lrf=0.001,
        mosaic=1.0,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        save=True,
        plots=True,
        verbose=True,
    )

    best = Path(project) / name / "weights" / "best.pt"
    print(f"\n[INFO] Training complete.")
    print(f"[INFO] Best weights: {best}")
    print(f"[INFO] Copy to models/: cp {best} models/best.pt")
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/rdd2022.yaml")
    ap.add_argument("--model",  default="yolov8m.pt",  help="Base model (n/s/m/l/x)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--img",    type=int, default=640)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.img,
        batch=args.batch,
        resume=args.resume,
    )

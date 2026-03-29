import argparse
from ultralytics import YOLO


def evaluate(model_path="models/best.pt", data_yaml="data/rdd2022.yaml", img_size=640, conf=0.35, iou=0.5):
    print(f"[INFO] Evaluating: {model_path}")
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        plots=True,
        verbose=True,
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"mAP@0.50      : {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95 : {metrics.box.map:.4f}")
    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")
    print("=========================================")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--data",  default="data/rdd2022.yaml")
    ap.add_argument("--img",   type=int,   default=640)
    ap.add_argument("--conf",  type=float, default=0.35)
    ap.add_argument("--iou",   type=float, default=0.50)
    args = ap.parse_args()
    evaluate(args.model, args.data, args.img, args.conf, args.iou)

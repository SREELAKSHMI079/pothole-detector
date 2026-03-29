import argparse
import cv2
import time
from pathlib import Path
from ultralytics import YOLO


SEVERITY_THRESHOLDS = {
    "Low":    (0.00, 0.01),
    "Medium": (0.01, 0.04),
    "High":   (0.04, 1.00),
}

SEVERITY_COLORS = {
    "Low":    (0, 255, 0),    # Green
    "Medium": (0, 165, 255),  # Orange
    "High":   (0, 0, 255),    # Red
}


def get_severity(box, frame_area):
    x1, y1, x2, y2 = box
    ratio = ((x2 - x1) * (y2 - y1)) / frame_area
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= ratio < hi:
            return label
    return "High"


def draw_overlay(frame, results, frame_area, fps=0.0):
    counts = {"Low": 0, "Medium": 0, "High": 0}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            sev = get_severity((x1, y1, x2, y2), frame_area)
            color = SEVERITY_COLORS[sev]
            counts[sev] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{sev}  {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    total = sum(counts.values())
    hud = [
        f"Potholes : {total}",
        f"  High   : {counts['High']}",
        f"  Medium : {counts['Medium']}",
        f"  Low    : {counts['Low']}",
        f"FPS      : {fps:.1f}",
    ]
    for i, line in enumerate(hud):
        y = 28 + i * 24
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

    return frame, counts


def run_detection(source, model_path="models/best.pt", conf=0.35, save=False, show=True):
    model = YOLO(model_path)
    print(f"[INFO] Model : {model_path}")
    print(f"[INFO] Source: {source}  |  conf={conf}")

    src = str(source)
    is_webcam = src.isdigit()
    is_image = (not is_webcam) and Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    cap = cv2.VideoCapture(int(src) if is_webcam else src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    writer = None
    if save and not is_image:
        Path("runs").mkdir(exist_ok=True)
        out_path = f"runs/output_{int(time.time())}.mp4"
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
        print(f"[INFO] Saving to: {out_path}")

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_area = frame.shape[0] * frame.shape[1]
        results = model(frame, conf=conf, verbose=False)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        frame, _ = draw_overlay(frame, results, frame_area, fps=fps)

        if show:
            cv2.imshow("Pothole Detector  [Q to quit]", frame)
            key = cv2.waitKey(0 if is_image else 1) & 0xFF
            if key == ord("q"):
                break

        if writer:
            writer.write(frame)

        if is_image:
            if save:
                out_path = f"runs/output_{int(time.time())}.jpg"
                cv2.imwrite(out_path, frame)
                print(f"[INFO] Saved: {out_path}")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YOLOv8 Pothole Detector")
    ap.add_argument("--source", required=True,             help="Image/video path or webcam index")
    ap.add_argument("--model",  default="models/best.pt",  help="Path to .pt weights")
    ap.add_argument("--conf",   type=float, default=0.35,  help="Confidence threshold (0-1)")
    ap.add_argument("--save",   action="store_true",       help="Save annotated output")
    ap.add_argument("--no-display", action="store_true",   help="Suppress display window")
    args = ap.parse_args()
    run_detection(args.source, args.model, args.conf, args.save, not args.no_display)

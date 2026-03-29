# pothole-detector
YOLOv8-based road damage detection 

#  Pothole Detector — YOLOv8-based Road Damage Detection

This project is a real-time road damage detection system that uses **YOLOv8** to identify and classify potholes and road surface defects from images, videos, or a live webcam feed. Each detection is annotated with a **severity rating** (Low / Medium / High) based on the size of the damage relative to the frame.



##  Problem Statement

Potholes and road damage are a persistent hazard across Indian cities, causing vehicle damage, accidents, and increased commute times. Manual inspection is slow, inconsistent, and expensive. This project builds an automated computer vision pipeline that can process dashcam footage or road images to detect and flag damage — enabling faster, data-driven road maintenance decisions.

---

## Project Structure

```
pothole-detector/
├── src/
│   ├── detect.py          # Run detection on image / video / webcam
│   ├── train.py           # Fine-tune YOLOv8 on RDD2022
│   └── evaluate.py        # Compute mAP, precision, recall
├── scripts/
│   └── prepare_dataset.py # Download & convert RDD2022 to YOLO format
├── data/
│   ├── sample_images/     # Drop test images here
│   └── rdd2022.yaml       # Dataset config (auto-generated)
├── models/
│   └── best.pt            # Trained weights (after training)
├── runs/                  # Training logs and detection outputs
├── requirements.txt
└── README.md
```

---

##  Setup

### 1. Clone the repository

```bash
git clone https://github.com/SREELAKSHMI079/pothole-detector.git
cd pothole-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA first:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

##  Dataset

This project uses the **RDD2022 (Road Damage Dataset 2022)** — a large-scale open dataset with 47,420 road images from India, Japan, Norway, and the USA, annotated in Pascal VOC format.

**Download:** https://github.com/sekilab/RoadDamageDetector

**Classes detected:**
| ID | Code | Description        |
|----|------|--------------------|
| 0  | D00  | Alligator crack    |
| 1  | D10  | Longitudinal crack |
| 2  | D20  | Transverse crack   |
| 3  | D40  | **Pothole**        |

### Prepare the dataset

```bash
# After downloading RDD2022 to data/RDD2022/
python scripts/prepare_dataset.py --country India
```

This converts Pascal VOC annotations to YOLO format and generates `data/rdd2022.yaml`.

---

##  Training

```bash
python src/train.py --epochs 50 --batch 16 --img 640
```

| Argument | Default       | Description                   |
|----------|---------------|-------------------------------|
| `--epochs` | 50          | Number of training epochs     |
| `--batch`  | 16          | Batch size                    |
| `--img`    | 640         | Input image size              |
| `--model`  | yolov8m.pt  | Base model (n/s/m/l/x)        |

After training, copy the best weights:
```bash
cp runs/train/pothole_v1/weights/best.pt models/best.pt
```

---

##  Detection

### On a single image
```bash
python src/detect.py --source data/sample_images/road.jpg --save
```

### On a video file
```bash
python src/detect.py --source data/road_video.mp4 --save
```

### Live webcam
```bash
python src/detect.py --source 0
```

| Argument | Default        | Description                          |
|----------|----------------|--------------------------------------|
| `--source` | *(required)* | Image/video path or webcam index `0` |
| `--model`  | models/best.pt | Path to trained weights              |
| `--conf`   | 0.35           | Confidence threshold (0–1)           |
| `--save`   | False          | Save annotated output to `runs/`     |

### Severity classification

| Severity | Colour | Bbox area (% of frame) |
|----------|--------|------------------------|
| Low      | Green  | < 1%              |
| Medium   | Orange | 1% – 4%           |
| High     | Red    | > 4%              |

---

##  Evaluation

```bash
python src/evaluate.py --model models/best.pt --data data/rdd2022.yaml
```

Outputs mAP@0.50, mAP@0.50:0.95, Precision, and Recall on the validation split.

---

##  Model

- **Architecture:** YOLOv8m (medium) — balanced speed/accuracy tradeoff
- **Pre-training:** COCO (via Ultralytics)
- **Fine-tuned on:** RDD2022 India subset
- **Input size:** 640 × 640
- **Inference speed:** ~25–30 FPS on a mid-range GPU

---

##  Dependencies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- PyTorch ≥ 2.0
- tqdm, matplotlib, NumPy

---

##  License

MIT License. Dataset (RDD2022) is licensed separately — see the [RDD2022 repository](https://github.com/sekilab/RoadDamageDetector) for terms.

---

## Author

**[SREELAKSHMI A]** 
REG NO: 23BAI11083 
BRANCH:CSE (AI ML)
[sreelakshmi.23bai11083@vitbhopal.ac.in]
VIT BHOPAL  
Computer Vision Course BYOP 
slot c11+c12 
Faculty Name:Raghavendra Mishra 100468

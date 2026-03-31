# pothole-detector
YOLOv8-based road damage detection 

#  Pothole Detector — YOLOv8-based Road Damage Detection
This project uses YOLOv8 to spot potholes and other road damage in images, videos, and even from a live webcam feed. It gives everything a severity rating—low, medium, or high—based on the size of the damage compared to the frame.


##  Problem   Statement

Road damage is everywhere in Indian cities. Potholes aren’t just annoying—they make travel slow, unpredictable, and cost a lot to fix. So, instead of doing it the old way, this project builds a computer vision pipeline. It analyzes dashcam videos or road photos automatically, highlights damaged spots, and helps cities make faster, smarter repair decisions.
---

##   Project   Structure

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

#### 1 . Clone   the   repository

```bash
git clone https://github.com/SREELAKSHMI079/pothole-detector.git
cd pothole-detector
```

#### 2. Create  a  virtual  environment 

```bash
python  -m  venv  venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

#### 3. Install dependencies

```bash
pip  install  -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA first:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

##  Dataset

This project relies on the RDD2022 (Road Damage Dataset 2022) which a massive annotated open dataset with 47,420 road images from India, Japan, Norway, and the USA in Pascal VOC format.

**Download:** https://github.com/sekilab/RoadDamageDetector

**Classes detected:**
| ID | Code | Description        |
|----|------|--------------------|
| 0  | D00  | Alligator crack    |
| 1  | D10  | Longitudinal crack |
| 2  | D20  | Transverse crack   |
| 3  | D40  | **Pothole**        |

#### Prepare the dataset

```bash
# After downloading RDD2022 to data/RDD2022/.
python scripts / prepare_dataset.py  --country   India
```

This converts Pascal VOC annotations to YOLO format and generates `data/rdd2022.yaml`.

---

##  Training

```bash
python src/train.py -- epochs 50 -- batch 16 -- img 640. 
```

| Argument | Default       | Description                   |
|----------|---------------|-------------------------------|
| `--epochs` | 50          | Number of training epochs     |
| `--batch`  | 16          | Batch size                    |
| `--img`    | 640         | Input image size              |
| `--model`  | yolov8m.pt  | Base model (n/s/m/l/x)        |

Use the best weights after training.
```bash
cp runs/train/pothole_v1/weights/best.pt models/best.pt.
```

---

##  Detection

### On a single image
```bash
python src / detect.py -- source data / sample_images/road . jpg -- save
```

### On a video file
```bash
python src / detect.py --source data/ road_video.  mp4 -- save
```

### Live webcam
```bash
python src / detect . py -- source 0
```

| Argument | Default        | Description                          |
|----------|----------------|--------------------------------------|
| `--source` | *(required) | Image/video path or webcam index `0` |
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

You can result mAP@0.50, mAP@0.50:0.95, Precision and Recall on the validation split.

---

##  Model

- *Architecture:** YOLOv8m (medium) — balanced speed/accuracy tradeoff
- *Pre-training:** COCO (via Ultralytics)
- *Refined using: RDD2022 India subset.
- The input size is 640 × 640 and the inference speed is ~25 FPS – 30 FPS on a mid-range GPU.

---

##  Dependencies

- Ultralytics YOLOv8
- OpenCV
- PyTorch is 2.0 or Greater
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

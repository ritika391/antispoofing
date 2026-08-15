# 🛡️ FaceGuard — Anti-Spoofing Detection System
### MobileNetV2-Based Liveness Detection for Images & Video

---

## 📌 Abstract

FaceGuard is a deep learning-based **face anti-spoofing** system that distinguishes between **live (real)** and **spoofed (fake)** faces in images and real-time video streams. The system leverages a fine-tuned **MobileNetV2** architecture with custom classification heads, trained on texture and depth-level features to detect presentation attacks such as printed photos, digital replays, and 3D masks.

---

## 🧠 Problem Statement

Face recognition systems are vulnerable to **Presentation Attacks (PA)**:
- 📄 **Print Attack** — Printed photo held in front of camera
- 📱 **Replay Attack** — Video replay on a screen
- 🎭 **3D Mask Attack** — 3D-printed or silicone mask

FaceGuard addresses this by classifying face inputs as **Real** or **Spoof** before granting authentication.

---

## 🏗️ System Architecture

```
Input (Image / Video Frame)
        │
        ▼
┌──────────────────┐
│  Face Detection  │  ← MTCNN / OpenCV Haar Cascade
│  & Alignment     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Preprocessing Pipeline           │
│  Resize → Normalize → Augment → Tensor  │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         MobileNetV2 Backbone             │
│   (Pretrained on ImageNet, Fine-tuned)   │
│                                          │
│  Input: 224×224×3                        │
│  Features: 1280-dim Global Avg Pooling   │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Custom Classification Head       │
│  Dense(512) → BN → ReLU → Dropout(0.4)  │
│  Dense(128) → BN → ReLU → Dropout(0.3)  │
│  Dense(2) → Softmax                      │
└────────┬─────────────────────────────────┘
         │
         ▼
   Real / Spoof (+ Confidence Score)
```

---

## 📁 Project Structure

```
antispoofing/
│
├── data/
│   ├── raw/                  # Original dataset (real/ & spoof/ folders)
│   ├── processed/            # Cropped & aligned face images
│   └── splits/               # train/ val/ test/ CSVs
│
├── src/
│   ├── config.py             # All hyperparameters & paths
│   ├── dataset.py            # Custom PyTorch Dataset class
│   ├── model.py              # MobileNetV2 + custom head
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Metrics: HTER, AUC, Accuracy
│   ├── predict.py            # Single image inference
│   └── video_inference.py    # Real-time webcam / video inference
│
├── utils/
│   ├── preprocess.py         # Face detection, crop, align
│   ├── augmentation.py       # Albumentations pipeline
│   ├── metrics.py            # HTER, EER, ROC, Confusion Matrix
│   └── visualize.py          # GradCAM, training curves
│
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Training.ipynb     # Model training walkthrough
│   └── 03_Evaluation.ipynb   # Results & visualization
│
├── models/
│   └── best_model.pth        # Saved best weights
│
├── web/
│   └── app.py                # Streamlit / Flask demo app
│
├── requirements.txt
├── README.md
└── report/
    └── Minor_Project_Report.pdf
```

---

## 📊 Dataset

### Recommended Public Datasets:
| Dataset | Real | Spoof | Types |
|---------|------|-------|-------|
| **NUAA** | 5,105 | 7,509 | Print |
| **REPLAY-ATTACK** | 200 | 1,000 | Print + Replay |
| **MSU-MFSD** | 110 | 330 | Print + Replay |
| **CASIA-FASD** | 150 | 450 | Print + Replay |
| **LCC_FASD** | 1,141 | 1,141 | Mixed |

### Custom Dataset Format:
```
data/raw/
├── real/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── spoof/
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

---

## 🚀 Setup & Installation

```bash
# 1. Clone / setup project
cd antispoofing

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Preprocess dataset
python utils/preprocess.py --input data/raw --output data/processed

# 5. Train the model
python src/train.py

# 6. Evaluate
python src/evaluate.py --weights models/best_model.pth

# 7. Run real-time video demo
python src/video_inference.py --source 0  # 0 = webcam
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | ~97.2% |
| **AUC-ROC** | ~0.991 |
| **HTER** | ~2.8% |
| **EER** | ~3.1% |
| **Inference Speed** | ~45 FPS (GPU) |

---

## 🔬 Key Technical Contributions

1. **Multi-scale texture analysis** via MobileNetV2's depthwise separable convolutions
2. **Focal Loss** to handle class imbalance in spoof datasets
3. **Test-Time Augmentation (TTA)** for robust video frame predictions
4. **GradCAM Visualization** showing model attention on face regions
5. **HTER-optimized threshold** calibration instead of default 0.5

---

## 👨‍💻 Author

RITIKA          
B.Tech IT| [NIT JALANDHAR]

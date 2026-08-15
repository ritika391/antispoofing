"""
config.py — Central Configuration for FaceGuard Anti-Spoofing
All hyperparameters, paths, and model settings in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CONFIG = {
    "raw_dir":       os.path.join(BASE_DIR, "data", "raw"),
    "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
    "splits_dir":    os.path.join(BASE_DIR, "data", "splits"),
    "train_csv":     os.path.join(BASE_DIR, "data", "splits", "train.csv"),
    "val_csv":       os.path.join(BASE_DIR, "data", "splits", "val.csv"),
    "test_csv":      os.path.join(BASE_DIR, "data", "splits", "test.csv"),
}

MODEL_CONFIG = {
    "save_dir":      os.path.join(BASE_DIR, "models"),
    "best_model":    os.path.join(BASE_DIR, "models", "best_model.pth"),
    "last_model":    os.path.join(BASE_DIR, "models", "last_model.pth"),
    "log_dir":       os.path.join(BASE_DIR, "logs"),
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    backbone: str          = "mobilenet_v2"
    pretrained: bool       = True
    num_classes: int       = 2
    image_size: int        = 224
    dropout1: float        = 0.4
    dropout2: float        = 0.3
    hidden_dim1: int       = 512
    hidden_dim2: int       = 128
    freeze_epochs: int     = 3         # Freeze backbone for first N epochs
    unfreeze_at_layer: str = "features.14"  # Unfreeze from this layer onwards


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Basic
    seed: int              = 42
    num_epochs: int        = 5
    batch_size: int        = 8
    num_workers: int       = 0
    device: str            = "cpu"       # "cpu" | "cpu"
    mixed_precision: bool  = True         # AMP for faster training

    # Optimizer — AdamW
    optimizer: str         = "adamw"
    lr: float              = 1e-4
    backbone_lr: float     = 1e-5         # Lower LR for pretrained layers
    weight_decay: float    = 1e-4

    # LR Scheduler — Cosine Annealing with Warmup
    scheduler: str         = "cosine_warmup"
    warmup_epochs: int     = 5
    min_lr: float          = 1e-7

    # Loss
    loss: str              = "focal"      # "focal" | "cross_entropy"
    focal_gamma: float     = 2.0
    focal_alpha: float     = 0.25
    label_smoothing: float = 0.1

    # Regularization
    gradient_clip: float   = 1.0

    # Early Stopping
    patience: int          = 10
    monitor: str           = "val_auc"
    mode: str              = "max"

    # Data Split
    train_ratio: float     = 0.70
    val_ratio: float       = 0.15
    test_ratio: float      = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AugConfig:
    # Geometric
    horizontal_flip: float   = 0.5
    vertical_flip: float     = 0.0
    rotation_limit: int      = 15
    shift_scale_rotate: bool = True
    elastic_transform: bool  = False

    # Color / Texture
    brightness_limit: float  = 0.2
    contrast_limit: float    = 0.2
    hue_shift: int           = 10
    saturation: int          = 20
    blur_limit: int          = 3
    gaussian_noise: bool     = True

    # Anti-spoofing specific
    jpeg_compression: bool   = True   # Simulate print/replay artifacts
    jpeg_quality_range: Tuple = (60, 100)
    coarse_dropout: bool     = True   # Simulate occlusions
    coarse_dropout_holes: int = 8

    # Normalization (ImageNet stats)
    mean: List[float]        = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float]        = field(default_factory=lambda: [0.229, 0.224, 0.225])


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    threshold: float       = 0.5       # Updated after threshold calibration
    tta_enabled: bool      = True      # Test-Time Augmentation
    tta_steps: int         = 5
    video_skip_frames: int = 2         # Process every N frames for speed
    display_confidence: bool = True
    gradcam_enabled: bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FaceDetConfig:
    detector: str          = "mtcnn"    # "mtcnn" | "opencv"
    min_face_size: int     = 40
    margin: int            = 20         # Pixels to expand crop
    confidence_threshold: float = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCES
# ─────────────────────────────────────────────────────────────────────────────
model_cfg     = ModelConfig()
train_cfg     = TrainConfig()
aug_cfg       = AugConfig()
inference_cfg = InferenceConfig()
face_cfg      = FaceDetConfig()

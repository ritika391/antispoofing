"""
utils/augmentation.py — Albumentations-based Augmentation Pipelines
Separate pipelines for train, validation, and test-time augmentation (TTA).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import aug_cfg, model_cfg



# ─────────────────────────────────────────────────────────────────────────────
# TRAINING AUGMENTATION — Heavy, anti-spoofing aware
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Aggressive augmentation pipeline for training.
    Includes JPEG compression to simulate print/replay attack artifacts.
    """
    return A.Compose([
        # ── Geometric ──────────────────────────────────────────────────────
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=aug_cfg.horizontal_flip),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=aug_cfg.rotation_limit,
            border_mode=0,
            p=0.5,
        ),
        A.OneOf([
            A.ElasticTransform(alpha=60, sigma=6, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.1, p=0.3),
        ], p=0.2),

        # ── Color / Texture ────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.brightness_limit,
            contrast_limit=aug_cfg.contrast_limit,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug_cfg.hue_shift,
            sat_shift_limit=aug_cfg.saturation,
            p=0.4,
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),      # Enhance local contrast
        A.Sharpen(alpha=(0.2, 0.5), p=0.2),   # Simulate texture sharpness

        # ── Anti-Spoofing Specific Augmentations ───────────────────────────
        A.ImageCompression(                    # Simulate JPEG artifacts
            quality_lower=aug_cfg.jpeg_quality_range[0],
            quality_upper=aug_cfg.jpeg_quality_range[1],
            p=0.4,
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=aug_cfg.blur_limit, p=0.3),
            A.GaussianBlur(blur_limit=aug_cfg.blur_limit, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
        ], p=0.3),                             # Simulate screen blur / defocus
        A.GaussNoise(var_limit=(10, 50), p=0.3),  # Camera sensor noise
        A.CoarseDropout(                       # Simulate partial occlusions
            max_holes=aug_cfg.coarse_dropout_holes,
            max_height=16,
            max_width=16,
            min_holes=2,
            fill_value=0,
            p=0.25,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),  # Screen gamma variation
        A.Downscale(scale_min=0.7, scale_max=0.95, p=0.2),  # Low-res spoofs

        # ── Normalization ──────────────────────────────────────────────────
        A.Normalize(mean=aug_cfg.mean, std=aug_cfg.std),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION / TEST TRANSFORMS — Minimal, deterministic
# ─────────────────────────────────────────────────────────────────────────────
def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=aug_cfg.mean, std=aug_cfg.std),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TEST-TIME AUGMENTATION (TTA) — Multiple views, averaged prediction
# ─────────────────────────────────────────────────────────────────────────────
def get_tta_transforms(image_size: int = 224) -> list:
    """
    Returns a list of transform pipelines for TTA.
    Each image is processed through all transforms and predictions averaged.
    """
    base_norm = [
        A.Normalize(mean=aug_cfg.mean, std=aug_cfg.std),
        ToTensorV2(),
    ]
    return [
        # 1. Original (no augmentation)
        A.Compose([A.Resize(image_size, image_size)] + base_norm),
        # 2. Horizontal flip
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0)] + base_norm),
        # 3. Slight brightness increase
        A.Compose([A.Resize(image_size, image_size), A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0, 0), p=1.0)] + base_norm),
        # 4. Slight brightness decrease
        A.Compose([A.Resize(image_size, image_size), A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=(0, 0), p=1.0)] + base_norm),
        # 5. Center crop
        A.Compose([A.Resize(int(image_size * 1.1), int(image_size * 1.1)), A.CenterCrop(image_size, image_size)] + base_norm),
    ]
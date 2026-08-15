"""
utils/preprocess.py — Face Detection, Crop, Align & Dataset Split
Processes raw data/real and data/spoof folders into aligned face crops.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
from typing import Optional, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import face_cfg, DATA_CONFIG, train_cfg


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTOR WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class FaceDetector:
    """
    Unified wrapper around MTCNN (preferred) and OpenCV Haar Cascade (fallback).
    MTCNN provides landmark-based alignment for better feature extraction.
    """

    def __init__(self, detector: str = "mtcnn", device: str = "cpu"):
        self.detector_type = detector
        self.device = device
        self._load_detector()

    def _load_detector(self):
        if self.detector_type == "mtcnn":
            try:
                from facenet_pytorch import MTCNN
                self.detector = MTCNN(
                    image_size=224,
                    margin=face_cfg.margin,
                    min_face_size=face_cfg.min_face_size,
                    thresholds=[0.6, 0.7, face_cfg.confidence_threshold],
                    keep_all=False,
                    device=self.device,
                )
                print("✅ MTCNN face detector loaded")
            except ImportError:
                print("⚠️  facenet-pytorch not found. Falling back to OpenCV.")
                self.detector_type = "opencv"
                self._load_opencv()
        else:
            self._load_opencv()

    def _load_opencv(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        print("✅ OpenCV Haar Cascade loaded")

    def detect_and_crop(
        self, image: np.ndarray, target_size: int = 224
    ) -> Optional[np.ndarray]:
        """
        Detect face in image and return aligned, cropped face tensor.
        Returns None if no face detected.

        Args:
            image: BGR numpy array (OpenCV format)
            target_size: output image size (square)

        Returns:
            Cropped face as numpy array (RGB, target_size x target_size)
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.detector_type == "mtcnn":
            return self._detect_mtcnn(rgb, target_size)
        else:
            return self._detect_opencv(rgb, image, target_size)

    def _detect_mtcnn(self, rgb: np.ndarray, size: int) -> Optional[np.ndarray]:
        """MTCNN detection with landmark-based alignment."""
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        face_tensor = self.detector(pil_img)
        if face_tensor is None:
            return None
        # face_tensor is (C, H, W) in [-1, 1], convert to uint8 RGB
        face_np = ((face_tensor.permute(1, 2, 0).numpy() + 1) / 2 * 255).astype(np.uint8)
        face_np = cv2.resize(face_np, (size, size))
        return face_np

    def _detect_opencv(
        self, rgb: np.ndarray, bgr: np.ndarray, size: int
    ) -> Optional[np.ndarray]:
        """OpenCV Haar Cascade detection."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(face_cfg.min_face_size,) * 2
        )
        if len(faces) == 0:
            return None
        # Take largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = face_cfg.margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(rgb.shape[1], x + w + margin)
        y2 = min(rgb.shape[0], y + h + margin)
        face = rgb[y1:y2, x1:x2]
        return cv2.resize(face, (size, size))


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class DatasetProcessor:
    """
    Processes raw images → detects faces → saves aligned crops.
    Also generates train/val/test CSV splits.
    """

    CLASSES = {"real": 0, "spoof": 1}
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        detector: FaceDetector,
        target_size: int = 224,
    ):
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.detector   = detector
        self.target_size = target_size
        self.records    = []  # [{path, label, split}]

    def process_all(self):
        """Main pipeline: detect → crop → save → split."""
        print("\n" + "═" * 60)
        print("  FaceGuard — Dataset Preprocessing Pipeline")
        print("═" * 60)

        total_saved = 0
        total_failed = 0

        for class_name, label in self.CLASSES.items():
            class_dir = self.input_dir / class_name
            if not class_dir.exists():
                print(f"⚠️  Directory not found: {class_dir}. Skipping.")
                continue

            out_class_dir = self.output_dir / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            images = [
                p for p in class_dir.rglob("*")
                if p.suffix.lower() in self.IMG_EXTS
            ]
            print(f"\n📂 Processing '{class_name}' — {len(images)} images")

            saved = failed = 0
            for img_path in tqdm(images, desc=f"  {class_name}", unit="img"):
                out_path = out_class_dir / img_path.name
                if out_path.exists():
                    self.records.append({"path": str(out_path), "label": label})
                    saved += 1
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    failed += 1
                    continue

                face = self.detector.detect_and_crop(img, self.target_size)
                if face is None:
                    # Fallback: use full image resized (better than losing the sample)
                    face = cv2.cvtColor(
                        cv2.resize(img, (self.target_size, self.target_size)),
                        cv2.COLOR_BGR2RGB,
                    )

                cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                self.records.append({"path": str(out_path), "label": label})
                saved += 1

            total_saved  += saved
            total_failed += failed
            print(f"  ✅ Saved: {saved} | ❌ Failed: {failed}")

        print(f"\n📊 Total processed: {total_saved} images saved, {total_failed} failed")
        self._generate_splits()

    def _generate_splits(self):
        """Stratified train/val/test split with class balance."""
        import random
        from sklearn.model_selection import train_test_split

        splits_dir = Path(DATA_CONFIG["splits_dir"])
        splits_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.records)
        print(f"\n📈 Class distribution:\n{df['label'].value_counts().to_string()}")

        # Stratified split
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_cfg.train_ratio),
            stratify=df["label"],
            random_state=train_cfg.seed,
        )
        val_size = train_cfg.val_ratio / (train_cfg.val_ratio + train_cfg.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df["label"],
            random_state=train_cfg.seed,
        )

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            out_csv = splits_dir / f"{split_name}.csv"
            split_df["split"] = split_name
            split_df.to_csv(out_csv, index=False)
            r = (split_df["label"] == 0).sum()
            s = (split_df["label"] == 1).sum()
            print(f"  {split_name:5s}: {len(split_df):5d} samples  (real={r}, spoof={s})")

        print(f"\n✅ CSV splits saved to: {splits_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FaceGuard — Preprocessing Pipeline"
    )
    parser.add_argument(
        "--input",  default=DATA_CONFIG["raw_dir"],
        help="Input directory with real/ and spoof/ sub-folders",
    )
    parser.add_argument(
        "--output", default=DATA_CONFIG["processed_dir"],
        help="Output directory for processed face crops",
    )
    parser.add_argument(
        "--detector", default=face_cfg.detector,
        choices=["mtcnn", "opencv"],
        help="Face detection backend",
    )
    parser.add_argument(
        "--size", type=int, default=224,
        help="Output image size (default: 224)",
    )
    args = parser.parse_args()

    detector = FaceDetector(detector=args.detector)
    processor = DatasetProcessor(
        input_dir=args.input,
        output_dir=args.output,
        detector=detector,
        target_size=args.size,
    )
    processor.process_all()


if __name__ == "__main__":
    main()
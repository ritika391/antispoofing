"""
src/predict.py - Single image inference for FaceGuard
"""

import os, sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_CONFIG, train_cfg, model_cfg
from src.model import load_model
from utils.augmentation import get_val_transforms


def predict_image(model, image_path: str, device, threshold: float = 0.5):
    transform = get_val_transforms(model_cfg.image_size)

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        spoof_prob = probs[1].item()
        real_prob  = probs[0].item()
        pred_label = 1 if spoof_prob >= threshold else 0

    label_str = "🚨 SPOOF" if pred_label == 1 else "✅ REAL"
    confidence = spoof_prob if pred_label == 1 else real_prob

    print(f"\n{'='*45}")
    print(f"  Image      : {Path(image_path).name}")
    print(f"  Prediction : {label_str}")
    print(f"  Confidence : {confidence*100:.1f}%")
    print(f"  Real prob  : {real_prob*100:.1f}%")
    print(f"  Spoof prob : {spoof_prob*100:.1f}%")
    print(f"  Threshold  : {threshold}")
    print(f"{'='*45}\n")

    return pred_label, confidence


def main():
    parser = argparse.ArgumentParser(description="FaceGuard single image prediction")
    parser.add_argument("--model_path",  default=MODEL_CONFIG["best_model"], help="Path to model weights")
    parser.add_argument("--image_path",  required=True, help="Path to input image")
    parser.add_argument("--threshold",   type=float, default=0.5, help="Spoof decision threshold")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        # List available images
        real_dir = "data/processed/real"
        if os.path.exists(real_dir):
            files = os.listdir(real_dir)[:5]
            print(f"   Available in {real_dir}: {files}")
        return

    device = torch.device(train_cfg.device)
    model  = load_model(args.model_path, train_cfg.device)
    predict_image(model, args.image_path, device, args.threshold)


if __name__ == "__main__":
    main()
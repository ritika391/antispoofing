"""
src/video_inference.py - Real-time webcam / video inference for FaceGuard
"""

import os, sys
import cv2
import torch
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_CONFIG, train_cfg, model_cfg
from src.model import load_model
from utils.augmentation import get_val_transforms


def draw_result(frame, label, confidence, fps=None):
    h, w = frame.shape[:2]
    is_spoof = label == 1
    color  = (0, 0, 255) if is_spoof else (0, 255, 0)
    text   = f"SPOOF  {confidence*100:.1f}%" if is_spoof else f"REAL  {confidence*100:.1f}%"

    # Background bar
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # FPS
    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Border
    border_color = (0, 0, 255) if is_spoof else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 3)

    return frame


def run_inference(model, source, device, threshold=0.5, skip_frames=2):
    transform = get_val_transforms(model_cfg.image_size)
    model.eval()

    # Open source
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        return

    print(f"\n✅ Running inference on: {source}")
    print(f"   Press 'q' to quit | 's' to save frame")
    print(f"   Threshold: {threshold}\n")

    import time
    frame_count = 0
    last_label, last_conf = 0, 0.0
    fps_timer = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⏹ Stream ended.")
            break

        frame_count += 1

        # Run model every N frames for speed
        if frame_count % skip_frames == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = transform(image=img_rgb)
            tensor = transformed["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                spoof_prob = probs[1].item()
                real_prob  = probs[0].item()
                last_label = 1 if spoof_prob >= threshold else 0
                last_conf  = spoof_prob if last_label == 1 else real_prob

            # FPS
            now = time.time()
            fps = skip_frames / (now - fps_timer + 1e-6)
            fps_timer = now

        # Draw on every frame
        display = draw_result(frame.copy(), last_label, last_conf, fps)
        cv2.imshow("FaceGuard Anti-Spoofing", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Quit.")
            break
        elif key == ord('s'):
            fname = f"capture_{frame_count}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸 Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="FaceGuard real-time inference")
    parser.add_argument("--model_path",   default=MODEL_CONFIG["best_model"])
    parser.add_argument("--source",       default="0", help="0=webcam or path to video file")
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--skip_frames",  type=int,   default=2, help="Process every N frames")
    args = parser.parse_args()

    device = torch.device(train_cfg.device)
    model  = load_model(args.model_path, train_cfg.device)
    run_inference(model, args.source, device, args.threshold, args.skip_frames)


if __name__ == "__main__":
    main()
"""
src/evaluate.py - Evaluation: Accuracy, AUC, HTER, Confusion Matrix
"""

import os, sys
import numpy as np
import torch
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_CONFIG, MODEL_CONFIG, train_cfg
from src.dataset import build_dataloaders
from src.model import load_model, build_criterion


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        logits, _ = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(1)
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# def compute_hter(labels, preds):
#     real_mask  = labels == 0
#     spoof_mask = labels == 1
#     far = (preds[real_mask]  == 1).mean() if real_mask.any()  else 0.0  # False Accept Rate
#     frr = (preds[spoof_mask] == 0).mean() if spoof_mask.any() else 0.0  # False Reject Rate
#     hter = (far + frr) / 2
#     return far, frr, hter
def compute_hter(labels, preds):
    real_mask = labels == 0
    spoof_mask = labels == 1

    frr = (preds[real_mask] == 1).mean() if real_mask.any() else 0.0
    far = (preds[spoof_mask] == 0).mean() if spoof_mask.any() else 0.0

    hter = (far + frr) / 2

    return far, frr, hter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default=MODEL_CONFIG["best_model"])
    parser.add_argument("--splits_dir",  default=DATA_CONFIG["splits_dir"])
    args = parser.parse_args()

    device = torch.device(train_cfg.device)
    print(f"\n{'='*50}")
    print(f"  FaceGuard Evaluation")
    print(f"  Model  : {args.model_path}")
    print(f"  Device : {device}")
    print(f"{'='*50}\n")

    loaders = build_dataloaders(
        train_csv=os.path.join(args.splits_dir, "train.csv"),
        val_csv  =os.path.join(args.splits_dir, "val.csv"),
        test_csv =os.path.join(args.splits_dir, "test.csv"),
    )
    test_loader = loaders["test"]

    model = load_model(args.model_path, train_cfg.device)
    labels, preds, probs = run_evaluation(model, test_loader, device)

    # Metrics
    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                  classification_report, confusion_matrix)

    acc  = accuracy_score(labels, preds)
    auc  = roc_auc_score(labels, probs)
    far, frr, hter = compute_hter(labels, preds)

    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  AUC      : {auc:.4f}")
    print(f"  HTER     : {hter:.4f}")
    print(f"  FAR      : {far:.4f}  (real faces wrongly rejected)")
    print(f"  FRR      : {frr:.4f}  (spoof faces wrongly accepted)")
    print()
    print("  Classification Report:")
    print(classification_report(labels, preds, target_names=["Real", "Spoof"]))
    print("  Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(f"              Pred Real  Pred Spoof")
    print(f"  True Real  :   {cm[0][0]:>4}       {cm[0][1]:>4}")
    print(f"  True Spoof :   {cm[1][0]:>4}       {cm[1][1]:>4}")
    print(f"\n{'='*50}")


if __name__ == "__main__":
    main()
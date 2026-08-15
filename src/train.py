"""
src/train.py - Training loop for FaceGuard Anti-Spoofing
"""

import os, sys, time, random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_CONFIG, MODEL_CONFIG, train_cfg, model_cfg
from src.dataset import build_dataloaders
from src.model import build_model, build_criterion


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return total_loss / total, correct / total, auc


def main():
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    print(f"\n{'='*50}")
    print(f"  FaceGuard Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {train_cfg.num_epochs}")
    print(f"{'='*50}")

    # Dirs
    os.makedirs(MODEL_CONFIG["save_dir"], exist_ok=True)
    os.makedirs(MODEL_CONFIG["log_dir"],  exist_ok=True)

    # Data
    loaders = build_dataloaders()
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    class_weights = loaders["class_weights"].to(device)

    # Model
    model = build_model(train_cfg.device)
    criterion = build_criterion(class_weights)

    # Optimizer
    optimizer = optim.AdamW([
        {"params": model.head.parameters(), "lr": train_cfg.lr},
    ], weight_decay=train_cfg.weight_decay)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.num_epochs,
        eta_min=train_cfg.min_lr,
    )

    best_auc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Val AUC':>9} {'LR':>10}")
    print("-" * 75)

    for epoch in range(1, train_cfg.num_epochs + 1):

        # Unfreeze backbone after freeze_epochs
        if epoch == model_cfg.freeze_epochs + 1:
            model.unfreeze_top_layers(model_cfg.unfreeze_at_layer)
            optimizer.add_param_group({
                "params": [p for p in model.features.parameters() if p.requires_grad],
                "lr": train_cfg.backbone_lr,
            })

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>9.4f} {val_auc:>9.4f} {lr:>10.2e}  ({elapsed:.1f}s)")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "val_acc": val_acc,
            }, MODEL_CONFIG["best_model"])
            print(f"  ✅ Best model saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1

        # Save last model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_auc": best_auc,
        }, MODEL_CONFIG["last_model"])

        # Early stopping
        if patience_counter >= train_cfg.patience:
            print(f"\n⏹ Early stopping at epoch {epoch} (no improvement for {train_cfg.patience} epochs)")
            break

    print(f"\n{'='*50}")
    print(f"  Training complete! Best Val AUC: {best_auc:.4f}")
    print(f"  Model saved to: {MODEL_CONFIG['best_model']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

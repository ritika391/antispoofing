"""
src/dataset.py — Custom PyTorch Dataset & DataLoader Factory
Reads CSV splits, loads preprocessed images, and applies transforms.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Callable, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_CONFIG, train_cfg, model_cfg
from utils.augmentation import get_train_transforms, get_val_transforms


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class AntiSpoofDataset(Dataset):
    """
    Face anti-spoofing dataset from CSV manifest.

    CSV columns expected:
        path  : absolute or relative path to image
        label : 0 = real, 1 = spoof

    Args:
        csv_path   : path to split CSV
        transform  : albumentations transform pipeline
        augment    : if True, apply heavier augmentation (for training)
    """

    CLASS_NAMES = {0: "Real", 1: "Spoof"}

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        self.df         = pd.read_csv(csv_path)
        self.transform  = transform
        self.return_path = return_path

        # Validate CSV
        assert "path"  in self.df.columns, "CSV must have 'path' column"
        assert "label" in self.df.columns, "CSV must have 'label' column"

        self.paths  = self.df["path"].tolist()
        self.labels = self.df["label"].tolist()

        print(f"  📂 Loaded {len(self.df)} samples | "
              f"Real: {self.labels.count(0)} | "
              f"Spoof: {self.labels.count(1)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.paths[idx]
        label    = self.labels[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            # Return black image if file is corrupted/missing
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return img, label_tensor, img_path
        return img, label_tensor

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss balancing."""
        counts = pd.Series(self.labels).value_counts().sort_index()
        weights = 1.0 / counts.values.astype(np.float32)
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_counts = pd.Series(self.labels).value_counts().to_dict()
        sample_weights = [1.0 / class_counts[l] for l in self.labels]
        return torch.tensor(sample_weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    train_csv: str = DATA_CONFIG["train_csv"],
    val_csv:   str = DATA_CONFIG["val_csv"],
    test_csv:  str = DATA_CONFIG["test_csv"],
    image_size: int = model_cfg.image_size,
    batch_size: int = train_cfg.batch_size,
    num_workers: int = train_cfg.num_workers,
    oversample: bool = True,
) -> dict:
    """
    Build train/val/test DataLoaders with proper augmentation and sampling.

    Args:
        oversample : Use WeightedRandomSampler to handle class imbalance

    Returns:
        dict with keys: 'train', 'val', 'test', 'class_weights'
    """
    print("\n🔄 Building DataLoaders...")

    train_ds = AntiSpoofDataset(train_csv, get_train_transforms(image_size))
    val_ds   = AntiSpoofDataset(val_csv,   get_val_transforms(image_size))
    test_ds  = AntiSpoofDataset(test_csv,  get_val_transforms(image_size), return_path=True)

    # Weighted sampling to handle imbalanced datasets
    train_sampler = None
    if oversample:
        sample_weights = train_ds.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_weights = train_ds.get_class_weights()

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")
    print(f"  Class weights : {class_weights.tolist()}")

    return {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
        "class_weights": class_weights,
    }
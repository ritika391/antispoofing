"""
src/model.py — FaceGuard MobileNetV2 Anti-Spoofing Model
Complete architecture with custom classification head, Focal Loss,
and selective layer unfreezing strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from typing import Tuple, Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import model_cfg


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS — Handles class imbalance better than Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard, misclassified examples.
    Particularly effective for face anti-spoofing where spoof patterns
    can be subtle and easy examples (obvious spoofs) dominate.

    Args:
        gamma : focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha : class weight tensor [w_real, w_spoof]
        label_smoothing : prevents overconfidence
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Will be moved to device during forward
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : raw model output (B, num_classes)
            targets : ground truth labels (B,)
        """
        num_classes = logits.size(1)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# SQUEEZE-EXCITATION BLOCK — Channel-wise attention
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Recalibrates channel-wise feature responses adaptively.
    Helps model focus on texture-discriminative channels.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────
class AntiSpoofHead(nn.Module):
    """
    Deep classification head replacing MobileNetV2's default classifier.

    Architecture:
        GAP → 1280-dim
        → Dense(512) → BN → GELU → Dropout(0.4)
        → Dense(128)  → BN → GELU → Dropout(0.3)
        → Dense(2)    → (Softmax at inference)

    BN before activation stabilizes training on fine-tuned features.
    GELU used instead of ReLU for smoother gradient flow.
    """

    def __init__(
        self,
        in_features: int = 1280,
        hidden1: int = 512,
        hidden2: int = 128,
        num_classes: int = 2,
        dropout1: float = 0.4,
        dropout2: float = 0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            # Block 1
            nn.Linear(in_features, hidden1, bias=False),
            nn.BatchNorm1d(hidden1),
            nn.GELU(),
            nn.Dropout(dropout1),

            # Block 2
            nn.Linear(hidden1, hidden2, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
            nn.Dropout(dropout2),

            # Output
            nn.Linear(hidden2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL: FaceGuard MobileNetV2
# ─────────────────────────────────────────────────────────────────────────────
class FaceGuardModel(nn.Module):
    """
    FaceGuard — MobileNetV2-based Face Anti-Spoofing Detector.

    Architecture Overview:
    ┌─────────────────────────────────────────────────────────┐
    │  Input: 224 × 224 × 3 (RGB, ImageNet normalized)        │
    │                                                          │
    │  MobileNetV2 Backbone (pretrained on ImageNet)           │
    │  ├─ Initial Conv 3×3 (32 filters, stride 2)              │
    │  ├─ Inverted Residual Blocks (t=6, various strides)      │
    │  │   Features capture: edges → textures → patterns       │
    │  └─ Conv 1×1 → 1280 feature maps                         │
    │                                                          │
    │  Global Average Pooling → 1280-d vector                  │
    │                                                          │
    │  Custom Head:                                            │
    │  Dense(512) → BN → GELU → Dropout(0.4)                  │
    │  Dense(128)  → BN → GELU → Dropout(0.3)                 │
    │  Dense(2)   [Real / Spoof]                               │
    └─────────────────────────────────────────────────────────┘

    Total Parameters: ~3.4M (backbone: ~2.2M, head: ~0.7M)
    Trainable (fine-tune): ~1.5M
    """

    def __init__(
        self,
        num_classes: int        = model_cfg.num_classes,
        pretrained: bool        = model_cfg.pretrained,
        hidden1: int            = model_cfg.hidden_dim1,
        hidden2: int            = model_cfg.hidden_dim2,
        dropout1: float         = model_cfg.dropout1,
        dropout2: float         = model_cfg.dropout2,
    ):
        super().__init__()

        # ── Load MobileNetV2 Backbone ──────────────────────────────────────
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # Remove original classifier; keep features + avgpool
        self.features = mobilenet.features      # Output: (B, 1280, 7, 7)
        self.avgpool  = nn.AdaptiveAvgPool2d(1) # Output: (B, 1280, 1, 1)

        # ── Custom Classification Head ─────────────────────────────────────
        self.head = AntiSpoofHead(
            in_features=1280,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=num_classes,
            dropout1=dropout1,
            dropout2=dropout2,
        )

        # ── Freeze backbone initially ──────────────────────────────────────
        self._freeze_backbone()

        # ── Print model summary ────────────────────────────────────────────
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n🧠 FaceGuard MobileNetV2 Model")
        print(f"   Total parameters     : {total:,}")
        print(f"   Trainable parameters : {trainable:,}")
        print(f"   Frozen parameters    : {total - trainable:,}")

    def _freeze_backbone(self):
        """Freeze entire backbone; only head trains initially."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, from_layer: str = "features.14"):
        """
        Progressively unfreeze backbone from a specific layer onwards.
        Called after initial head training (warm-up phase).

        MobileNetV2 layer structure:
          features.0  — Initial conv
          features.1–7  — Inverted residuals (early, low-level)
          features.8–14 — Inverted residuals (mid, texture)
          features.15–18 — Inverted residuals (high-level)
        """
        unfreeze = False
        for name, module in self.features.named_children():
            full_name = f"features.{name}"
            if full_name == from_layer:
                unfreeze = True
            if unfreeze:
                for param in module.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Unfroze layers from '{from_layer}'. Trainable: {trainable:,}")

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1280-d feature vector (for visualization, retrieval)."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x : input tensor (B, 3, 224, 224)

        Returns:
            logits   : raw scores (B, 2)
            features : 1280-d feature vector (B, 1280) — for GradCAM
        """
        features = self.get_feature_vector(x)
        logits   = self.head(features)
        return logits, features


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_model(device: str = "cuda") -> FaceGuardModel:
    """Build and move model to device."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model  = FaceGuardModel()
    model  = model.to(device)
    print(f"   Device: {device}")
    return model


def load_model(weights_path: str, device: str = "cuda") -> FaceGuardModel:
    """Load model from saved checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model  = FaceGuardModel()
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    auc   = checkpoint.get("best_auc", "?")
    print(f"✅ Loaded model — Epoch: {epoch} | Best AUC: {auc}")
    return model


def build_criterion(class_weights: Optional[torch.Tensor] = None) -> FocalLoss:
    """Build Focal Loss with optional class weights."""
    from src.config import train_cfg
    return FocalLoss(
        gamma=train_cfg.focal_gamma,
        alpha=class_weights,
        label_smoothing=train_cfg.label_smoothing,
    )
"""
model.py
Defines the ArtStyleEmbeddingModel — a ConvNeXtV2-Tiny backbone plus a linear
head, shared by the classification and contrastive training stages.
"""

import torch
import torch.nn as nn
import timm
from typing import Any


class ArtStyleEmbeddingModel(nn.Module):
    """
    Art style embedding model.

    Architecture:
        ConvNeXtV2-Tiny backbone -> 1-layer projection head (backbone -> embedding_dim).

    The model projects art style images into a shared embedding space. With
    `embedding_dim` set to the number of classes, the same head acts as a linear
    classifier (Stage 1); with `embedding_dim=256` it produces embeddings for
    contrastive training (Stage 2).
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        embedding_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        """
        Initialize the model.

        Args:
            backbone_name: timm model name for the backbone.
            embedding_dim: Output dimension of the projection head.
            pretrained: Whether to load pretrained weights for the backbone.
        """
        super().__init__()

        # Create backbone with default classifier removed (outputs feature vector)
        self.backbone: nn.Module = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Image preprocessing config for use with timm.data.create_transform()
        self.data_config: dict[str, Any] = timm.data.resolve_model_data_config(
            self.backbone
        )

        # Linear projection head
        self.head: nn.Linear = nn.Linear(
            self.backbone.num_features,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Extract features from the backbone.
        2. Project to embedding_dim dimensions.
        Returns the projection head output (embedding_dim dimensions).
        """
        features: torch.Tensor = self.backbone(x)
        embedding: torch.Tensor = self.head(features)
        return embedding

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict[str, Any]]:
        """
        Return parameter groups with separate learning rates for backbone and head.

        Biases and normalization parameters (anything with ndim <= 1) are placed in
        a group with weight_decay=0.0, so the optimizer-level weight_decay only
        applies to the remaining (decay) groups.

        Args:
            backbone_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for head parameters.

        Returns:
            List of parameter group dicts for use with an optimizer.
        """

        def _split(module: nn.Module, lr: float) -> list[dict[str, Any]]:
            decay: list[nn.Parameter] = []
            no_decay: list[nn.Parameter] = []
            for _, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (no_decay if param.ndim <= 1 else decay).append(param)
            return [
                {"params": decay, "lr": lr},
                {"params": no_decay, "lr": lr, "weight_decay": 0.0},
            ]

        return _split(self.backbone, backbone_lr) + _split(self.head, head_lr)


if __name__ == "__main__":
    # Basic test: verify output shape
    model: ArtStyleEmbeddingModel = ArtStyleEmbeddingModel()
    x: torch.Tensor = torch.randn(2, 3, 224, 224)
    out: torch.Tensor = model(x)

    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    # Verify param groups: (backbone decay/no-decay, head decay/no-decay)
    groups: list[dict[str, Any]] = model.get_param_groups(backbone_lr=1e-5, head_lr=1e-3)
    assert len(groups) == 4, f"Expected 4 param groups, got {len(groups)}"
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-5
    assert groups[2]["lr"] == 1e-3
    assert groups[3]["lr"] == 1e-3
    assert groups[1]["weight_decay"] == 0.0
    assert groups[3]["weight_decay"] == 0.0

    # Verify data_config is usable for creating transforms
    train_transform = timm.data.create_transform(is_training=True, **model.data_config)
    eval_transform = timm.data.create_transform(is_training=False, **model.data_config)
    assert train_transform is not None
    assert eval_transform is not None

    print(f"Output shape: {out.shape}")
    print(f"Param groups: {len(groups)} groups")
    print(f"Data config keys: {list(model.data_config.keys())}")
    print(f"Train transform: {type(train_transform).__name__}")
    print(f"Eval transform: {type(eval_transform).__name__}")
    print("All tests passed.")

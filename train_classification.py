"""
train_classification.py
Stage 1: supervised classification pretraining of the ArtStyleEmbeddingModel.

Trains a ConvNeXtV2-Tiny backbone + linear classifier head with cross-entropy
so the backbone learns style-discriminative features. Saves the best checkpoint
by validation accuracy.

Runnable directly (uses the defaults below) or callable from `main.py` with
explicit arguments.
"""

import os
import random
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ArtStyleEmbeddingModel
from dataset import create_train_val_split
from utils import build_class_to_idx, load_checkpoint, save_checkpoint


# --- Default configuration ---
DATASET_NAME = "huggan/wikiart"
LABEL_NAME = "style"
BACKBONE_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
EPOCHS = 3
BATCH_SIZE = 128
NUM_WORKERS = 4
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
VAL_RATIO = 0.1
OUTPUT_DIR = "checkpoints"
CHECKPOINT_NAME = "stage1_classification_WikiArt.pt"
RESUME_FROM = None  # optional path to a Stage 1 checkpoint to resume from
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def main(
    dataset_name: str = DATASET_NAME,
    label_name: str = LABEL_NAME,
    backbone_name: str = BACKBONE_NAME,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    backbone_lr: float = BACKBONE_LR,
    head_lr: float = HEAD_LR,
    weight_decay: float = WEIGHT_DECAY,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    val_ratio: float = VAL_RATIO,
    output_dir: str = OUTPUT_DIR,
    checkpoint_name: str = CHECKPOINT_NAME,
    resume_from: str | None = RESUME_FROM,
    device: str = DEVICE,
    seed: int = SEED,
) -> Path:
    """Run Stage 1 training and return the path of the best checkpoint."""
    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_device = torch.device(device)
    print(f"Device: {torch_device}")

    # Resolve the backbone's preprocessing config without building a throwaway model
    data_config = timm.data.resolve_model_data_config(backbone_name)

    # Load the dataset once; the same seed gives Stage 2 the same train/val boundary.
    train_ds, val_ds = create_train_val_split(
        dataset_name=dataset_name,
        data_config=data_config,
        label_name=label_name,
        val_ratio=val_ratio,
        seed=seed,
    )

    class_to_idx = build_class_to_idx(train_ds.hf_dataset[label_name])
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Build the model once with the correct classification head
    model = ArtStyleEmbeddingModel(
        backbone_name=backbone_name,
        embedding_dim=num_classes,
    )
    model = model.to(torch_device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Optimizer (weight_decay applies only to the decay param groups)
    optimizer = AdamW(
        model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr),
        weight_decay=weight_decay,
    )

    # Scheduler (cosine annealing per epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision
    use_amp = torch_device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Output path
    checkpoint_path = Path(output_dir) / checkpoint_name
    os.makedirs(output_dir, exist_ok=True)

    # Resume an interrupted run (restores model, optimizer, scheduler and epoch)
    start_epoch = 0
    best_val_acc = 0.0
    if resume_from is not None and Path(resume_from).exists():
        checkpoint = load_checkpoint(resume_from, torch_device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("metrics", {}).get("val_acc", 0.0)
        print(
            f"Resumed from {resume_from} at epoch {start_epoch} "
            f"(best val acc {best_val_acc:.4f})"
        )

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images = images.to(torch_device, non_blocking=True)
            labels = labels.to(torch_device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(torch_device.type, enabled=use_amp):
                embeddings = model(images)
                loss = criterion(embeddings, labels)

            # Backward
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # Track
            train_loss += loss.item() * images.size(0)
            _, predicted = embeddings.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(torch_device, non_blocking=True)
                labels = labels.to(torch_device, non_blocking=True)

                with autocast(torch_device.type, enabled=use_amp):
                    embeddings = model(images)
                    loss = criterion(embeddings, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = embeddings.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        # Compute metrics
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model (shared checkpoint format, backbone reusable by Stage 2)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                checkpoint_path,
                stage="classification",
                backbone_name=backbone_name,
                num_classes=num_classes,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={"val_acc": val_acc},
            )
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return checkpoint_path


if __name__ == "__main__":
    main()

"""
train_contrastive.py
Stage 2: supervised contrastive training of the ArtStyleEmbeddingModel.

Initializes the backbone from the Stage 1 classification checkpoint, then trains
with Supervised Contrastive Loss (SupConLoss) and a PKSampler (P images per K
classes per batch). Validates with a linear probe fit on a subset of the TRAIN
split and evaluated on the held-out validation split. Saves the best checkpoint
by validation accuracy.

Runnable directly (uses the defaults below) or callable from `main.py` with
explicit arguments.
"""

import random
from pathlib import Path
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from model import ArtStyleEmbeddingModel
from dataset import create_train_val_split
from utils import (
    SupConLoss,
    PKSampler,
    build_class_to_idx,
    load_backbone_weights,
    load_checkpoint,
    save_embedding_model,
)


# --- Default configuration ---
DATASET_NAME = "huggan/wikiart"
LABEL_NAME = "style"
BACKBONE_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
EMBEDDING_DIM = 256
EPOCHS = 3
P = 10
K = 8
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
TEMPERATURE = 0.07
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
VAL_RATIO = 0.1
PROBE_SIZE = 4096         # train images used to fit the linear probe
PROBE_BATCH_SIZE = 256
PROBE_LR = 0.1
PROBE_STEPS = 50
OUTPUT_DIR = "checkpoints"
INIT_CHECKPOINT = "checkpoints/stage1_classification_WikiArt.pt"
CHECKPOINT_NAME = "stage2_contrastive_WikiArt.pt"
RESUME_FROM = None  # optional path to a Stage 2 checkpoint to resume from
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def extract_embeddings(model, loader, device):
    """Return (embeddings, labels) for every sample in `loader`, in order."""
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            embeddings.append(model(images).cpu())
            labels.extend(int(v) for v in y)
    return torch.cat(embeddings, dim=0), labels


def validate(
    model,
    probe_loader,
    val_loader,
    device,
    num_classes,
    class_to_idx,
    probe_lr=PROBE_LR,
    probe_steps=PROBE_STEPS,
):
    """
    Linear-probe validation.

    1. Extract frozen embeddings for a subset of the TRAIN split (probe_loader).
    2. Fit a linear classifier on those embeddings.
    3. Evaluate it on the held-out VAL split (val_loader).

    Fitting the probe on the train subset (rather than on the val set) keeps the
    probe fit and its evaluation disjoint, so the reported accuracy is not
    inflated.

    Args:
        model: The ArtStyleEmbeddingModel.
        probe_loader: DataLoader over a train subset used to fit the probe.
        val_loader: DataLoader over the held-out validation split.
        device: Target device.
        num_classes: Number of classes.
        class_to_idx: Mapping from class label to integer index.
        probe_lr: Learning rate for the linear probe.
        probe_steps: Number of optimization steps for the linear probe.

    Returns:
        Validation accuracy (float).
    """
    probe_emb, probe_labels = extract_embeddings(model, probe_loader, device)
    val_emb, val_labels = extract_embeddings(model, val_loader, device)

    probe_y = torch.tensor([class_to_idx[label] for label in probe_labels], dtype=torch.long)
    val_y = torch.tensor([class_to_idx[label] for label in val_labels], dtype=torch.long)

    # Fit a simple linear classifier (logistic regression via SGD)
    clf = nn.Linear(probe_emb.size(1), num_classes)
    opt = torch.optim.SGD(clf.parameters(), lr=probe_lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    clf.train()
    for _ in range(probe_steps):
        opt.zero_grad()
        loss = criterion(clf(probe_emb), probe_y)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        preds = clf(val_emb).argmax(dim=1)
        accuracy = (preds == val_y).float().mean().item()

    return accuracy


def main(
    dataset_name: str = DATASET_NAME,
    label_name: str = LABEL_NAME,
    backbone_name: str = BACKBONE_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    epochs: int = EPOCHS,
    p: int = P,
    k: int = K,
    backbone_lr: float = BACKBONE_LR,
    head_lr: float = HEAD_LR,
    temperature: float = TEMPERATURE,
    weight_decay: float = WEIGHT_DECAY,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    num_workers: int = NUM_WORKERS,
    val_ratio: float = VAL_RATIO,
    probe_size: int = PROBE_SIZE,
    probe_batch_size: int = PROBE_BATCH_SIZE,
    probe_lr: float = PROBE_LR,
    probe_steps: int = PROBE_STEPS,
    output_dir: str = OUTPUT_DIR,
    init_checkpoint: str | None = INIT_CHECKPOINT,
    checkpoint_name: str = CHECKPOINT_NAME,
    resume_from: str | None = RESUME_FROM,
    device: str = DEVICE,
    seed: int = SEED,
) -> Path:
    """Run Stage 2 training and return the path of the best checkpoint."""
    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_device = torch.device(device)
    print(f"Device: {torch_device}")
    print(f"Dataset: {dataset_name}, Label: {label_name}")
    print(f"Backbone: {backbone_name}, Embedding dim: {embedding_dim}")
    print(f"Batch: P={p} x K={k} = {p * k}")
    print(f"Epochs: {epochs}, LR: backbone={backbone_lr}, head={head_lr}")

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Resolve the backbone's preprocessing config without building a throwaway model
    data_config = timm.data.resolve_model_data_config(backbone_name)

    # Load the dataset once; the same seed gives Stage 1 the same train/val boundary.
    train_dataset, val_dataset = create_train_val_split(
        dataset_name=dataset_name,
        data_config=data_config,
        label_name=label_name,
        val_ratio=val_ratio,
        seed=seed,
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Build class mapping from train + val (shared indexing scheme with Stage 1)
    all_labels = set(train_dataset.hf_dataset[train_dataset.label_name]) | set(
        val_dataset.hf_dataset[val_dataset.label_name]
    )
    class_to_idx = build_class_to_idx(all_labels)
    num_classes = len(class_to_idx)
    print(f"Num classes: {num_classes}")

    # Create model
    model = ArtStyleEmbeddingModel(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=True,
    )
    model.to(torch_device)
    print(f"Model created. Data config: {list(model.data_config.keys())}")

    # Stage 1 -> Stage 2 handoff (skipped when resuming a Stage 2 run)
    resuming = resume_from is not None and Path(resume_from).exists()
    if not resuming and init_checkpoint is not None:
        if not Path(init_checkpoint).exists():
            raise FileNotFoundError(
                f"INIT_CHECKPOINT '{init_checkpoint}' not found. Run "
                "train_classification.py first, or set INIT_CHECKPOINT = None "
                "to train from the pretrained backbone."
            )
        info = load_backbone_weights(model, init_checkpoint, device=str(torch_device))
        print(
            f"Initialized backbone from {init_checkpoint} "
            f"({info['loaded_keys']} keys, missing={len(info['missing_keys'])}, "
            f"unexpected={len(info['unexpected_keys'])})"
        )

    # Create train sampler (P images per K classes)
    train_sampler = PKSampler(train_dataset, P=p, K=k, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=p * k,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Fixed random subset of the train split used to fit the linear probe
    probe_count = min(probe_size, len(train_dataset))
    generator = torch.Generator().manual_seed(seed)
    probe_indices = torch.randperm(
        len(train_dataset), generator=generator
    )[:probe_count].tolist()
    probe_loader = DataLoader(
        Subset(train_dataset, probe_indices),
        batch_size=probe_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create val loader (standard, no special sampling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create optimizer with separate LRs (weight_decay applies only to decay groups)
    param_groups = model.get_param_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
    )
    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    # Scheduler (cosine annealing per epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision
    use_amp = torch_device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Create loss
    criterion = SupConLoss(temperature=temperature)

    checkpoint_path = output_dir_path / checkpoint_name

    # Resume an interrupted run (restores model, optimizer, scheduler and epoch)
    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0
    if resuming:
        checkpoint = load_checkpoint(resume_from, torch_device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0)
        best_acc = checkpoint.get("metrics", {}).get("val_acc", 0.0)
        best_epoch = start_epoch
        print(
            f"Resumed from {resume_from} at epoch {start_epoch} "
            f"(best val acc {best_acc:.4f})"
        )

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images = images.to(torch_device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(torch_device.type, enabled=use_amp):
                embeddings = model(images)

            # Compute the loss in fp32 for numerical stability
            loss = criterion(embeddings.float(), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        # Validate
        val_acc = validate(
            model,
            probe_loader,
            val_loader,
            torch_device,
            num_classes,
            class_to_idx,
            probe_lr=probe_lr,
            probe_steps=probe_steps,
        )

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            save_embedding_model(
                model,
                checkpoint_path,
                backbone_name=backbone_name,
                embedding_dim=embedding_dim,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={"val_acc": val_acc},
            )
            print(f"  -> Saved best model (epoch {best_epoch}, acc {val_acc:.4f})")

    # Final summary
    print(f"\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    main()

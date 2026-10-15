import random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn


class PKSampler(torch.utils.data.Sampler):
    """
    A sampler for metric learning with the contrastive loss.

    Each batch contains exactly P images from each of K different classes.
    Batch size = P * K.

    Args:
        dataset: The dataset to sample from (ArtStyleDataset).
        P: Number of images per class in a batch.
        K: Number of different classes in a batch.
        labels: Optional sequence of labels aligned with the dataset. If not
            provided, labels are read from the dataset's HuggingFace backing
            store via `dataset.hf_dataset[dataset.label_name]`.
        seed: Optional seed for reproducible sampling. If None, sampling is
            non-deterministic across runs.
    """

    def __init__(self, dataset, P, K, labels=None, seed=None):
        if P < 1 or K < 1:
            raise ValueError(f"P and K must be >= 1, got P={P}, K={K}")

        self.dataset = dataset
        self.P = P

        # Read labels from the dataset unless explicitly provided
        if labels is None:
            labels = dataset.hf_dataset[dataset.label_name]
        self.labels = list(labels)

        # Build class-to-indices mapping
        self.class_to_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.class_to_indices[label].append(i)

        # Only classes with at least P samples can contribute a full group
        self.eligible_classes = [
            cls for cls, idx in self.class_to_indices.items() if len(idx) >= P
        ]
        if len(self.eligible_classes) < K:
            raise ValueError(
                f"K={K} exceeds the number of classes with at least P={P} "
                f"samples ({len(self.eligible_classes)})."
            )

        self.K = K

        # Each batch is P * K samples; drop any leftover samples that do not
        # fill a complete batch.
        self.num_batches = len(dataset) // (self.P * self.K)
        self.num_samples = self.num_batches * (self.P * self.K)

        self._rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            # Select K different classes
            classes = self._rng.sample(self.eligible_classes, self.K)
            # For each class, select P indices
            for cls in classes:
                yield from self._rng.sample(self.class_to_indices[cls], self.P)

    def __len__(self):
        return self.num_samples


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (from "Supervised Contrastive Learning", Khosla et al., 2020).

    For each anchor, pulls its embedding toward same-class embeddings (positives)
    and away from different-class embeddings (negatives).

    Args:
        temperature: Scaling factor for similarity (default: 0.07).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Ensure labels live on the same device as the embeddings
        labels = labels.to(embeddings.device)

        N = embeddings.size(0)

        # N x N similarity matrix, scaled by temperature
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Numerical stability: subtract the per-row max (detached)
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

        # Mask out self-similarity (diagonal), both as a positive and in the denominator
        logits_mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)

        # Label mask: True where labels match, excluding self-similarity
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.t()) & logits_mask

        # log-softmax over the masked logits
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Mean log probability of positives for each anchor
        pos_count = label_mask.sum(1)
        mean_log_prob_pos = (label_mask * log_prob).sum(1) / pos_count.clamp(min=1)

        # Only compute loss for anchors with at least one positive
        valid = pos_count > 0
        loss = -mean_log_prob_pos[valid].mean()

        return loss


def build_class_to_idx(labels):
    """
    Build a label -> contiguous index mapping.

    Assumes labels are integers 0..K-1 (as produced by HuggingFace ClassLabel
    columns), which CrossEntropyLoss targets require. Shared by both training
    stages so the same indexing scheme is used everywhere.

    Args:
        labels: Sequence of labels.

    Returns:
        A dict mapping each label to a contiguous integer index.

    Raises:
        ValueError: If labels are not contiguous integers starting at 0.
    """
    unique = sorted(set(labels))
    if unique != list(range(len(unique))):
        raise ValueError(
            f"Labels must be contiguous integers starting at 0; got {unique[:10]}"
            f"{'...' if len(unique) > 10 else ''}."
        )
    return {label: idx for idx, label in enumerate(unique)}


def _infer_backbone_name(backbone):
    """
    Best-effort recovery of the timm model name from a backbone instance.

    timm does not store the original `create_model` name directly, so this
    reconstructs it from the resolved pretrained config:
    - prefer the HuggingFace hub id (e.g. "timm/convnextv2_tiny.fcmae_ft_in22k_in1k"),
    - otherwise combine the "architecture" and "tag" fields.

    Args:
        backbone: The timm backbone module (e.g. `model.backbone`).

    Returns:
        The inferred timm model name.

    Raises:
        ValueError: If the name cannot be inferred from the backbone config.
    """
    cfg = getattr(backbone, "default_cfg", None) or {}

    hf_hub_id = cfg.get("hf_hub_id")
    if hf_hub_id and "/" in hf_hub_id:
        return hf_hub_id.split("/", 1)[1]

    architecture = cfg.get("architecture")
    tag = cfg.get("tag")
    if architecture:
        return f"{architecture}.{tag}" if tag else architecture

    raise ValueError(
        "Could not infer the backbone name from the model; pass it explicitly "
        "via the `backbone_name` argument."
    )


CHECKPOINT_FORMAT_VERSION = 1


def load_checkpoint(checkpoint_path, device="cpu"):
    """
    Load a raw checkpoint dict from disk and validate its format version.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to map tensors onto during load.

    Returns:
        The checkpoint dict.

    Raises:
        ValueError: If the checkpoint's `format_version` does not match
            `CHECKPOINT_FORMAT_VERSION`.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has format_version={version!r}, "
            f"but this code expects {CHECKPOINT_FORMAT_VERSION}."
        )

    return checkpoint


def save_checkpoint(
    model,
    checkpoint_path,
    stage,
    backbone_name=None,
    embedding_dim=None,
    num_classes=None,
    optimizer=None,
    scheduler=None,
    epoch=None,
    metrics=None,
):
    """
    Save a model checkpoint in the shared project format.

    All stages write the same schema so any checkpoint can be inspected and its
    backbone transferred into a later-stage model:

        {
            "format_version": int,
            "stage": "classification" | "contrastive",
            "backbone_name": str,
            "model_state": state_dict,
            "embedding_dim": int,          # embedding models only
            "num_classes": int,            # classification only
            "optimizer_state": dict,       # optional
            "scheduler_state": dict,       # optional
            "epoch": int,                  # optional
            "metrics": dict,               # optional
        }

    Args:
        model: The ArtStyleEmbeddingModel instance.
        checkpoint_path: Path to save the checkpoint file.
        stage: Which training stage produced this checkpoint.
        backbone_name: Optional timm backbone name (inferred if not provided).
        embedding_dim: Optional embedding dimension (contrastive models).
        num_classes: Optional number of classes (classification models).
        optimizer: Optional optimizer whose state should be saved.
        scheduler: Optional LR scheduler whose state should be saved.
        epoch: Optional epoch number.
        metrics: Optional dict of metrics (e.g. {"val_acc": 0.9}).

    Returns:
        The checkpoint dict that was written.
    """
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "backbone_name": (
            backbone_name if backbone_name is not None
            else _infer_backbone_name(model.backbone)
        ),
        "model_state": model.state_dict(),
    }

    if embedding_dim is not None:
        checkpoint["embedding_dim"] = embedding_dim
    if num_classes is not None:
        checkpoint["num_classes"] = num_classes
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = dict(metrics)

    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def save_embedding_model(
    model,
    checkpoint_path,
    backbone_name=None,
    embedding_dim=None,
    optimizer=None,
    scheduler=None,
    epoch=None,
    metrics=None,
):
    """
    Save the ArtStyleEmbeddingModel as a contrastive-stage checkpoint.

    Thin wrapper around `save_checkpoint` that infers `embedding_dim` from the
    model's head when not provided.

    Args:
        model: The ArtStyleEmbeddingModel instance.
        checkpoint_path: Path to save the checkpoint file.
        backbone_name: Optional timm backbone name. Inferred from the model's
            backbone config if not provided.
        embedding_dim: Optional embedding dimension. Inferred from the model's
            head if not provided.
        optimizer: Optional optimizer whose state should be saved.
        scheduler: Optional LR scheduler whose state should be saved.
        epoch: Optional epoch number.
        metrics: Optional dict of metrics.
    """
    if embedding_dim is None:
        embedding_dim = model.head.out_features

    return save_checkpoint(
        model,
        checkpoint_path,
        stage="contrastive",
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=metrics,
    )


def load_embedding_model(checkpoint_path, device="cpu"):
    """
    Load the ArtStyleEmbeddingModel from a checkpoint.

    Reconstructs the model using the saved constructor arguments, loads the
    state dict, and returns it in eval mode. This function builds and returns a
    new model; it does not populate an existing one.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Target device (e.g. "cpu", "cuda", "cuda:0").

    Returns:
        The loaded ArtStyleEmbeddingModel in eval mode.
    """
    from model import ArtStyleEmbeddingModel

    checkpoint = load_checkpoint(checkpoint_path, device)

    required_keys = ("model_state", "backbone_name", "embedding_dim")
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        stage = checkpoint.get("stage", "unknown")
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' is missing required key(s): {missing} "
            f"(stage={stage!r}). Found keys: {sorted(checkpoint.keys())}. "
            "Only contrastive-stage checkpoints can be loaded as embedding models; "
            "a classification checkpoint has no 'embedding_dim'."
        )

    model = ArtStyleEmbeddingModel(
        backbone_name=checkpoint["backbone_name"],
        embedding_dim=checkpoint["embedding_dim"],
        pretrained=False,  # Loading saved weights, no need for pretrained init
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model


def load_backbone_weights(model, checkpoint_path, device="cpu", strict=True):
    """
    Transfer only the backbone weights from a checkpoint into a model.

    This is the Stage 1 -> Stage 2 handoff: a classification checkpoint's head
    has `num_classes` outputs and cannot be loaded as an embedding model, but
    its backbone is reusable. The saved `"backbone.*"` tensors are stripped of
    their prefix and loaded into `model.backbone`; the head is left untouched.

    Args:
        model: The target ArtStyleEmbeddingModel (head weights are not modified).
        checkpoint_path: Path to a checkpoint saved in the shared format.
        device: Device to map checkpoint tensors onto during load.
        strict: If True, raise if any backbone parameter is missing or unexpected.

    Returns:
        A dict with the number of loaded keys and any missing/unexpected keys.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)

    state = checkpoint["model_state"]
    prefix = "backbone."
    backbone_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' contains no '{prefix}*' parameters; "
            "cannot transfer backbone weights."
        )

    result = model.backbone.load_state_dict(backbone_state, strict=False)

    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            "Backbone transfer failed to match exactly. "
            f"Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}."
        )

    return {
        "loaded_keys": len(backbone_state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }

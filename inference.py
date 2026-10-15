"""
inference.py
Inference script for art style recognition using DBSCAN clustering.

Loads the trained model, extracts embeddings for the input images,
and uses DBSCAN to group them into clusters of similar art style.
Only the pairwise distances between embeddings drive the clustering —
no reference images or predefined classes are used.

Runnable directly (uses the defaults below) or callable from `main.py` with
explicit arguments.
"""

import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from utils import load_embedding_model

# ============================================================
# Default configuration
# ============================================================
CHECKPOINT = "checkpoints/stage2_contrastive_WikiArt.pt"
INPUT_IMAGES = "images_test/input_images/"  # Folder path, single file path, or list of paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DBSCAN parameters
EPS = 0.2       # Max distance between samples in the same cluster
MIN_SAMPLES = 3  # Min samples to form a dense region
# ============================================================


def load_image(path, data_config):
    """
    Load and preprocess a single image into a model-ready tensor.

    Args:
        path: File path or http(s) URL.
        data_config: Dict from model.data_config.

    Returns:
        Tensor of shape (1, C, H, W).
    """
    if path.startswith("http://") or path.startswith("https://"):
        from urllib.request import urlretrieve
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            urlretrieve(path, tmp.name)
            img = Image.open(tmp.name).convert("RGB")
        os.unlink(tmp.name)
    else:
        img = Image.open(path).convert("RGB")

    # Match the training-time preprocessing exactly (resize/crop/normalization)
    transform = timm.data.create_transform(is_training=False, **data_config)
    return transform(img).unsqueeze(0)  # (1, C, H, W)


def load_images(input_path, data_config):
    """
    Load multiple images from a folder path, single file, or list of paths.

    Each entry may be a file, a directory (all valid images inside are used),
    or an http(s) URL.

    Args:
        input_path: A string/path or a list of strings/paths.
        data_config: Dict from model.data_config.

    Returns:
        List of (name, tensor) tuples.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if isinstance(input_path, (str, Path)):
        entries = [input_path]
    elif isinstance(input_path, list):
        entries = input_path
    else:
        raise TypeError(f"Unsupported input_path type: {type(input_path)}")

    paths = []
    for entry in entries:
        if isinstance(entry, str) and entry.startswith(("http://", "https://")):
            paths.append(entry)
            continue
        p = Path(entry)
        if p.is_dir():
            paths.extend(
                sorted(f for f in p.iterdir() if f.suffix.lower() in valid_extensions)
            )
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"Path not found: {entry}")

    images = []
    for path in paths:
        name = Path(path).stem
        tensor = load_image(str(path), data_config)
        images.append((name, tensor))

    return images


def cluster_images(embeddings, eps=EPS, min_samples=MIN_SAMPLES):
    """
    Use DBSCAN to cluster input image embeddings.

    Args:
        embeddings: Tensor of shape (N, D).
        eps: Max distance between samples in the same cluster.
        min_samples: Min samples to form a dense region.

    Returns:
        Labels array of shape (N,). -1 denotes noise.
    """
    emb_np = embeddings.numpy()
    # Cosine distance matrix: (N, N)
    dist_matrix = cosine_distances(emb_np)
    # DBSCAN with precomputed distance
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    return labels


def main(
    checkpoint: str = CHECKPOINT,
    input_images=INPUT_IMAGES,
    device: str = DEVICE,
    eps: float = EPS,
    min_samples: int = MIN_SAMPLES,
    output: str | None = None,
) -> dict:
    """
    Cluster the input images by embedding similarity.

    Args:
        checkpoint: Path to a contrastive-stage checkpoint.
        input_images: Folder path, single file path, or list of paths/URLs.
        device: Device to run the model on.
        eps: DBSCAN epsilon (max cosine distance within a cluster).
        min_samples: DBSCAN minimum points to form a dense region.
        output: Optional path to write a JSON image-name -> cluster mapping.

    Returns:
        A dict with the clustering result:
        {
            "checkpoint": str,
            "eps": float,
            "min_samples": int,
            "num_clusters": int,
            "num_noise": int,
            "assignments": {image_name: cluster_id, ...},  # -1 = noise
        }
    """
    torch_device = torch.device(device)
    print(f"Device: {torch_device}")

    # Load model (rebuilt from the checkpoint's saved config)
    print(f"Loading model from {checkpoint}...")
    model = load_embedding_model(checkpoint, device)
    data_config = model.data_config
    print(f"Model loaded.")

    # Load input images
    print(f"Loading input images from {input_images}...")
    images = load_images(input_images, data_config)
    print(f"Loaded {len(images)} images.")

    # Extract embeddings for input images (batched for speed)
    batch_size = 32
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.cat(
                [tensor for _, tensor in images[start:start + batch_size]], dim=0
            ).to(torch_device)
            all_embeddings.append(model(batch).cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)

    # Cluster using DBSCAN
    print(f"\nClustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
    labels = cluster_images(embeddings, eps=eps, min_samples=min_samples)

    # Get unique cluster IDs (exclude -1 noise)
    unique_labels = set(labels) - {-1}
    num_clusters = len(unique_labels)
    num_noise = list(labels).count(-1)
    print(f"Found {num_clusters} clusters, {num_noise} noise points.")

    # Report each cluster
    for cluster_id in sorted(unique_labels):
        mask = np.array(labels) == cluster_id
        cluster_names = [images[i][0] for i in range(len(images)) if mask[i]]

        print(f"\n{'='*60}")
        print(f"Cluster {cluster_id} ({len(cluster_names)} images):")
        print(f"  Images: {cluster_names}")

    # Report noise points
    if num_noise > 0:
        noise_mask = np.array(labels) == -1
        noise_names = [images[i][0] for i in range(len(images)) if noise_mask[i]]
        print(f"\n{'='*60}")
        print(f"Noise points ({num_noise} images):")
        print(f"  Images: {noise_names}")

    result = {
        "checkpoint": str(checkpoint),
        "eps": eps,
        "min_samples": min_samples,
        "num_clusters": num_clusters,
        "num_noise": num_noise,
        "assignments": {
            images[i][0]: int(labels[i]) for i in range(len(images))
        },
    }

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nWrote clustering result to {output_path}")

    return result


if __name__ == "__main__":
    main()

"""
dataset.py
Defines the art style dataset:
- ArtStyleDataset: thin wrapper around a HuggingFace datasets.Dataset
- create_dataset: factory that loads from HF, splits train/val, creates transforms
"""

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from datasets import load_dataset
import timm
from typing import Any
from PIL import Image


class ArtStyleDataset(Dataset):
    """
    Thin wrapper around a HuggingFace datasets.Dataset.

    Returns (image_tensor, label) pairs where label is determined by label_name.

    Args:
        hf_dataset: A HuggingFace datasets.Dataset object.
        transform: A callable that transforms a PIL Image to a tensor.
        label_name: Column name to use as label (e.g., "style" or "artist").
    """

    def __init__(self, hf_dataset: HFDataset, transform, label_name: str) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.label_name = label_name

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        item = self.hf_dataset[idx]

        # Get image — HF datasets may return PIL Image, numpy array, or dict
        image = item["image"]
        if isinstance(image, dict):
            image = Image.open(image["path"])
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Ensure a 3-channel RGB image for the timm transform
        image = image.convert("RGB")

        # Apply transform
        image_tensor = self.transform(image)

        # Get label
        label = item[self.label_name]

        return image_tensor, label


def create_dataset(
    dataset_name: str,
    split: str,
    data_config: dict[str, Any],
    label_name: str = "style",
    **load_kwargs,
) -> ArtStyleDataset:
    """
    Factory function that loads a HuggingFace dataset, applies the requested split,
    creates the appropriate transform, and returns an ArtStyleDataset.

    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Which split to use ("train", "validation", "test").
        data_config: Data config from model (for creating transforms).
        label_name: Column name to use as label.
        **load_kwargs: Additional keyword arguments for load_dataset.

    Returns:
        An ArtStyleDataset instance.
    """
    # Load dataset from HuggingFace
    ds = load_dataset(dataset_name, **load_kwargs)

    # Get the requested split
    hf_split: HFDataset = ds[split]

    # Create transform
    is_training = (split == "train")
    transform = timm.data.create_transform(is_training=is_training, **data_config)

    return ArtStyleDataset(hf_split, transform, label_name)


def create_train_val_split(
    dataset_name: str,
    data_config: dict[str, Any],
    label_name: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    **load_kwargs,
) -> tuple[ArtStyleDataset, ArtStyleDataset]:
    """
    Load a dataset once and return (train, val) ArtStyleDataset pairs.

    Uses an existing "validation"/"val"/"test" split when the dataset provides
    one; otherwise holds out `val_ratio` of the "train" split. The held-out set
    is deterministic for a fixed `seed`, so every stage shares the same
    train/val boundary and the dataset is loaded only once.

    Args:
        dataset_name: Name of the HuggingFace dataset.
        data_config: Data config from model (for creating transforms).
        label_name: Column name to use as label.
        val_ratio: Fraction of the train split held out when no split exists.
        seed: Seed for the deterministic train/val carve.
        **load_kwargs: Additional keyword arguments for load_dataset.

    Returns:
        A (train_dataset, val_dataset) tuple.
    """
    ds = load_dataset(dataset_name, **load_kwargs)

    train_transform = timm.data.create_transform(is_training=True, **data_config)
    val_transform = timm.data.create_transform(is_training=False, **data_config)

    for split_name in ("validation", "val", "test"):
        if split_name in ds:
            return (
                ArtStyleDataset(ds["train"], train_transform, label_name),
                ArtStyleDataset(ds[split_name], val_transform, label_name),
            )

    split = ds["train"].train_test_split(test_size=val_ratio, seed=seed)
    return (
        ArtStyleDataset(split["train"], train_transform, label_name),
        ArtStyleDataset(split["test"], val_transform, label_name),
    )


if __name__ == "__main__":
    # Test with the WikiArt dataset from HuggingFace
    test_dataset_name = "huggan/wikiart"

    # Minimal data_config for testing (matches ConvNeXtV2-Tiny defaults)
    test_data_config = {
        "input_size": (3, 224, 224),
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "crop_pct": 0.875,
        "crop_mode": "squash",
    }

    # Create train dataset
    train_ds = create_dataset(
        dataset_name=test_dataset_name,
        split="train",
        data_config=test_data_config,
        label_name="style",
    )

    print(f"Train dataset size: {len(train_ds)}")

    # Get a sample
    image, label = train_ds[0]
    print(f"Image shape: {image.shape}")
    print(f"Label (style): {label}")

    # Verify image is a tensor with expected shape
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    assert image.shape == (3, 224, 224), f"Expected (3, 224, 224), got {image.shape}"

    print("All tests passed.")

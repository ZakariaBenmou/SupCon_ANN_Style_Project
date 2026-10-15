"""
main.py
Command hub for the art-style recognition pipeline.

Centralizes every CLI command and delegates to the training / inference scripts,
so the individual modules stay runnable on their own.

Usage:
    venv/bin/python main.py <command> [options]

Commands:
    classification (stage1, train1)  Stage 1: classification pretraining
    contrastive    (stage2, train2)  Stage 2: supervised contrastive training
    pipeline       (all, train)      Run Stage 1 then Stage 2
    infer          (cluster)         Cluster images by embedding similarity
    inspect                          Print checkpoint metadata
    list-checkpoints (list)          List checkpoints in a directory
    env            (info)            Show environment / version info
    test           (smoke)           Run the module smoke tests
    download-data                    Pre-fetch the HuggingFace dataset

Run `venv/bin/python main.py <command> --help` for command-specific options.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import train_classification
import train_contrastive
import inference
from utils import load_checkpoint


BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_config(title: str, config: dict) -> None:
    print(f"{title}:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def _dry_run(args, title: str, config: dict) -> bool:
    """Print the resolved config; return True when a dry-run should skip execution."""
    _print_config(title, config)
    if getattr(args, "dry_run", False):
        print("  (dry-run: nothing executed)")
        return True
    return False


def _add_training_common(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by the classification / contrastive / pipeline commands."""
    parser.add_argument("--dataset", default=train_classification.DATASET_NAME,
                        help="HuggingFace dataset name")
    parser.add_argument("--label", default=train_classification.LABEL_NAME,
                        help="label column name")
    parser.add_argument("--backbone", default=train_classification.BACKBONE_NAME,
                        help="timm backbone name")
    parser.add_argument("--device", default=train_classification.DEVICE,
                        help="cuda / cpu / cuda:0 (default: auto)")
    parser.add_argument("--seed", type=int, default=train_classification.SEED)
    parser.add_argument("--output-dir", default=train_classification.OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=train_classification.NUM_WORKERS)
    parser.add_argument("--val-ratio", type=float, default=train_classification.VAL_RATIO)
    parser.add_argument("--lr-backbone", type=float, default=train_classification.BACKBONE_LR)
    parser.add_argument("--lr-head", type=float, default=train_classification.HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=train_classification.WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=train_classification.GRAD_CLIP_NORM)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved config and exit without running")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_classification(args) -> Path:
    config = {
        "dataset_name": args.dataset,
        "label_name": args.label,
        "backbone_name": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "backbone_lr": args.lr_backbone,
        "head_lr": args.lr_head,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip,
        "val_ratio": args.val_ratio,
        "output_dir": args.output_dir,
        "checkpoint_name": args.checkpoint_name,
        "resume_from": args.resume,
        "device": args.device,
        "seed": args.seed,
    }
    if _dry_run(args, "Stage 1 classification", config):
        return Path(args.output_dir) / args.checkpoint_name
    return train_classification.main(**config)


def cmd_contrastive(args) -> Path:
    config = {
        "dataset_name": args.dataset,
        "label_name": args.label,
        "backbone_name": args.backbone,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "p": args.p,
        "k": args.k,
        "backbone_lr": args.lr_backbone,
        "head_lr": args.lr_head,
        "temperature": args.temperature,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip,
        "num_workers": args.workers,
        "val_ratio": args.val_ratio,
        "probe_size": args.probe_size,
        "probe_batch_size": args.probe_batch_size,
        "probe_lr": args.probe_lr,
        "probe_steps": args.probe_steps,
        "output_dir": args.output_dir,
        "init_checkpoint": args.init_checkpoint,
        "checkpoint_name": args.checkpoint_name,
        "resume_from": args.resume,
        "device": args.device,
        "seed": args.seed,
    }
    if _dry_run(args, "Stage 2 contrastive", config):
        return Path(args.output_dir) / args.checkpoint_name
    return train_contrastive.main(**config)


def cmd_pipeline(args):
    stage1_config = {
        "dataset_name": args.dataset,
        "label_name": args.label,
        "backbone_name": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "backbone_lr": args.lr_backbone,
        "head_lr": args.lr_head,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip,
        "val_ratio": args.val_ratio,
        "output_dir": args.output_dir,
        "checkpoint_name": args.stage1_checkpoint,
        "device": args.device,
        "seed": args.seed,
    }
    stage2_config = {
        "dataset_name": args.dataset,
        "label_name": args.label,
        "backbone_name": args.backbone,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "p": args.p,
        "k": args.k,
        "backbone_lr": args.lr_backbone,
        "head_lr": args.lr_head,
        "temperature": args.temperature,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip,
        "num_workers": args.workers,
        "val_ratio": args.val_ratio,
        "probe_size": args.probe_size,
        "probe_batch_size": args.probe_batch_size,
        "probe_lr": args.probe_lr,
        "probe_steps": args.probe_steps,
        "output_dir": args.output_dir,
        "checkpoint_name": args.stage2_checkpoint,
        "device": args.device,
        "seed": args.seed,
    }

    if getattr(args, "dry_run", False):
        _print_config("Stage 1 classification", stage1_config)
        _print_config("Stage 2 contrastive", stage2_config)
        print("  (dry-run: nothing executed)")
        return None

    print("=== Stage 1: classification ===")
    stage1_path = train_classification.main(**stage1_config)
    print(f"Stage 1 checkpoint: {stage1_path}")

    if args.stop_after == "stage1":
        return stage1_path

    print("\n=== Stage 2: contrastive ===")
    stage2_config["init_checkpoint"] = str(stage1_path)
    stage2_path = train_contrastive.main(**stage2_config)
    print(f"Stage 2 checkpoint: {stage2_path}")
    return stage2_path


def cmd_inference(args) -> dict:
    input_images = args.input if args.input else inference.INPUT_IMAGES
    config = {
        "checkpoint": args.checkpoint,
        "input_images": input_images,
        "device": args.device,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "output": args.output,
    }
    if _dry_run(args, "Inference (DBSCAN clustering)", config):
        return {}
    return inference.main(**config)


def cmd_inspect(args) -> dict:
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    info = {
        "path": str(args.checkpoint),
        "format_version": checkpoint.get("format_version"),
        "stage": checkpoint.get("stage"),
        "backbone_name": checkpoint.get("backbone_name"),
        "embedding_dim": checkpoint.get("embedding_dim"),
        "num_classes": checkpoint.get("num_classes"),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "has_optimizer_state": "optimizer_state" in checkpoint,
        "has_scheduler_state": "scheduler_state" in checkpoint,
        "num_model_state_keys": len(checkpoint.get("model_state", {})),
    }
    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        _print_config(f"Checkpoint: {args.checkpoint}", info)
    return info


def cmd_list_checkpoints(args) -> list:
    directory = Path(args.dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    files = sorted(directory.glob("*.pt"))
    if not files:
        print(f"No .pt checkpoints in {directory}")
        return files

    for path in files:
        size_mb = path.stat().st_size / (1024 * 1024)
        line = f"{path.name}  ({size_mb:.1f} MB)"
        if args.metadata:
            try:
                checkpoint = load_checkpoint(path, "cpu")
                line += (
                    f"  stage={checkpoint.get('stage')}"
                    f"  epoch={checkpoint.get('epoch')}"
                    f"  metrics={checkpoint.get('metrics')}"
                )
            except Exception as error:  # noqa: BLE001 - report and continue
                line += f"  (unreadable: {error})"
        print(line)
    return files


def cmd_env(args) -> dict:
    import platform

    import numpy
    import PIL
    import sklearn
    import timm
    import torch
    import torchvision

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "scikit-learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    _print_config("Environment", info)
    return info


def cmd_test(args) -> list:
    scripts = []
    if args.which in ("model", "all"):
        scripts.append("model.py")
    if args.which in ("dataset", "all"):
        scripts.append("dataset.py")

    for script in scripts:
        print(f"=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, str(BASE_DIR / script)], check=True, cwd=str(BASE_DIR))
    return scripts


def cmd_download_data(args) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset)
    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)}")
    return dataset


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Command hub for the art-style recognition pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    # classification -------------------------------------------------------
    p = subparsers.add_parser(
        "classification", aliases=["stage1", "train1"],
        help="Stage 1: classification pretraining",
    )
    _add_training_common(p)
    p.add_argument("--epochs", type=int, default=train_classification.EPOCHS)
    p.add_argument("--batch-size", type=int, default=train_classification.BATCH_SIZE)
    p.add_argument("--checkpoint-name", default=train_classification.CHECKPOINT_NAME)
    p.add_argument("--resume", default=train_classification.RESUME_FROM,
                   help="path to a Stage 1 checkpoint to resume from")
    p.set_defaults(handler=cmd_classification)

    # contrastive ----------------------------------------------------------
    p = subparsers.add_parser(
        "contrastive", aliases=["stage2", "train2"],
        help="Stage 2: supervised contrastive training",
    )
    _add_training_common(p)
    p.add_argument("--epochs", type=int, default=train_contrastive.EPOCHS)
    p.add_argument("--embedding-dim", type=int, default=train_contrastive.EMBEDDING_DIM)
    p.add_argument("--p", type=int, default=train_contrastive.P,
                   help="images per class in a batch")
    p.add_argument("--k", type=int, default=train_contrastive.K,
                   help="classes per batch")
    p.add_argument("--temperature", type=float, default=train_contrastive.TEMPERATURE)
    p.add_argument("--probe-size", type=int, default=train_contrastive.PROBE_SIZE)
    p.add_argument("--probe-batch-size", type=int, default=train_contrastive.PROBE_BATCH_SIZE)
    p.add_argument("--probe-lr", type=float, default=train_contrastive.PROBE_LR)
    p.add_argument("--probe-steps", type=int, default=train_contrastive.PROBE_STEPS)
    p.add_argument("--init-checkpoint", default=train_contrastive.INIT_CHECKPOINT)
    p.add_argument("--checkpoint-name", default=train_contrastive.CHECKPOINT_NAME)
    p.add_argument("--resume", default=train_contrastive.RESUME_FROM,
                   help="path to a Stage 2 checkpoint to resume from")
    p.set_defaults(handler=cmd_contrastive)

    # pipeline -------------------------------------------------------------
    p = subparsers.add_parser(
        "pipeline", aliases=["all", "train"],
        help="Run Stage 1 then Stage 2 (auto-wires the Stage 1 checkpoint)",
    )
    _add_training_common(p)
    p.add_argument("--epochs", type=int, default=train_classification.EPOCHS)
    p.add_argument("--batch-size", type=int, default=train_classification.BATCH_SIZE)
    p.add_argument("--embedding-dim", type=int, default=train_contrastive.EMBEDDING_DIM)
    p.add_argument("--p", type=int, default=train_contrastive.P)
    p.add_argument("--k", type=int, default=train_contrastive.K)
    p.add_argument("--temperature", type=float, default=train_contrastive.TEMPERATURE)
    p.add_argument("--probe-size", type=int, default=train_contrastive.PROBE_SIZE)
    p.add_argument("--probe-batch-size", type=int, default=train_contrastive.PROBE_BATCH_SIZE)
    p.add_argument("--probe-lr", type=float, default=train_contrastive.PROBE_LR)
    p.add_argument("--probe-steps", type=int, default=train_contrastive.PROBE_STEPS)
    p.add_argument("--stage1-checkpoint", default=train_classification.CHECKPOINT_NAME)
    p.add_argument("--stage2-checkpoint", default=train_contrastive.CHECKPOINT_NAME)
    p.add_argument("--stop-after", choices=["stage1", "stage2"], default="stage2",
                   help="stop the pipeline after this stage")
    p.set_defaults(handler=cmd_pipeline)

    # infer ----------------------------------------------------------------
    p = subparsers.add_parser(
        "infer", aliases=["cluster"],
        help="Cluster images by embedding similarity",
    )
    p.add_argument("--checkpoint", default=inference.CHECKPOINT)
    p.add_argument("--input", action="append", default=None,
                   help="file, directory or URL to cluster (repeatable)")
    p.add_argument("--device", default=inference.DEVICE)
    p.add_argument("--eps", type=float, default=inference.EPS)
    p.add_argument("--min-samples", type=int, default=inference.MIN_SAMPLES)
    p.add_argument("--output", default=None,
                   help="write a JSON image-name -> cluster-id mapping to this path")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(handler=cmd_inference)

    # inspect --------------------------------------------------------------
    p = subparsers.add_parser("inspect", help="Print checkpoint metadata")
    p.add_argument("checkpoint", help="path to a .pt checkpoint")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(handler=cmd_inspect)

    # list-checkpoints -----------------------------------------------------
    p = subparsers.add_parser(
        "list-checkpoints", aliases=["list"],
        help="List checkpoints in a directory",
    )
    p.add_argument("--dir", default=train_classification.OUTPUT_DIR)
    p.add_argument("--metadata", action="store_true",
                   help="load each checkpoint to show metadata (slow for large files)")
    p.set_defaults(handler=cmd_list_checkpoints)

    # env ------------------------------------------------------------------
    p = subparsers.add_parser("env", aliases=["info"], help="Show environment info")
    p.set_defaults(handler=cmd_env)

    # test -----------------------------------------------------------------
    p = subparsers.add_parser("test", aliases=["smoke"], help="Run the module smoke tests")
    p.add_argument("--which", choices=["model", "dataset", "all"], default="model")
    p.set_defaults(handler=cmd_test)

    # download-data --------------------------------------------------------
    p = subparsers.add_parser("download-data", help="Pre-fetch the HuggingFace dataset")
    p.add_argument("--dataset", default=train_classification.DATASET_NAME)
    p.set_defaults(handler=cmd_download_data)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        handler(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

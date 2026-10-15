# SupCon_ANN_Style_Project
A two-stage PyTorch pipeline for **art-style representation learning** and **unsupervised style clustering**. It fine-tunes a timm model with one or multiple dataset(s) — first with supervised style classification, then with supervised contrastive learning — producing embeddings that group paintings by visual style. At inference time the model clusters arbitrary input images with DBSCAN, so **no labels are required** to find stylistic groups.

## Installation

```bash
git clone https://github.com/ZakariaBenmou/siamese-artstyle-recognition.git
cd siamese-artstyle-recognition

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

`main.py` is the command hub — it parses options and dispatches to the training/inference modules. Run `python main.py <command> --help` for command-specific flags.

| Command | Aliases | Purpose |
|---|---|---|
| `classification` | `stage1`, `train1` | Stage 1 — classification pretraining |
| `contrastive` | `stage2`, `train2` | Stage 2 — supervised contrastive training |
| `pipeline` | `all`, `train` | Run Stage 1 then Stage 2 (auto-wires the Stage 1 checkpoint) |
| `infer` | `cluster` | Cluster input images by embedding similarity |
| `inspect <ckpt>` | | Print checkpoint metadata (`--json`) |
| `list-checkpoints` | `list` | List checkpoints in a directory (`--metadata`) |
| `env` | `info` | Show python/torch/timm versions and CUDA info |
| `test` | `smoke` | Run the module smoke tests (`--which model\|dataset\|all`) |
| `download-data` | | Pre-fetch the WikiArt dataset |

### Full pipeline

```bash
python main.py pipeline            # Stage 1 -> Stage 2, end to end
```

### Run stages individually

```bash
python main.py classification      # Stage 1 (writes stage1_classification_WikiArt.pt)
python main.py contrastive         # Stage 2 (loads the Stage 1 checkpoint)
```

### Inference / clustering

```bash
# Cluster a folder of images
python main.py infer --input path/to/images --output clusters.json

# Multiple inputs (repeatable): files, folders, or URLs
python main.py infer --input img_a.jpg --input path/to/folder --input https://example.com/pic.jpg
```
Tune `--eps` (max cosine distance within a cluster) and `--min-samples` (minimum points to form a dense region) for your image set. Defaults are `eps=0.2`, `min_samples=3`.

## Model weights
> <to be added>
Download it into `checkpoints/` if you don't want to test the codebase.

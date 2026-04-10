"""
infer.py — Inference script for the X-DSTF deepfake detection model.

Supports single-image and batch (folder) inference.

Usage examples:
    # Single image
    python infer.py --checkpoint checkpoints/best_model.pth.tar \
                    --input assets/sample.jpg

    # Entire folder
    python infer.py --checkpoint checkpoints/best_model.pth.tar \
                    --input data/test/ \
                    --output results/predictions.csv

    # Adjust decision threshold
    python infer.py --checkpoint checkpoints/best_model.pth.tar \
                    --input data/test/ \
                    --threshold 0.6

Hardware used in paper inference experiments:
    1× NVIDIA L4 (22.16 GB VRAM), Intel Xeon (6-core, 2.2 GHz), 53 GB RAM
    Framework: PyTorch 2.8.0, CUDA 12.6
"""

import os
import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from models.xdstf import DualStreamModel
from transforms.dft import get_transforms

# ─────────────────────────────────────────────────────────────────────────────
#  Supported image extensions
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with the X-DSTF deepfake detection model."
    )

    # ── Required ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth.tar). E.g. checkpoints/best_model.pth.tar",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a single image file OR a directory of images.",
    )

    # ── Optional ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Path to save a CSV of results. "
            "If omitted, results are printed to stdout only."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for fake classification (default: 0.5).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for folder inference (default: 16).",
    )
    parser.add_argument(
        "--d_model",    type=int, default=512,
        help="Transformer hidden dimension (must match checkpoint).",
    )
    parser.add_argument(
        "--nhead",      type=int, default=8,
        help="Number of attention heads (must match checkpoint).",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="Number of Transformer encoder layers (must match checkpoint).",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, d_model: int, nhead: int,
               num_layers: int, device: torch.device) -> nn.Module:
    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    model = DualStreamModel(
        num_classes=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip DataParallel 'module.' prefix if present
    state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)

    epoch = checkpoint.get("epoch", "unknown")
    val_loss = checkpoint.get("best_val_loss", float("nan"))
    print(f"  Epoch: {epoch}  |  Best val loss: {val_loss:.4f}")

    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Image Collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_images(input_path: str) -> list:
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {p.suffix}")
        return [p]
    elif p.is_dir():
        images = sorted([
            f for f in p.rglob("*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not images:
            raise FileNotFoundError(f"No images found in directory: {input_path}")
        return images
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Single-Image Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: Path, spatial_tf, freq_tf, device: torch.device):
    """Load a PIL image and return (spatial_tensor, freq_tensor) on device."""
    img = Image.open(image_path).convert("RGB")
    spatial = spatial_tf(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    freq    = freq_tf(img).unsqueeze(0).to(device)       # (1, 3, 224, 224)
    return spatial, freq


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    image_paths: list,
    spatial_tf,
    freq_tf,
    device: torch.device,
    threshold: float,
    batch_size: int,
) -> list:
    """
    Returns a list of dicts:
        [{"file": str, "probability": float, "prediction": str}, ...]
    """
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        spatial_batch, freq_batch = [], []

        for path in batch_paths:
            try:
                s, f = preprocess(path, spatial_tf, freq_tf, device)
                spatial_batch.append(s)
                freq_batch.append(f)
            except Exception as e:
                print(f"  [Warning] Skipping {path.name}: {e}")
                results.append({
                    "file": str(path),
                    "probability": None,
                    "prediction": "ERROR",
                })
                continue

        if not spatial_batch:
            continue

        spatial_tensor = torch.cat(spatial_batch, dim=0)   # (B, 3, 224, 224)
        freq_tensor    = torch.cat(freq_batch,    dim=0)   # (B, 3, 224, 224)

        logits = model(spatial_tensor, freq_tensor)         # (B, 1)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().tolist()

        for path, prob in zip(batch_paths, probs):
            label = "FAKE" if prob >= threshold else "REAL"
            results.append({
                "file":        str(path),
                "probability": round(prob, 6),
                "prediction":  label,
            })

        n_done = min(i + batch_size, len(image_paths))
        print(f"  Processed {n_done}/{len(image_paths)} images ...", end="\r")

    print()  # newline after progress
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list, threshold: float):
    n_fake  = sum(1 for r in results if r["prediction"] == "FAKE")
    n_real  = sum(1 for r in results if r["prediction"] == "REAL")
    n_error = sum(1 for r in results if r["prediction"] == "ERROR")

    print(f"\n{'='*60}")
    print(f"  X-DSTF Inference Results  (threshold = {threshold})")
    print(f"{'='*60}")
    print(f"  {'File':<45} {'Prob':>8}  {'Label'}")
    print(f"  {'─'*55}")
    for r in results:
        prob_str = f"{r['probability']:.4f}" if r["probability"] is not None else "  N/A"
        fname = Path(r["file"]).name
        print(f"  {fname:<45} {prob_str:>8}  {r['prediction']}")
    print(f"  {'─'*55}")
    print(f"  FAKE: {n_fake}  |  REAL: {n_real}  |  Errors: {n_error}")
    print(f"{'='*60}\n")


def save_csv(results: list, output_path: str):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "probability", "prediction"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Output] Results saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  X-DSTF Inference")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Input      : {args.input}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Batch size : {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Transforms (no augmentation at inference) ───────────────────────────
    spatial_tf, freq_tf = get_transforms(augmentation="none")

    # ── Model ───────────────────────────────────────────────────────────────
    model = load_model(
        args.checkpoint, args.d_model, args.nhead, args.num_layers, device
    )

    # ── Collect Images ──────────────────────────────────────────────────────
    image_paths = collect_images(args.input)
    print(f"[Input] Found {len(image_paths)} image(s).\n")

    # ── Run ─────────────────────────────────────────────────────────────────
    results = run_inference(
        model, image_paths, spatial_tf, freq_tf,
        device, args.threshold, args.batch_size,
    )

    # ── Output ──────────────────────────────────────────────────────────────
    print_results(results, args.threshold)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()

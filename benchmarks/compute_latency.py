import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import time
import numpy as np
from PIL import Image
from models.xdstf import DualStreamModel
from transforms.dft import get_transforms
import torch.nn as nn

def load_images(image_dir, max_images=200):
    images = []

    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, f)
                img = Image.open(img_path).convert("RGB")
                images.append(img)

                if len(images) >= max_images:
                    return images
    return images


def benchmark_inference_batch_real_images(
    model,
    images,
    device,
    batch_size=32,
    runs=100,
    warmup=20
):
    model.eval()

    spatial_tf, freq_tf = get_transforms()

    assert len(images) >= batch_size, "Need at least batch_size images"

    # ---- Preload ONE fixed batch (CPU preprocessing excluded from timing) ----
    batch_imgs = images[:batch_size]

    spatial_list = []
    freq_list = []

    for img in batch_imgs:
        spatial = spatial_tf(img)
        freq = freq_tf(img)
        spatial_list.append(spatial)
        freq_list.append(freq)

    spatial = torch.stack(spatial_list).to(device)
    freq = torch.stack(freq_list).to(device)

    timings = []

    with torch.no_grad():
        # ---- Warmup ----
        for _ in range(warmup):
            _ = model(spatial, freq)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # ---- Timed Runs ----
        for _ in range(runs):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()
            _ = model(spatial, freq)
            ender.record()

            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    mean_ms = np.mean(timings)
    std_ms = np.std(timings)
    throughput = (batch_size * 1000) / mean_ms

    return mean_ms, std_ms, throughput
if __name__ == "__main__":
  IMAGE_DIR = "/path/to/your/images"
  BATCH_SIZE = 32
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model = DualStreamModel().to(device)
  images = load_images(IMAGE_DIR, max_images=200)
  print(f"Loaded {len(images)} images")
  mean_ms, std_ms, fps = benchmark_inference_batch_real_images(
        model,
        images,
        device,
        batch_size=BATCH_SIZE,
        runs=100,
        warmup=20
    )
  print(f"Batch Latency: {mean_ms:.2f} ± {std_ms:.2f} ms")
  print(f"Throughput: {fps:.2f} FPS")

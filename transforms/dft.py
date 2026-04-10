"""
transforms/dft.py — Spatial & frequency-domain transforms for the XDSTF model.

Augmentation strategies (Table 15 in the paper):
    none      : No augmentation. Baseline experiment.
    light     : Geometric only — rotation ±15°, horizontal & vertical flip.
    mild      : Perceptual artifacts — JPEG compression (Q=50),
                Gaussian blur σ∈[0.1,2], brightness ±30%.   ← Best results.
    extensive : Combined geometric + photometric — rotation ±45°, flip,
                brightness ±30%, contrast ±20%, saturation ±20%,
                hue ±10%, Gaussian blur σ∈[0.1,5].

Usage:
    from transforms.dft import get_transforms
    spatial_tf, freq_tf = get_transforms(augmentation="mild")

Notes:
    - Augmentation is applied dynamically to the raw PIL image before any
      tensor conversion, so both the spatial stream and the frequency stream
      see the same augmented image every epoch.
    - Validation transforms always use augmentation="none" regardless of the
      training strategy chosen.
    - All augmentations use torchvision.transforms for reproducibility and
      on-the-fly application (no pre-stored augmented samples).
"""

import torch
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
#  DFT Transform
# ─────────────────────────────────────────────────────────────────────────────

class DFTTransform:
    """
    Converts a float32 RGB tensor (C, H, W) to a 3-channel frequency
    representation: [log-magnitude, phase, raw-magnitude].
    """

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.shape[0] > 1:
            grayscale_tensor = T.functional.rgb_to_grayscale(image_tensor)
        else:
            grayscale_tensor = image_tensor

        dft           = torch.fft.fft2(grayscale_tensor.squeeze(0))
        dft           = torch.fft.fftshift(dft)
        raw_magnitude = torch.abs(dft)
        log_magnitude = torch.log(raw_magnitude + 1e-8)
        phase         = torch.angle(dft)

        # Shape: (3, H, W) — channels are [log_magnitude, phase, raw_magnitude]
        return torch.stack([log_magnitude, phase, raw_magnitude], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation Strategies (PIL-level, applied before ToTensor)
# ─────────────────────────────────────────────────────────────────────────────

def _build_augmentation(strategy: str) -> T.Compose:
    """
    Returns a PIL-level augmentation transform for the given strategy.
    Applied before Resize/ToTensor so that both streams see the same image.
    """
    if strategy == "none":
        return T.Compose([])                               # identity

    elif strategy == "light":
        # Experiment 2 — geometric transformations only
        return T.Compose([
            T.RandomRotation(degrees=15),                  # ±15°
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

    elif strategy == "mild":
        # Experiment 3 — perceptual/encoding artifacts (best performing)
        # JPEG compression is simulated via RandomApply + jpeg_quality
        return T.Compose([
            T.RandomApply(
                [T.Lambda(lambda img: T.functional.adjust_brightness(img, 1.30))],
                p=0.5,
            ),                                             # brightness Δ=30 %
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))],
                p=0.5,
            ),                                             # σ∈[0.1, 2]
            T.RandomApply(
                [T.Lambda(
                    lambda img: T.functional.jpeg(img, quality=50)
                    if hasattr(T.functional, "jpeg")
                    else img
                )],
                p=0.5,
            ),                                             # JPEG Q=50
        ])

    elif strategy == "extensive":
        # Experiment 4 — combined geometric + photometric
        return T.Compose([
            T.RandomRotation(degrees=45),                  # ±45°
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.30,                           # Δ=30 %
                contrast=0.20,                             # ±20 %
                saturation=0.20,                           # ±20 %
                hue=0.10,                                  # ±10 %
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))],
                p=0.5,
            ),                                             # σ∈[0.1, 5]
        ])

    else:
        raise ValueError(
            f"Unknown augmentation strategy: '{strategy}'. "
            f"Choose from: none | light | mild | extensive."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(augmentation: str = "none"):
    """
    Returns (spatial_transform, freq_transform) for the chosen augmentation.

    Args:
        augmentation: One of "none" | "light" | "mild" | "extensive".
                      Defaults to "none" (clean images, no augmentation).

    Returns:
        spatial_transform : torchvision transform for the EfficientNet-V2-S stream.
        freq_transform    : torchvision transform for the ConvNeXt-Base / DFT stream.

    Both transforms apply the same augmentation to the raw PIL image first,
    ensuring that the frequency representation reflects the augmented content.
    """
    aug = _build_augmentation(augmentation)

    spatial_transform = T.Compose([
        aug,                                               # PIL-level augmentation
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    freq_transform = T.Compose([
        aug,                                               # same PIL-level augmentation
        T.Resize((224, 224)),
        T.ToTensor(),
        DFTTransform(),                                    # RGB tensor → [log_mag, phase, raw_mag]
    ])

    return spatial_transform, freq_transform

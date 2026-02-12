import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
from torchinfo import summary
from PIL import Image
from models.xdstf import DualStreamModel
from transforms.dft import get_transforms

def compute_params(model, image_path, device):
    spatial_tf, freq_tf = get_transforms()
    img = Image.open(image_path).convert("RGB")

    spatial = spatial_tf(img).unsqueeze(0).to(device)
    freq = freq_tf(img).unsqueeze(0).to(device)

    info = summary(
        model,
        input_data=(spatial, freq),
        verbose=0
    )

    total_params = info.total_params
    trainable_params = info.trainable_params

    return total_params / 1e6


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamModel().to(device)

    total = compute_params(model, "assets/sample.jpg", device)

    print(f"Total Parameters: {total:.2f} M")

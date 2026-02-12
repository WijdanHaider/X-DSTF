import torch
from thop import profile
from PIL import Image
from models.xdstf import DualStreamModel
from transforms.dft import get_transforms

def compute_flops(model, image_path, device):
    model.eval()

    spatial_tf, freq_tf = get_transforms()
    img = Image.open(image_path).convert("RGB")

    spatial = spatial_tf(img).unsqueeze(0).to(device)
    freq = freq_tf(img).unsqueeze(0).to(device)

    flops, _ = profile(model, inputs=(spatial, freq), verbose=False)

    return flops / 1e9  # GFLOPs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamModel().to(device)

    flops = compute_flops(model, "assets/sample.jpg", device)

    print(f"FLOPs: {flops:.2f} GFLOPs")

import torch
import argparse
from models.xdstf import DualStreamModel
from transforms.dft import get_transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    model = DualStreamModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    spatial_t, freq_t = get_transforms()

    img = Image.open(args.image).convert("RGB")
    spatial = spatial_t(img).unsqueeze(0).to(device)
    freq = freq_t(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(spatial, freq)
        prob = torch.sigmoid(logits).item()

    print(f"Forgery probability: {prob:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args)

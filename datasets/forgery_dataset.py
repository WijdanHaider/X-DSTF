import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

class ForgeryDataset(Dataset):
    def __init__(self, image_paths, labels, spatial_transform, freq_transform):
        self.image_paths = image_paths
        self.labels = labels
        self.spatial_transform = spatial_transform
        self.freq_transform = freq_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logging.warning(f"Could not load {img_path}")
            dummy = torch.randn(3,224,224)
            return dummy, dummy, torch.tensor(-1.0)

        spatial_img = self.spatial_transform(image)
        freq_img = self.freq_transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return spatial_img, freq_img, label

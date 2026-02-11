import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import numpy as np
from PIL import Image


class DualStreamModel(nn.Module):
    def __init__(self, num_classes=1, d_model=512, nhead=8, num_layers=2):
        super(DualStreamModel, self).__init__()

        # Spatial Stream - EfficientNetV2
        self.spatial_stream = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        num_spatial_features = self.spatial_stream.classifier[-1].in_features
        self.spatial_stream.classifier = nn.Identity()

        # Frequency Stream - ConvNeXt
        self.freq_stream = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        num_freq_features = self.freq_stream.classifier[-1].in_features
        self.freq_stream.classifier = nn.Identity()

        # Linear projections into common dimension
        self.proj_spatial = nn.Linear(num_spatial_features, d_model)
        self.proj_freq = nn.Linear(num_freq_features, d_model)

        # Transformer encoder for fusion
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_fusion = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, spatial_input, freq_input):
        spatial_features = self.spatial_stream(spatial_input)   # [B, D1]
        freq_features = self.freq_stream(freq_input)            # [B, D2]

        if freq_features.ndim == 4:  # flatten convnext output if needed
            freq_features = torch.flatten(freq_features, start_dim=1)

        # Project to common dim
        spatial_proj = self.proj_spatial(spatial_features)   # [B, d_model]
        freq_proj = self.proj_freq(freq_features)           # [B, d_model]

        # Stack as sequence of length 2
        tokens = torch.stack([spatial_proj, freq_proj], dim=1)  # [B, 2, d_model]

        # Transformer fusion
        fused_tokens = self.transformer_fusion(tokens)  # [B, 2, d_model]

        # Pool: mean over tokens
        fused = fused_tokens.mean(dim=1)  # [B, d_model]


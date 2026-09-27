import torch
import torch.nn as nn
from src.models.cnn_backbone import build_densenet_backbone

class MultiTaskDenseNet121(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskDenseNet121, self).__init__()
        # Shared Encoder Trunk
        self.backbone, num_features = build_densenet_backbone(pretrained=pretrained)
        
        # Shared Latent Bottleneck
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-Specific Decoders (Routes)
        self.head_decoder = nn.Linear(256, 8)     # 4 endpoints (BPD, OFD)
        self.abdomen_decoder = nn.Linear(256, 8)  # 4 endpoints (TAD, APAD)
        self.femur_decoder = nn.Linear(256, 4)    # 2 endpoints (FL)

    def forward(self, x, task='abdomen'):
        features = self.backbone(x)
        latent = self.shared_fc(features)
        
        # Route to the specific anatomy head
        if task == 'head':
            return self.head_decoder(latent)
        elif task == 'abdomen':
            return self.abdomen_decoder(latent)
        elif task == 'femur':
            return self.femur_decoder(latent)
        else:
            raise ValueError(f"Unknown anatomical task: {task}")
""" Classical Baselines for Benchmarking against HQNN. """
import torch
import torch.nn as nn
import torchvision.models as models

class PureDenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classical_projection = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.segmentation_decoder = nn.Sequential(
            nn.Linear(256, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, images):
        x = self.features(images)
        x = torch.flatten(self.pool(x), 1)
        features = self.classical_projection(x)
        
        oval_mask_logits = self.segmentation_decoder(features)
        coords_pred = torch.clamp(self.regression_head(features), min=-1.0, max=1.0)
        
        if self.training:
            return coords_pred, oval_mask_logits
        return coords_pred
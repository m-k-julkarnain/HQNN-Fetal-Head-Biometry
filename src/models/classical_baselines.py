""" Classical Baselines for Benchmarking against HQNN. """
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.heatmap_decoder = nn.Sequential(
            nn.Linear(256, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, images):
        x = self.features(images)
        x = torch.flatten(self.pool(x), 1)
        features = self.classical_projection(x)
        
        oval_mask_logits = self.segmentation_decoder(features)
        
        heatmaps = self.heatmap_decoder(features)
        B, C, H, W = heatmaps.shape
        spatial_softmax = F.softmax(heatmaps.view(B, C, -1), dim=-1).view(B, C, H, W)
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=images.device),
                                        torch.linspace(-1, 1, W, device=images.device), indexing='ij')
        
        expected_x = torch.sum(spatial_softmax * grid_x, dim=(2, 3))
        expected_y = torch.sum(spatial_softmax * grid_y, dim=(2, 3))
        
        coords_pred = torch.empty((B, 8), device=images.device)
        coords_pred[:, 0::2] = expected_x
        coords_pred[:, 1::2] = expected_y
        
        if self.training:
            return coords_pred, oval_mask_logits
        return coords_pred, oval_mask_logits
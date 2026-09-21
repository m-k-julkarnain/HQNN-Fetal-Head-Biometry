""" Hybrid Quantum-Classical Domain Adversarial Neural Network with DSNT Heatmaps. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from src.models.cnn_backbone import FetalCNNBackbone
from src.quantum.pqc_layer import PQCLayer

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class FetalHeadHQNN(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4):
        super().__init__()
        self.cnn_backbone = FetalCNNBackbone(n_qubits=n_qubits)
        self.pqc_layer = PQCLayer(n_qubits=n_qubits, n_layers=n_layers)
        
        self.segmentation_decoder = nn.Sequential(
            nn.Linear(256 + n_qubits, 512 * 7 * 7),
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
        
        # 1. DSNT Heatmap Generator (Outputs 4 probability maps for 4 endpoints)
        self.heatmap_decoder = nn.Sequential(
            nn.Linear(256 + n_qubits, 512 * 7 * 7),
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
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(256 + n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, images, alpha=1.0):
        classical_features, quantum_inputs = self.cnn_backbone(images)
        quantum_features = self.pqc_layer(quantum_inputs)
        
        quantum_features = quantum_features.type_as(classical_features)
        combined_features = torch.cat((classical_features, quantum_features), dim=1)
        
        oval_mask_logits = self.segmentation_decoder(combined_features)
        
        # 2. Differentiable Spatial to Numerical Transform (DSNT)
        heatmaps = self.heatmap_decoder(combined_features)
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
            reversed_features = GradientReversalLayer.apply(combined_features, alpha)
            domain_logits = self.domain_classifier(reversed_features)
            return coords_pred, oval_mask_logits, domain_logits
            
        return coords_pred, oval_mask_logits
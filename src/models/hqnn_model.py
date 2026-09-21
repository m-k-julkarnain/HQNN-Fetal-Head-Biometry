""" Hybrid Quantum-Classical Domain Adversarial Neural Network (HQ-DANN). """
import torch
import torch.nn as nn
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
        
        self.regression_head = nn.Sequential(
            nn.Linear(256 + n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        
        # 3. Domain Discriminator (The DANN Upgrade)
        # Forces the combined features to be hospital-agnostic
        self.domain_classifier = nn.Sequential(
            nn.Linear(256 + n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2) # Predicts Source Domain A vs Source Domain B
        )

    def forward(self, images, alpha=1.0):
        classical_features, quantum_inputs = self.cnn_backbone(images)
        quantum_features = self.pqc_layer(quantum_inputs)
        
        quantum_features = quantum_features.type_as(classical_features)
        combined_features = torch.cat((classical_features, quantum_features), dim=1)
        
        oval_mask_logits = self.segmentation_decoder(combined_features)
        coords_pred = torch.clamp(self.regression_head(combined_features), min=-1.0, max=1.0)
        
        if self.training:
            # Reverse gradients to penalize domain memorization
            reversed_features = GradientReversalLayer.apply(combined_features, alpha)
            domain_logits = self.domain_classifier(reversed_features)
            return coords_pred, oval_mask_logits, domain_logits
            
        return coords_pred, oval_mask_logits
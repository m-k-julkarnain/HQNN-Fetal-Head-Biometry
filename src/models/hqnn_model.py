""" Hybrid Quantum-Classical Neural Network (HQNN) model architecture. """
import torch
import torch.nn as nn
from src.models.cnn_backbone import FetalCNNBackbone
from src.quantum.pqc_layer import PQCLayer

class FetalHeadHQNN(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4):
        super().__init__()
        self.cnn_backbone = FetalCNNBackbone(n_qubits=n_qubits, pretrained=True)
        self.pqc_layer = PQCLayer(n_qubits=n_qubits, n_layers=n_layers)
        
        # 1. Anatomical Structure Segmentation Decoder (Fetal Head Circumference)
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
        
        # 2. Dual-Diameter Boundary Regression Head (8 coordinates: 4 BPD + 4 OFD)
        self.regression_head = nn.Sequential(
            nn.Linear(256 + n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, images):
        classical_features, quantum_inputs = self.cnn_backbone(images)
        quantum_features = self.pqc_layer(quantum_inputs)
        
        quantum_features = quantum_features.type_as(classical_features)
        combined_features = torch.cat((classical_features, quantum_features), dim=1)
        
        oval_mask_logits = self.segmentation_decoder(combined_features)
        coords_pred = self.regression_head(combined_features)
        coords_pred = torch.clamp(coords_pred, min=-1.0, max=1.0)
        
        if self.training:
            return coords_pred, oval_mask_logits
        return coords_pred
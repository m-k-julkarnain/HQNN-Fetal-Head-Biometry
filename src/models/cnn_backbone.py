""" CNN Backbone with Spatial Feature Extraction for Fetal Biometry. """
import torch
import torch.nn as nn
import torchvision.models as models

class FetalCNNBackbone(nn.Module):
    def __init__(self, n_qubits=8):
        super().__init__()
        # Pre-trained backbone is mathematically necessary for feature extraction on small datasets
        weights = models.DenseNet121_Weights.DEFAULT
        densenet = models.densenet121(weights=weights)
        
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classical_projection = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.quantum_projection = nn.Linear(256, n_qubits)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        classical_features = self.classical_projection(x)
        quantum_inputs = self.quantum_projection(classical_features)
        
        return classical_features, quantum_inputs
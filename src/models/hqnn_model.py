import torch
import torch.nn as nn
from src.models.cnn_backbone import build_densenet_backbone
from src.quantum.pqc_layer import HybridPQC

class MultiTaskHQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, pretrained=True):
        super(MultiTaskHQNN, self).__init__()
        # Shared Classical Trunk
        self.backbone, num_features = build_densenet_backbone(pretrained=pretrained)
        self.reduce_dim = nn.Linear(num_features, n_qubits)
        
        # Shared Quantum PQC Layer
        self.quantum_layer = HybridPQC(n_qubits=n_qubits, n_layers=n_layers)
        
        # Shared Post-Quantum Expansion
        self.post_q_fc = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-Specific Decoders (Routes)
        self.head_decoder = nn.Linear(64, 8)     # 4 endpoints (BPD, OFD)
        self.abdomen_decoder = nn.Linear(64, 8)  # 4 endpoints (TAD, APAD)
        self.femur_decoder = nn.Linear(64, 4)    # 2 endpoints (FL)

    def forward(self, x, task='abdomen'):
        features = self.backbone(x)
        q_in = self.reduce_dim(features)
        
        # Pass through the shared Quantum Circuit
        q_out = self.quantum_layer(q_in)
        latent = self.post_q_fc(q_out)
        
        # Route to the specific anatomy head
        if task == 'head':
            coords = self.head_decoder(latent)
        elif task == 'abdomen':
            coords = self.abdomen_decoder(latent)
        elif task == 'femur':
            coords = self.femur_decoder(latent)
        else:
            raise ValueError(f"Unknown anatomical task: {task}")
            
        return coords, q_out
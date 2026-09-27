import torch.nn as nn
from src.models.cnn_backbone import FetalCNNBackbone

class MultiTaskDenseNet121(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskDenseNet121, self).__init__()
        # Use your custom CBAM backbone
        self.backbone = FetalCNNBackbone(n_qubits=8) 
        
        # Task-Specific Decoders (Using the 256-dim classical feature output)
        self.head_decoder = nn.Linear(256, 8)     # 4 endpoints (BPD, OFD)
        self.abdomen_decoder = nn.Linear(256, 8)  # 4 endpoints (TAD, APAD)
        self.femur_decoder = nn.Linear(256, 4)    # 2 endpoints (FL)

    def forward(self, x, task='abdomen'):
        # Extract features using your CBAM implementation
        classical_features, _ = self.backbone(x)
        
        # Route to the specific anatomy head
        if task == 'head':
            return self.head_decoder(classical_features)
        elif task == 'abdomen':
            return self.abdomen_decoder(classical_features)
        elif task == 'femur':
            return self.femur_decoder(classical_features)
        else:
            raise ValueError(f"Unknown anatomical task: {task}")
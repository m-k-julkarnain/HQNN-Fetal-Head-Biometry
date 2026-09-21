""" CNN Backbone with CBAM Spatial/Channel Attention for Fetal Biometry. """
import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class FetalCNNBackbone(nn.Module):
    def __init__(self, n_qubits=8):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features
        
        # Inject CBAM right before global pooling to force spatial awareness
        self.cbam = CBAM(1024)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classical_projection = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.quantum_projection = nn.Linear(256, n_qubits)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        classical_features = self.classical_projection(x)
        quantum_inputs = self.quantum_projection(classical_features)
        
        return classical_features, quantum_inputs
# app/clinical/cardiology/models/ef3dcnn_model.py

import torch
import torch.nn as nn

class EF3DCNN(nn.Module):
    """
    Simple 3D CNN: Input [B,1,T,H,W] -> output EF
    """
    def __init__(self, in_channels=1, T=32, H=112, W=112):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d((1,2,2))
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d((1,2,2))
        
        self.feature_dim = 32 * T * (H//4) * (W//4)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out.squeeze(1)
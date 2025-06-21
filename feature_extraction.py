import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    3-layer 3D CNN for spectral-spatial feature extraction, pretrained on ImageNet.
    Input: Patches (N x 32 x 32 x 30)
    Output: Feature maps (N x 32 x 32 x 64)
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights (simulating ImageNet pretraining)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (N, 32, 32, 30) -> (N, 1, 32, 32, 30)
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # Output: (N, 64, 32, 32, 1) -> (N, 32, 32, 64)
        x = x.squeeze(-1).permute(0, 2, 3, 1)
        return x
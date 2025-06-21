import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossDomainAdapter(nn.Module):
    """
    Lightweight transformer for aligning HSI and RGB features using MMD.
    Input: HSI features (N x 32 x 32 x 64)
    Output: Aligned features (N x 32 x 32 x 64)
    """
    def __init__(self, dim=64, num_heads=6, num_layers=4, prune_ratio=0.3):
        super(CrossDomainAdapter, self).__init__()
        self.dim = dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4),
            num_layers=num_layers
        )
        # Simulate pruning by reducing parameters (placeholder)
        self.prune_mask = torch.ones(dim)  # Simplified pruning
        self.prune_mask[int(dim * prune_ratio):] = 0

    def compute_mmd(self, x, y):
        """
        Compute Maximum Mean Discrepancy between two feature sets.
        Input: x, y (N x D)
        Output: MMD loss
        """
        def gaussian_kernel(x, y, sigma=1.0):
            x = x.unsqueeze(1)  # (N, 1, D)
            y = y.unsqueeze(0)  # (1, M, D)
            dist = torch.sum((x - y) ** 2, dim=-1)
            return torch.exp(-dist / (2 * sigma ** 2))

        xx = gaussian_kernel(x, x).mean()
        yy = gaussian_kernel(y, y).mean()
        xy = gaussian_kernel(x, y).mean()
        return xx + yy - 2 * xy

    def forward(self, hsi_features, rgb_features=None):
        # Input: (N, 32, 32, 64) -> (N*32*32, 64)
        N, H, W, C = hsi_features.shape
        x = hsi_features.view(-1, C)
        # Apply transformer
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        # Apply pruning mask (simplified)
        x = x * self.prune_mask.to(x.device)
        # Reshape back
        x = x.view(N, H, W, C)
        # MMD loss (if RGB features provided)
        if rgb_features is not None:
            hsi_flat = x.view(-1, C)
            rgb_flat = rgb_features.view(-1, C)
            mmd_loss = self.compute_mmd(hsi_flat, rgb_flat)
            return x, mmd_loss
        return x
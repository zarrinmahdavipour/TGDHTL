import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class MSSA(nn.Module):
    """
    Multi-Scale Stripe Attention module.
    Input: Aligned features (N x 32 x 32 x 64)
    Output: Multi-scale features (N x 32 x 32 x 64)
    """
    def __init__(self, dim=64, num_heads=6, scales=[4, 8, 16]):
        super(MSSA, self).__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads) for _ in scales
        ])
        self.linear = nn.Linear(dim * len(scales), dim)

    def forward(self, x):
        # Input: (N, 32, 32, 64)
        N, H, W, C = x.shape
        features = []
        for p, attn in zip(self.scales, self.attentions):
            # Partition into stripes
            patch_size = 32 // p
            x_p = x.view(N, p, patch_size, p, patch_size, C)
            x_p = x_p.permute(0, 1, 3, 2, 4, 5).reshape(N, p*p, patch_size*patch_size*C)
            # Apply attention
            x_p, _ = attn(x_p, x_p, x_p)
            # Reshape back
            x_p = x_p.view(N, p, p, patch_size, patch_size, C).permute(0, 1, 3, 2, 4, 5)
            x_p = x_p.reshape(N, H, W, C)
            features.append(x_p)
        # Aggregate
        x = torch.cat(features, dim=-1)  # (N, 32, 32, 64*3)
        x = self.linear(x)  # (N, 32, 32, 64)
        return x

class GCN(nn.Module):
    """
    Graph Convolutional Network for spatial coherence.
    Input: MSSA features (N x 32 x 32 x 64)
    Output: Fused features (N x 32 x 32 x 64)
    """
    def __init__(self, dim=64, threshold=0.85, lambda_weight=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.threshold = threshold
        self.lambda_weight = lambda_weight
        self.relu = nn.ReLU(inplace=True)

    def build_graph(self, features):
        # Input: (N, 32, 32, 64) -> (N*32*32, 64)
        N, H, W, C = features.shape
        x = features.view(-1, C)
        # Compute cosine similarity
        x_norm = F.normalize(x, dim=1)
        sim = torch.matmul(x_norm, x_norm.t())
        # Create sparse edge index
        edge_index = (sim > self.threshold).nonzero().t()
        return Data(x=x, edge_index=edge_index)

    def forward(self, x):
        # Build graph
        graph = self.build_graph(x)
        # Apply GCN
        h = self.conv1(graph.x, graph.edge_index)
        h = self.relu(h)
        h = self.conv2(h, graph.edge_index)
        h = h.view(x.shape)
        # Fuse with input
        x_fused = x + self.lambda_weight * h
        return x_fused
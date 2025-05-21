"""Model definitions for the TransHTL framework.

This module contains the implementations of the Multi-Scale Stripe Attention (MSSA),
Graph Convolutional Network (GCN), Diffusion Model, and the main TransHTLPlus model
for hyperspectral image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class MSSA(nn.Module):
    """Multi-Scale Stripe Attention (MSSA) module.

    This module applies attention at multiple spatial scales to capture spectral-spatial
    dependencies in hyperspectral data.

    Args:
        dim (int): Input feature dimension.
        scales (list of int): List of spatial scales for attention (default: [2, 4, 8, 16]).
        num_heads (int): Number of attention heads (default: 6).
    """
    def __init__(self, dim, scales=[2, 4, 8, 16], num_heads=6):
        super(MSSA, self).__init__()
        self.scales = scales
        # Create attention modules for each scale
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            for _ in scales
        ])
        self.linear = nn.Linear(dim * len(scales), dim)  # Linear layer to fuse outputs

    def forward(self, x, return_attention=False):
        """Forward pass of the MSSA module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width, depth).
            return_attention (bool): If True, returns attention weights (default: False).

        Returns:
            torch.Tensor: Output tensor after multi-scale attention.
            list of torch.Tensor, optional: Attention weights for each scale if return_attention is True.
        """
        batch, channels, height, width, depth = x.size()
        outputs = []
        attn_weights = []
        for scale, attn in zip(self.scales, self.attentions):
            stride = height // scale  # Compute stride for downsampling
            x_scaled = x[:, :, ::stride, ::stride, :]  # Downsample spatially
            x_flat = x_scaled.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)  # Flatten for attention
            attn_out, attn_w = attn(x_flat, x_flat, x_flat, need_weights=True)  # Apply attention
            outputs.append(attn_out)
            if return_attention:
                attn_weights.append(attn_w.mean(dim=1))  # Average attention weights over heads
        x_concat = torch.cat(outputs, dim=-1)  # Concatenate outputs from all scales
        x_out = self.linear(x_concat)  # Fuse features
        x_out = x_out.reshape(batch, channels, height // stride, width // stride, depth)  # Reshape to 5D
        if return_attention:
            return x_out, attn_weights
        return x_out

class GCN(nn.Module):
    """Graph Convolutional Network (GCN) module.

    This module applies graph convolutions to model pixel relationships in hyperspectral data.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        num_layers (int): Number of GCN layers (default: 2).
    """
    def __init__(self, in_dim, out_dim, num_layers=2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList([
            pyg_nn.GCNConv(in_dim if i == 0 else out_dim, out_dim)
            for i in range(num_layers)
        ])
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """Forward pass of the GCN module.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch, num_nodes, features).
            edge_index (torch.Tensor): Graph edge indices of shape (2, num_edges).

        Returns:
            torch.Tensor: Output features after graph convolution.
        """
        for layer in self.layers:
            x = layer(x, edge_index)  # Apply graph convolution
            x = self.relu(x)  # Apply ReLU activation
        return x

class DiffusionModel(nn.Module):
    """Diffusion Model for data augmentation.

    This module generates synthetic hyperspectral data using a diffusion process.

    Args:
        input_dim (int): Input feature dimension (number of spectral bands).
        timesteps (int): Number of diffusion timesteps (default: 15).
    """
    def __init__(self, input_dim, timesteps=15):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        # Define U-Net architecture for noise prediction
        self.unet = nn.Sequential(
            nn.Conv3d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, input_dim, kernel_size=3, padding=1)
        )
        # Define diffusion schedules
        self.beta = torch.linspace(0.0001, 0.02, timesteps).cuda()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        """Forward pass for adding noise in the diffusion process.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width, depth).
            t (torch.Tensor): Timestep indices.

        Returns:
            tuple: Noisy data and added noise.
        """
        noise = torch.randn_like(x).cuda()  # Generate random noise
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise  # Add noise to input
        return x_t, noise

    def generate(self, shape):
        """Generate synthetic data using the reverse diffusion process.

        Args:
            shape (tuple): Shape of the synthetic data (batch, channels, height, width, depth).

        Returns:
            torch.Tensor: Generated synthetic data.
        """
        x = torch.randn(shape).cuda()  # Start with random noise
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long).cuda()
            noise_pred = self.unet(x, t_tensor)  # Predict noise
            alpha = self.alpha[t].view(-1, 1, 1, 1, 1)
            alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1, 1)
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha)  # Denoise step
        return x

class TransHTLPlus(nn.Module):
    """TransHTL+ model for hyperspectral image classification.

    This model integrates a feature extractor, cross-domain adapter, MSSA, GCN, and classifier.

    Args:
        classes (int): Number of output classes.
        input_dim (int): Input feature dimension after PCA (default: 30).
    """
    def __init__(self, classes, input_dim=30):
        super(TransHTLPlus, self).__init__()
        # Feature extractor with 3D convolutions
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        # Cross-domain adapter using a pruned transformer
        self.cross_domain_adapter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=6, batch_first=True),
            num_layers=4
        )
        self.mssa = MSSA(dim=64, scales=[2, 4, 8, 16], num_heads=6)  # Multi-scale attention
        self.gcn = GCN(in_dim=64, out_dim=64, num_layers=2)  # Graph convolution
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, classes)
        )

    def build_graph(self, x):
        """Build a graph based on cosine similarity between features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width, depth).

        Returns:
            torch.Tensor: Edge indices for the graph.
        """
        batch, channels, height, width, depth = x.size()
        x_flat = x.view(batch, channels, -1).permute(0, 2, 1)
        cosine_sim = F.cosine_similarity(x_flat.unsqueeze(2), x_flat.unsqueeze(1), dim=3)
        edge_index = torch.nonzero(cosine_sim > 0.8, as_tuple=False).t()  # Threshold similarity
        return edge_index

    def forward(self, x, return_attention=False, return_features=False):
        """Forward pass of the TransHTL+ model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, height, width, depth).
            return_attention (bool): If True, returns attention weights (default: False).
            return_features (bool): If True, returns pooled features (default: False).

        Returns:
            torch.Tensor: Classification logits.
            list of torch.Tensor, optional: Attention weights if return_attention is True.
            torch.Tensor, optional: Pooled features if return_features is True.
        """
        x = self.feature_extractor(x)  # Extract features
        batch, channels, height, width, depth = x.size()
        x_flat = x.view(batch, channels, -1).permute(0, 2, 1)  # Flatten for transformer
        x_adapted = self.cross_domain_adapter(x_flat)  # Apply cross-domain adaptation
        x_adapted = x_adapted.permute(0, 2, 1).view(batch, channels, height, width, depth)
        if return_attention:
            x_mssa, attn_weights = self.mssa(x_adapted, return_attention=True)  # Apply MSSA
        else:
            x_mssa = self.mssa(x_adapted)
        edge_index = self.build_graph(x_adapted)  # Build graph
        x_gcn = self.gcn(x_adapted.view(batch, channels, -1).permute(0, 2, 1), edge_index)  # Apply GCN
        x_gcn = x_gcn.permute(0, 2, 1).view(batch, channels, height, width, depth)
        x_fused = x_mssa + 0.5 * x_gcn  # Fuse MSSA and GCN outputs
        x_pool = F.adaptive_avg_pool3d(x_fused, (1, 1, 1)).view(batch, -1)  # Global pooling
        out = self.classifier(x_pool)  # Classify
        if return_attention and return_features:
            return out, attn_weights, x_pool
        elif return_attention:
            return out, attn_weights
        elif return_features:
            return out, x_pool
        return out
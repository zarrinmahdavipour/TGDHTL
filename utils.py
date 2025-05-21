"""Utility functions for the TransHTL framework.

This module provides helper functions for data preprocessing, metric computation,
and data generation.
"""

import numpy as np
import torch
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def list_to_colormap(x_list):
    """Convert class labels to RGB colors for visualization.

    Args:
        x_list (np.ndarray): Array of class labels.

    Returns:
        np.ndarray: Array of RGB colors of shape (num_samples, 3).
    """
    y = np.zeros((x_list.shape[0], 3))
    color_map = {
        0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1], 4: [0.5, 0.5, 0],
        5: [0, 1, 1], 6: [1, 0, 1], 7: [0.5, 0, 0], 8: [0, 0.5, 0], 9: [0.5, 0, 0.5],
        10: [1, 0.5, 0], 11: [0, 0.5, 0.5], 12: [0.5, 0.5, 0], 13: [0.5, 0, 0.5],
        14: [0, 0.5, 1], 15: [0.75, 0.75, 0.75], 16: [0.25, 0.25, 0.25]
    }
    for idx, val in enumerate(x_list):
        y[idx] = color_map[int(val)]  # Map each label to its RGB color
    return y

def compute_sam(real_data, synthetic_data):
    """Compute Spectral Angle Mapper (SAM) between real and synthetic data.

    Args:
        real_data (np.ndarray): Real hyperspectral data of shape (..., bands).
        synthetic_data (np.ndarray): Synthetic data of shape (..., bands).

    Returns:
        float: Average SAM score in radians.
    """
    real_data = real_data.reshape(-1, real_data.shape[-1])
    synthetic_data = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    sam_scores = []
    for r, s in zip(real_data, synthetic_data):
        r_norm = np.linalg.norm(r)
        s_norm = np.linalg.norm(s)
        if r_norm > 0 and s_norm > 0:
            cos_theta = np.dot(r, s) / (r_norm * s_norm)  # Compute cosine similarity
            cos_theta = np.clip(cos_theta, -1, 1)
            sam = np.arccos(cos_theta)  # Convert to angle
            sam_scores.append(sam)
    return np.mean(sam_scores) if sam_scores else 0.0

def compute_mmd(x, y, sigma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) between two feature sets.

    Args:
        x (torch.Tensor): First feature set of shape (num_samples, features).
        y (torch.Tensor): Second feature set of shape (num_samples, features).
        sigma (float): Gaussian kernel bandwidth (default: 1.0).

    Returns:
        float: MMD score.
    """
    def gaussian_kernel(x, y, sigma):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))

    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    xx = gaussian_kernel(x, x, sigma).mean()  # Kernel mean for x
    yy = gaussian_kernel(y, y, sigma).mean()  # Kernel mean for y
    xy = gaussian_kernel(x, y, sigma).mean()  # Cross-kernel mean
    return xx + yy - 2 * xy

def preprocess_data(data, gt, patch_length=16, n_components=30):
    """Preprocess hyperspectral data by applying PCA and extracting patches.

    Args:
        data (np.ndarray): Input hyperspectral data of shape (height, width, bands).
        gt (np.ndarray): Ground truth labels of shape (height, width).
        patch_length (int): Half-size of the patch (default: 16).
        n_components (int): Number of PCA components (default: 30).

    Returns:
        tuple: Patches (torch.Tensor), labels (torch.Tensor), coordinates (list), processed data (np.ndarray).
    """
    h, w, bands = data.shape
    # Normalize data
    data = scale(data.reshape(-1, bands)).reshape(h, w, bands)
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data.reshape(-1, bands)).reshape(h, w, n_components)
    patches = []
    labels = []
    coords = []
    # Extract patches centered at labeled pixels
    for i in range(patch_length, h - patch_length):
        for j in range(patch_length, w - patch_length):
            if gt[i, j] > 0:
                patch = data[i-patch_length:i+patch_length, j-patch_length:j+patch_length, :]
                patches.append(patch)
                labels.append(gt[i, j] - 1)  # Adjust labels to 0-based indexing
                coords.append((i, j))
    patches = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    labels = torch.tensor(labels, dtype=torch.long)
    return patches, labels, coords, data

def generate_rgb_data(hsi_data):
    """Generate RGB-like data from hyperspectral data for cross-domain adaptation.

    Args:
        hsi_data (torch.Tensor): Hyperspectral data of shape (batch, 1, height, width, bands).

    Returns:
        torch.Tensor: RGB-like data of shape (batch, 1, height, width, 3).
    """
    hsi_data = hsi_data.squeeze(1)  # Remove channel dimension
    rgb_data = hsi_data[:, :, :, :3]  # Take first three bands
    return rgb_data.unsqueeze(1)  # Add channel dimension back
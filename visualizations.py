"""Visualization functions for the TransHTL framework.

This module provides functions to generate ground truth, false color, classification maps,
attention maps, and t-SNE visualizations for hyperspectral image classification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import torch
from .config import DATASETS
from .utils import list_to_colormap

def generate_groundtruth_image(data, gt, dataset_name):
    """Generate and save ground truth visualization.

    Args:
        data (np.ndarray): Input hyperspectral data (for shape reference).
        gt (np.ndarray): Ground truth labels of shape (height, width).
        dataset_name (str): Name of the dataset (e.g., 'IndianPines').
    """
    h, w = gt.shape
    gt_rgb = list_to_colormap(gt.ravel()).reshape(h, w, 3)  # Convert labels to RGB
    
    output_dir = f'results/{dataset_name}/ground_truth'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(gt_rgb)
    plt.title(f'{dataset_name} Ground Truth')
    plt.axis('off')
    plt.savefig(f'{output_dir}/{dataset_name}_groundtruth.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_falsecolor_image(data, dataset_name):
    """Generate and save false color visualization using selected bands.

    Args:
        data (np.ndarray): Hyperspectral data of shape (height, width, bands).
        dataset_name (str): Name of the dataset (e.g., 'IndianPines').
    """
    h, w, bands = data.shape
    band_indices = DATASETS[dataset_name]['false_color_bands']
    # Ensure band indices are valid
    if max(band_indices) >= bands:
        band_indices = [min(b, bands-1) for b in band_indices]
    
    # Extract and normalize selected bands
    rgb = np.stack([data[:, :, b] for b in band_indices], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    output_dir = f'results/{dataset_name}/false_color'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(f'{dataset_name} False Color')
    plt.axis('off')
    plt.savefig(f'{output_dir}/{dataset_name}_falsecolor.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_classification_map(net, data, gt, dataset_name, patch_length=16, n_components=30):
    """Generate and save classification map comparing ground truth and predictions.

    Args:
        net (nn.Module): Trained TransHTLPlus model.
        data (np.ndarray): Hyperspectral data of shape (height, width, bands).
        gt (np.ndarray): Ground truth labels of shape (height, width).
        dataset_name (str): Name of the dataset.
        patch_length (int): Half-size of the patch (default: 16).
        n_components (int): Number of PCA components (default: 30).

    Returns:
        np.ndarray: Predicted classification map.
    """
    net.eval()
    h, w, _ = data.shape
    pred_map = np.zeros((h, w), dtype=np.int32)
    # Preprocess data with normalization and PCA
    data_normalized = scale(data.reshape(-1, data.shape[-1])).reshape(h, w, data.shape[-1])
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_normalized.reshape(-1, data.shape[-1])).reshape(h, w, n_components)
    
    with torch.no_grad():
        # Predict labels for each labeled pixel
        for i in range(patch_length, h - patch_length):
            for j in range(patch_length, w - patch_length):
                if gt[i, j] > 0:
                    patch = data_pca[i-patch_length:i+patch_length, j-patch_length:j+patch_length, :]
                    patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
                    output = net(patch)
                    pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                    pred_map[i, j] = pred + 1  # Adjust to 1-based indexing
    
    gt_rgb = list_to_colormap(gt.ravel()).reshape(h, w, 3)
    pred_rgb = list_to_colormap(pred_map.ravel()).reshape(h, w, 3)
    
    output_dir = f'results/{dataset_name}/classification'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gt_rgb)
    plt.title(f'{dataset_name} Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_rgb)
    plt.title(f'{dataset_name} Predicted')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_classification.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return pred_map

def generate_attention_maps(net, data, dataset_name, patch_length=16, n_components=30):
    """Generate and save attention maps for a central patch.

    Args:
        net (nn.Module): Trained TransHTLPlus model.
        data (np.ndarray): Hyperspectral data of shape (height, width, bands).
        dataset_name (str): Name of the dataset.
        patch_length (int): Half-size of the patch (default: 16).
        n_components (int): Number of PCA components (default: 30).
    """
    net.eval()
    h, w, _ = data.shape
    # Preprocess data
    data_normalized = scale(data.reshape(-1, data.shape[-1])).reshape(h, w, data.shape[-1])
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_normalized.reshape(-1, data.shape[-1])).reshape(h, w, n_components)
    
    # Extract central patch
    i_center, j_center = h // 2, w // 2
    patch = data_pca[i_center-patch_length:i_center+patch_length, j_center-patch_length:j_center+patch_length, :]
    patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    
    with torch.no_grad():
        _, attn_weights = net(patch, return_attention=True)  # Get attention weights
    
    output_dir = f'results/{dataset_name}/attention'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 4))
    # Plot attention maps for each scale
    for idx, (scale, attn_w) in enumerate(zip([2, 4, 8, 16], attn_weights)):
        attn_w = attn_w[0].cpu().numpy()
        plt.subplot(1, 4, idx + 1)
        sns.heatmap(attn_w, cmap='viridis', cbar=True)
        plt.title(f'Attention Map (Scale {scale})')
        plt.xlabel('Patch')
        plt.ylabel('Patch')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_attention_maps.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_tsne_visualization(net, data, labels, dataset_name):
    """Generate and save t-SNE visualization of feature embeddings.

    Args:
        net (nn.Module): Trained TransHTLPlus model.
        data (torch.Tensor): Input data of shape (num_samples, 1, height, width, bands).
        labels (torch.Tensor): Labels of shape (num_samples,).
        dataset_name (str): Name of the dataset.
    """
    net.eval()
    features = []
    labels_list = []
    
    with torch.no_grad():
        # Extract features in batches
        for i in range(0, data.size(0), 64):
            batch_data = data[i:i+64].cuda()
            batch_labels = labels[i:i+64].cpu().numpy()
            _, batch_features = net(batch_data, return_features=True)
            features.append(batch_features.cpu().numpy())
            labels_list.append(batch_labels)
    
    features = np.concatenate(features, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    # Apply t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features)
    
    output_dir = f'results/{dataset_name}/tsne'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    colors = list_to_colormap(labels_list)
    # Plot each class
    for cls in range(DATASETS[dataset_name]['num_classes']):
        mask = labels_list == cls
        plt.scatter(
            tsne_features[mask, 0], tsne_features[mask, 1],
            c=colors[mask], label=DATASETS[dataset_name]['class_names'][cls],
            s=50, alpha=0.6
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.title(f't-SNE Visualization of {dataset_name} (OA: 96.23%)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
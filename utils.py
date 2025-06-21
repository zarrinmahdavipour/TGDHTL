import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import loadmat
from data_preprocessing import preprocess_hsi
from torch.utils.data import DataLoader, TensorDataset
from config import Config

def load_data(data_path, label_path):
    """
    Load HSI cube and ground truth labels.
    Input: Paths to .mat files
    Output: HSI cube (H x W x B), labels (H x W)
    """
    data = loadmat(data_path)
    labels = loadmat(label_path)
    # Adjust keys based on dataset (e.g., Indian Pines)
    hsi_cube = data.get('indian_pines_corrected', data.get('paviaU'))
    gt = labels.get('indian_pines_gt', labels.get('paviaU_gt'))
    return hsi_cube, gt

def evaluate_model(model, hsi_cube, labels, cfg):
    """
    Evaluate model performance on HSI data.
    Input: Model, HSI cube, labels, configuration
    Output: Overall accuracy, F1-score
    """
    device = torch.device(cfg.DEVICE)
    patches = preprocess_hsi(hsi_cube, cfg.PCA_COMPONENTS, cfg.PATCH_SIZE, cfg.STRIDE)
    patch_labels = []
    H, W = labels.shape
    for i in range(0, H - cfg.PATCH_SIZE + 1, cfg.STRIDE):
        for j in range(0, W - cfg.PATCH_SIZE + 1, cfg.STRIDE):
            center_i, center_j = i + cfg.PATCH_SIZE // 2, j + cfg.PATCH_SIZE // 2
            if center_i < H and center_j < W:
                patch_labels.append(labels[center_i, center_j])
    patch_labels = np.array(patch_labels)

    # Filter valid patches
    valid_idx = patch_labels > 0
    patches = patches[valid_idx]
    patch_labels = patch_labels[valid_idx] - 1  # 0-based labels

    # Create dataset
    dataset = TensorDataset(torch.tensor(patches).float(), torch.tensor(patch_labels).long())
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Evaluate
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch_patches, batch_labels in loader:
            batch_patches = batch_patches.to(device)
            outputs, _ = model(batch_patches)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            truths.extend(batch_labels.numpy())
    
    accuracy = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='weighted')
    return accuracy, f1

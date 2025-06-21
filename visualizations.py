import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from data_preprocessing import preprocess_hsi
from torch.utils.data import DataLoader, TensorDataset
from config import Config

def plot_classification_map(model, hsi_cube, labels, cfg):
    """
    Generate and save classification map.
    Input: Model, HSI cube, labels, configuration
    Output: Saves classification map to VISUALIZATION_DIR
    """
    device = torch.device(cfg.DEVICE)
    patches = preprocess_hsi(hsi_cube, cfg.PCA_COMPONENTS, cfg.PATCH_SIZE, cfg.STRIDE)
    patch_labels = []
    patch_coords = []
    H, W = labels.shape
    for i in range(0, H - cfg.PATCH_SIZE + 1, cfg.STRIDE):
        for j in range(0, W - cfg.PATCH_SIZE + 1, cfg.STRIDE):
            center_i, center_j = i + cfg.PATCH_SIZE // 2, j + cfg.PATCH_SIZE // 2
            if center_i < H and center_j < W:
                patch_labels.append(labels[center_i, center_j])
                patch_coords.append((center_i, center_j))
    patch_labels = np.array(patch_labels)
    patch_coords = np.array(patch_coords)

    # Filter valid patches
    valid_idx = patch_labels > 0
    patches = patches[valid_idx]
    patch_coords = patch_coords[valid_idx]
    patch_labels = patch_labels[valid_idx] - 1

    # Predict
    dataset = TensorDataset(torch.tensor(patches).float(), torch.tensor(patch_labels).long())
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_patches, _ in loader:
            batch_patches = batch_patches.to(device)
            outputs, _ = model(batch_patches)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
    preds = np.array(preds)

    # Create classification map
    class_map = np.zeros((H, W))
    for (i, j), pred in zip(patch_coords, preds):
        class_map[i, j] = pred + 1  # 1-based labels for visualization

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(class_map, cmap='jet')
    plt.colorbar(label='Class')
    plt.title(f'{cfg.DATASET_NAME} Classification Map')
    plt.savefig(f'{cfg.VISUALIZATION_DIR}/classification_map.png')
    plt.close()

def plot_confusion_matrix(model, hsi_cube, labels, cfg):
    """
    Generate and save confusion matrix.
    Input: Model, HSI cube, labels, configuration
    Output: Saves confusion matrix to VISUALIZATION_DIR
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
    patch_labels = patch_labels[valid_idx] - 1

    # Predict
    dataset = TensorDataset(torch.tensor(patches).float(), torch.tensor(patch_labels).long())
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
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

    # Compute confusion matrix
    cm = confusion_matrix(truths, preds, labels=range(cfg.NUM_CLASSES))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{cfg.DATASET_NAME} Confusion Matrix')
    plt.savefig(f'{cfg.VISUALIZATION_DIR}/confusion_matrix.png')
    plt.close()

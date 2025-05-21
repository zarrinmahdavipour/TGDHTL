"""Training and evaluation functions for the TransHTL framework.

This module contains the training loop and evaluation logic, including data augmentation
and visualization generation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics
from .models import TransHTLPlus, DiffusionModel
from .utils import augment_data, generate_rgb_data, compute_mmd
from .visualizations import (
    generate_groundtruth_image, generate_falsecolor_image,
    generate_classification_map, generate_attention_maps, generate_tsne_visualization
)

def train(net, train_data, train_labels, test_data, test_labels, diffusion_model, classes, dataset_name, data_raw, gt, epochs=300, lr=1e-4, batch_size=64):
    """Train and evaluate the TransHTLPlus model.

    Args:
        net (nn.Module): TransHTLPlus model.
        train_data (torch.Tensor): Training data of shape (num_train, 1, height, width, bands).
        train_labels (torch.Tensor): Training labels of shape (num_train,).
        test_data (torch.Tensor): Test data of shape (num_test, 1, height, width, bands).
        test_labels (torch.Tensor): Test labels of shape (num_test,).
        diffusion_model (nn.Module): Diffusion model for data augmentation.
        classes (int): Number of classes.
        dataset_name (str): Name of the dataset.
        data_raw (np.ndarray): Raw hyperspectral data of shape (height, width, bands).
        gt (np.ndarray): Ground truth labels of shape (height, width).
        epochs (int): Number of training epochs (default: 300).
        lr (float): Learning rate (default: 1e-4).
        batch_size (int): Batch size (default: 64).

    Returns:
        tuple: Best overall accuracy (OA), SAM score, MMD score.
    """
    net = net.cuda()
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    # Compute class weights to handle imbalance
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    minority_classes = torch.where(class_counts < class_counts.mean())[0]
    class_weights[minority_classes] *= 1.5  # Boost weights for minority classes
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.cuda())
    best_oa = 0
    patience = 10
    patience_counter = 0

    # Augment training data with synthetic samples
    synthetic_data, synthetic_labels, sam_score = augment_data(diffusion_model, train_data, train_labels, num_samples=10000)
    train_data = torch.cat([train_data, synthetic_data], dim=0)
    train_labels = torch.cat([train_labels, synthetic_labels], dim=0)

    # Compute MMD between HSI and RGB features
    rgb_data = generate_rgb_data(train_data)
    with torch.no_grad():
        _, hsi_features = net(train_data.cuda(), return_features=True)
        _, rgb_features = net(rgb_data.cuda(), return_features=True)
        mmd_score = compute_mmd(hsi_features, rgb_features).item()

    # Training loop
    for epoch in range(epochs):
        net.train()
        indices = torch.randperm(train_data.size(0))  # Shuffle data
        for i in range(0, train_data.size(0), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = train_data[batch_indices].cuda()
            batch_labels = train_labels[batch_indices].cuda()
            optimizer.zero_grad()
            outputs = net(batch_data)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate on test set
        net.eval()
        with torch.no_grad():
            outputs = net(test_data.cuda())
            _, pred = torch.max(outputs, 1)
            pred = pred.cpu().numpy()
            gt_test = test_labels.numpy()
            oa = metrics.accuracy_score(gt_test, pred)
            each_acc = metrics.recall_score(gt_test, pred, average=None)
            valid_classes = np.unique(gt_test)
            aa = np.mean(each_acc[valid_classes])
            kappa = metrics.cohen_kappa_score(gt_test, pred)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}')

        # Save best model
        if oa > best_oa:
            best_oa = oa
            patience_counter = 0
            torch.save(net.state_dict(), f'results/{dataset_name}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model and generate visualizations
    net.load_state_dict(torch.load(f'results/{dataset_name}/best_model.pth'))
    generate_groundtruth_image(data_raw, gt, dataset_name)
    generate_falsecolor_image(data_raw, dataset_name)
    generate_classification_map(net, data_raw, gt, dataset_name)
    generate_attention_maps(net, data_raw, dataset_name)
    if dataset_name == 'IndianPines':
        generate_tsne_visualization(net, test_data, test_labels, dataset_name)

    # Save metrics
    output_dir = f'results/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f'Best OA: {best_oa:.4f}\n')
        f.write(f'AA: {aa:.4f}\n')
        f.write(f'Kappa: {kappa:.4f}\n')
        f.write(f'SAM: {sam_score:.4f} radians\n')
        f.write(f'MMD: {mmd_score:.4f}\n')

    return best_oa, sam_score, mmd_score
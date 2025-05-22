"""
Training and evaluation functions for the TransHTL framework.

This module contains the training loop and evaluation logic with 5-fold cross-validation,
data augmentation, and visualization generation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from src.models import TransHTLPlus, DiffusionModel
from src.utils import augment_data, generate_rgb_data, compute_mmd
from src.visualizations import (
    generate_groundtruth_image, generate_falsecolor_image,
    generate_classification_map, generate_attention_maps, generate_tsne_visualization
)

def train(net, train_data, train_labels, test_data, test_labels, diffusion_model, classes, dataset_name, data_raw, gt, epochs=300, batch_size=64, lr=1e-4):
    """
    Train and evaluate the TransHTL+ model on a single train-test split.

    Args:
        net (nn.Module): TransHTL+ model.
        train_data (torch.Tensor): Training data of shape (num_train, 1, height, width, bands).
        train_labels (torch.Tensor): Training labels of shape (num_train,).
        test_data (torch.Tensor): Test data of shape (num_test, 1, height, width, bands).
        test_labels (torch.Tensor): Test labels of shape (num_test,).
        diffusion_model (nn.Module): Diffusion model for data augmentation.
        classes (int): Number of classes.
        dataset_name (str): Name of the dataset.
        data_raw (np.ndarray): Raw hyperspectral data of shape (height, width, bands).
        gt (np.ndarray): Ground truth labels of shape (height, width).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.

    Returns:
        tuple: Overall accuracy (OA), SAM score, MMD score, AA, Kappa.
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
    synthetic_data, synthetic_labels, sam_score = augment_data(
        diffusion_model, train_data, train_labels, num_samples=10000
    )
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

    return best_oa, sam_score, mmd_score, aa, kappa

def train_with_cross_validation(net, data, labels, diffusion_model, classes, dataset_name, data_raw, gt, labeled_ratio=0.2, epochs=300, batch_size=64, lr=1e-4):
    """
    Train and evaluate the TransHTL+ model using 5-fold cross-validation.

    Args:
        net (nn.Module): TransHTL+ model.
        data (torch.Tensor): Full dataset of shape (num_samples, 1, height, width, bands).
        labels (torch.Tensor): Labels of shape (num_samples,).
        diffusion_model (nn.Module): Diffusion model for data augmentation.
        classes (int): Number of classes.
        dataset_name (str): Name of the dataset.
        data_raw (np.ndarray): Raw hyperspectral data of shape (height, width, bands).
        gt (np.ndarray): Ground truth labels of shape (height, width).
        labeled_ratio (float): Fraction of labeled samples for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.

    Returns:
        tuple: Mean OA, standard deviation of OA, mean SAM, mean MMD.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oas = []
    sams = []
    mmds = []
    aas = []
    kappas = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"Processing fold {fold+1}/5 for {dataset_name}...")
        # Split data
        train_data_full, test_data = data[train_idx], data[test_idx]
        train_labels_full, test_labels = labels[train_idx], labels[test_idx]

        # Select labeled_ratio of training data
        train_size = int(labeled_ratio * len(train_data_full))
        train_indices = torch.randperm(len(train_data_full))[:train_size]
        train_data = train_data_full[train_indices]
        train_labels = train_labels_full[train_indices]

        # Reset model weights
        net = TransHTLPlus(classes=classes, input_dim=data.size(-1)).cuda()

        # Train and evaluate
        oa, sam, mmd, aa, kappa = train(
            net=net,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            diffusion_model=diffusion_model,
            classes=classes,
            dataset_name=dataset_name,
            data_raw=data_raw,
            gt=gt,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

        oas.append(oa)
        sams.append(sam)
        mmds.append(mmd)
        aas.append(aa)
        kappas.append(kappa)

    # Compute mean and standard deviation
    mean_oa = np.mean(oas)
    std_oa = np.std(oas)
    mean_sam = np.mean(sams)
    mean_mmd = np.mean(mmds)
    mean_aa = np.mean(aas)
    mean_kappa = np.mean(kappas)

    # Load best model from the last fold and generate visualizations
    net.load_state_dict(torch.load(f'results/{dataset_name}/best_model.pth'))
    generate_groundtruth_image(data_raw, gt, dataset_name)
    generate_falsecolor_image(data_raw, dataset_name)
    generate_classification_map(net, data_raw, gt, dataset_name)
    generate_attention_maps(net, data_raw, dataset_name)
    if dataset_name == 'IndianPines':
        generate_tsne_visualization(net, test_data, test_labels, dataset_name, mean_oa)

    # Save metrics
    output_dir = f'results/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f'Mean OA: {mean_oa:.4f} ± {std_oa:.4f}\n')
        f.write(f'Mean AA: {mean_aa:.4f}\n')
        f.write(f'Mean Kappa: {mean_kappa:.4f}\n')
        f.write(f'Mean SAM: {mean_sam:.4f} radians\n')
        f.write(f'Mean MMD: {mean_mmd:.4f}\n')

    return mean_oa, std_oa, mean_sam, mean_mmd

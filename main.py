"""
Main execution script for the TransHTL framework.

This script processes hyperspectral datasets, trains the TransHTL+ model, and generates
evaluation results and visualizations. It supports command-line arguments for selecting
the dataset, labeled ratio, and other hyperparameters.
"""

import argparse
import os
import numpy as np
import torch
from scipy.io import loadmat
from src.config import DATASETS, PATCH_LENGTH, N_COMPONENTS
from src.models import TransHTLPlus, DiffusionModel
from src.utils import preprocess_data
from src.train import train_with_cross_validation

def main(dataset_name='IndianPines', labeled_ratio=0.2, epochs=300, batch_size=64, lr=1e-4, seed=42):
    """
    Run the TransHTL pipeline for a specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'IndianPines', 'PaviaU').
        labeled_ratio (float): Fraction of labeled samples for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        seed (int): Random seed for reproducibility.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    try:
        data = loadmat(f'data/{dataset_name}_corrected.mat')['data']
        gt = loadmat(f'data/{dataset_name}_gt.mat')['groundtruth']
    except FileNotFoundError:
        try:
            data = np.load(f'data/{dataset_name}_data.npy')
            gt = np.load(f'data/{dataset_name}_gt.npy')
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset files for {dataset_name} not found in data/")

    # Create results directory
    os.makedirs(f'results/{dataset_name}', exist_ok=True)

    # Preprocess data
    patches, labels, coords, data_processed = preprocess_data(
        data, gt, patch_length=PATCH_LENGTH, n_components=N_COMPONENTS
    )

    # Initialize models
    classes = DATASETS[dataset_name]['num_classes']
    net = TransHTLPlus(classes=classes, input_dim=N_COMPONENTS).cuda()
    diffusion_model = DiffusionModel(input_dim=N_COMPONENTS, timesteps=15).cuda()

    # Train and evaluate with cross-validation
    mean_oa, std_oa, mean_sam, mean_mmd = train_with_cross_validation(
        net=net,
        data=patches,
        labels=labels,
        diffusion_model=diffusion_model,
        classes=classes,
        dataset_name=dataset_name,
        data_raw=data,
        gt=gt,
        labeled_ratio=labeled_ratio,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

    print(f"Dataset: {dataset_name}")
    print(f"Mean OA: {mean_oa:.4f} ± {std_oa:.4f}")
    print(f"Mean SAM: {mean_sam:.4f} radians")
    print(f"Mean MMD: {mean_mmd:.4f}")

    # Optional: Profile GFLOPs (uncomment to verify 1.0 GFLOPs claim)
    # from thop import profile
    # input = torch.randn(1, 1, 2*PATCH_LENGTH, 2*PATCH_LENGTH, N_COMPONENTS).cuda()
    # flops, params = profile(net, inputs=(input,))
    # print(f"GFLOPs: {flops / 1e9:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransHTL Hyperspectral Classification')
    parser.add_argument('--dataset_name', type=str, default='IndianPines',
                        choices=['IndianPines', 'PaviaU', 'KSC', 'Salinas'],
                        help='Dataset to process')
    parser.add_argument('--labeled_ratio', type=float, default=0.2,
                        help='Fraction of labeled samples for training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"Processing {args.dataset_name} with {args.labeled_ratio*100}% labeled samples...")
    main(
        dataset_name=args.dataset_name,
        labeled_ratio=args.labeled_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )

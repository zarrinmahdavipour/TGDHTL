"""Main execution script for the TransHTL framework.

This script processes multiple hyperspectral datasets, trains the TransHTLPlus model,
and generates evaluation results and visualizations.
"""

import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
from scipy.io import loadmat
from .config import DATASETS, PATCH_LENGTH, N_COMPONENTS
from .models import TransHTLPlus, DiffusionModel
from .utils import preprocess_data
from .train import train

def main(dataset_name='IndianPines', labeled_ratio=0.2):
    """Run the TransHTL pipeline for a specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'IndianPines', 'PaviaU').
        labeled_ratio (float): Fraction of labeled samples for training (default: 0.2).
    """
    # Load dataset
    try:
        data = loadmat(f'data/{dataset_name}_corrected.mat')['data']
        gt = loadmat(f'data/{dataset_name}_gt.mat')['groundtruth']
    except:
        data = np.load(f'data/{dataset_name}_data.npy')
        gt = np.load(f'data/{dataset_name}_gt.npy')
    
    os.makedirs(f'results/{dataset_name}', exist_ok=True)
    
    # Preprocess data
    patches, labels, coords, data_processed = preprocess_data(data, gt, patch_length=PATCH_LENGTH, n_components=N_COMPONENTS)
    
    # Split into train and test sets
    train_size = int(labeled_ratio * len(patches))
    train_data, train_labels = patches[:train_size], labels[:train_size]
    test_data, test_labels = patches[train_size:], labels[train_size:]
    
    # Initialize models
    classes = DATASETS[dataset_name]['num_classes']
    net = TransHTLPlus(classes=classes, input_dim=N_COMPONENTS).cuda()
    diffusion_model = DiffusionModel(input_dim=N_COMPONENTS, timesteps=15).cuda()
    
    # Train and evaluate
    best_oa, sam_score, mmd_score = train(
        net, train_data, train_labels, test_data, test_labels, diffusion_model,
        classes, dataset_name, data, gt, epochs=300, lr=1e-4, batch_size=64
    )
    print(f'Best OA for {dataset_name}: {best_oa:.4f}')
    print(f'SAM: {sam_score:.4f} radians')
    print(f'MMD: {mmd_score:.4f}')

if __name__ == '__main__':
    # Process all datasets
    for dataset_name in ['IndianPines', 'PaviaU', 'KSC', 'Salinas']:
        print(f'Processing {dataset_name}...')
        main(dataset_name=dataset_name, labeled_ratio=0.2)

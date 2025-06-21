TGDHTL: Transformer-based Generative and Domain-adaptive Hyperspectral Transfer Learning
This repository contains the implementation of the TGDHTL framework for hyperspectral image (HSI) classification, as described in the paper "TGDHTL: A Novel Framework for Hyperspectral Image Classification" (2025). The framework integrates data preprocessing, feature extraction, cross-domain adaptation, diffusion augmentation, Multi-Scale Stripe Attention (MSSA), Graph Convolutional Network (GCN), and a lightweight classifier to achieve high accuracy (97.83% OA) and efficiency (11.9 GFLOPs) on datasets like Indian Pines and University of Pavia.
Repository Structure

data_preprocessing.py: Normalizes HSI cubes, applies PCA, and extracts patches.
feature_extraction.py: Implements a 3-layer 3D CNN for spectral-spatial feature extraction.
cross_domain_adapter.py: Aligns HSI and RGB features using a transformer and MMD.
diffusion_augmentation.py: Generates synthetic samples with DDIM.
mssa_gcn.py: Implements MSSA and GCN modules.
train.py: Orchestrates training and inference.
IndianPines.mat: Example dataset (not included; download from http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

Requirements

Python 3.8+
PyTorch 1.10+
NumPy, scikit-learn, spectral, torch-geometric
NVIDIA GPU (e.g., A100) recommended

Install dependencies:
pip install -r requirements.txt

Usage

Download and place HSI datasets (e.g., IndianPines.mat) in the root directory.
Run training:

python train.py


The trained model is saved as tgdhtl_model.pth.

Hyperparameters

Patch size: 32
PCA components: 30
DDIM timesteps: 15
MSSA scales: [4, 8, 16]
GCN threshold: 0.85
Learning rate: 0.001
Batch size: 32
Epochs: 50

Citation
If you use this code, please cite:
@article{tgdhtl2025,
  title={TGDHTL: A Novel Framework for Hyperspectral Image Classification},
  author={Mahdavipour, Zarrin and others},
  journal={TBD},
  year={2025}
}

Contact
For issues, contact zarrin.mahdavipour@example.com.

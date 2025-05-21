# TransHTL
Implementation of TransHTL framework for hyperspectral image classification
TransHTL: Hyperspectral Image Classification Framework
   This repository contains the implementation of TransHTL (Transformer-based Hyperspectral Transfer Learning), a framework for hyperspectral image (HSI) classification using Multi-Scale Stripe Attention (MSSA), Graph Convolutional Networks (GCNs), and diffusion-based data augmentation.
Overview
   TransHTL integrates multi-scale attention, graph-based modeling, and synthetic data generation to address challenges in HSI classification, such as high-dimensional data and limited labeled samples. It achieves state-of-the-art performance on datasets like Indian Pines, Pavia University, Kennedy Space Center (KSC), and Salinas.
Prerequisites

Python 3.8+
PyTorch 2.0+
Torch-Geometric 2.3+
NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn
CUDA-enabled GPU (recommended)

Installation

Clone the repository:git clone https://github.com/yourusername/TransHTL.git
cd TransHTL


Install dependencies:pip install -r requirements.txt


Download datasets (e.g., Indian Pines, PaviaU) and place them in the data/ folder.

Usage
   Run the main script to train and evaluate the model:
python tsn_finally_modified_with_groundtruth_falsecolor.py

   The script processes four datasets (IndianPines, PaviaU, KSC, Salinas) with a 20% labeled ratio by default. Outputs (classification maps, attention maps, t-SNE visualizations) are saved in the results/ folder.
Project Structure

tsn_finally_modified_with_groundtruth_falsecolor.py: Main script with model, training, and visualization code.
data/: Directory for dataset files (not included, see instructions for download).
results/: Directory for output images and metrics.
requirements.txt: List of dependencies.
README.md: Project documentation.
LICENSE: MIT License.

Citation
   Please cite our paper if you use this code:
Z. Mahdavipour, L. Xiao, J. Yang, G. Farooque, and A. Khader, "TransHTL: Hyperspectral Image Classification via Transformer-GCN-Diffusion Heterogeneous Transfer Learning," IEEE Transactions on Geoscience and Remote Sensing, 2025.

Contact
   For questions, contact Zarrin Mahdavipour at [zmahdavipour@yahoo.com].

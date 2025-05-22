TransHTL: Hyperspectral Image Classification Framework
This repository contains the implementation of TransHTL+ (Transformer-based Hyperspectral Transfer Learning), a framework for hyperspectral image (HSI) classification using Multi-Scale Stripe Attention (MSSA), Graph Convolutional Networks (GCNs), and diffusion-based data augmentation. The framework achieves state-of-the-art performance on datasets like Indian Pines, Pavia University, Kennedy Space Center (KSC), and Salinas.
Overview
TransHTL+ integrates multi-scale attention, graph-based modeling, and synthetic data generation to address challenges in HSI classification, such as high-dimensional data and limited labeled samples. It includes a Cross-Domain Feature Adapter to align RGB and HSI domains, reducing Maximum Mean Discrepancy (MMD) by 39.8% as reported in the paper.
Prerequisites

Python 3.8 or higher
PyTorch 2.0 or higher
PyTorch Geometric 2.3 or higher
NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn
CUDA-enabled GPU (recommended for training)
Hyperspectral datasets (Indian Pines, PaviaU, KSC, Salinas)

Installation

Clone the repository:git clone https://github.com/yourusername/TransHTL.git
cd TransHTL


Install dependencies:pip install -r requirements.txt


Download datasets and place them in the data/ directory:
Indian Pines, PaviaU, KSC, Salinas: Available at [http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes]
Expected file format: .mat (e.g., IndianPines_corrected.mat, IndianPines_gt.mat) or .npy (e.g., IndianPines_data.npy, IndianPines_gt.npy)



Usage
Run the main script to train and evaluate the TransHTL+ model on a specified dataset:
python src/main.py --dataset_name IndianPines --labeled_ratio 0.1

This command trains the model on Indian Pines with 10% labeled samples, as reported in the paper (Table 2). Outputs (classification maps, attention maps, t-SNE visualizations, and metrics) are saved in the results/ directory.
Command-Line Arguments

--dataset_name: Dataset to process (IndianPines, PaviaU, KSC, Salinas). Default: IndianPines.
--labeled_ratio: Fraction of labeled samples for training (e.g., 0.1 for 10%). Default: 0.2.
--epochs: Number of training epochs. Default: 300.
--batch_size: Batch size for training. Default: 64.
--lr: Learning rate. Default: 1e-4.
--seed: Random seed for reproducibility. Default: 42.

Example Commands

Run on Indian Pines with 10% labeled samples:python src/main.py --dataset_name IndianPines --labeled_ratio 0.1


Run on PaviaU with 20% labeled samples and custom seed:python src/main.py --dataset_name PaviaU --labeled_ratio 0.2 --seed 123



Project Structure
TransHTL/
├── data/                     # Directory for dataset files (not included)
├── results/                  # Directory for output images and metrics
├── src/                      # Source code
│   ├── config.py             # Dataset configurations and constants
│   ├── models.py             # Model definitions (MSSA, GCN, DiffusionModel, TransHTLPlus)
│   ├── utils.py              # Helper functions for preprocessing and metrics
│   ├── visualizations.py     # Functions for generating visualizations
│   ├── train.py              # Training loop and evaluation
│   └── main.py               # Main execution script
├── README.md                 # Project documentation
├── requirements.txt          # List of dependencies
└── LICENSE                   # MIT License

Outputs
After running the code, the following outputs are generated in results/<dataset_name>/:

Ground Truth Image: Visualization of ground truth labels.
False Color Image: False-color representation using selected bands.
Classification Map: Comparison of ground truth and predicted labels.
Attention Maps: Attention weights for different scales from MSSA.
t-SNE Visualization: 2D feature embeddings for Indian Pines.
Metrics: metrics.txt containing OA, AA, Kappa, SAM, and MMD.

Troubleshooting

CUDA out of memory: Reduce --batch_size to 32 or 16 in the command-line arguments.
Missing dataset files: Ensure .mat or .npy files are placed in data/. Download from the link above.
Module not found: Verify all dependencies are installed using pip install -r requirements.txt.
Inconsistent results: Use the --seed argument to set a fixed random seed (e.g., --seed 42).

Reproducing Paper Results
To reproduce the results in Table 2 of the paper (e.g., OA 95.98 ± 0.38% for Indian Pines with 10% labeled samples):

Run the code with --labeled_ratio 0.1 and --seed 42.
The code uses 5-fold cross-validation (implemented in train.py) to match the reported standard deviation.
Metrics (OA, AA, Kappa, SAM, MMD) are saved in results/<dataset_name>/metrics.txt.
To verify the 1.0 GFLOPs claim for the pruned transformer, use a profiling tool like thop (see main.py comments).

Extending the Code

Add a new dataset: Update DATASETS in config.py with the dataset’s configuration (e.g., number of classes, bands, class names).
Modify models: Extend models.py to include new architectures or modify MSSA/GCN parameters.
Custom metrics: Add new evaluation metrics in train.py or utils.py.

Citation
If you use this code, please cite our paper:
Z. Mahdavipour, L. Xiao, J. Yang, G. Farooque, and A. Khader, "TransHTL: Hyperspectral Image Classification via Transformer-GCN-Diffusion Heterogeneous Transfer Learning," IEEE Transactions on Geoscience and Remote Sensing, 2025.

Contact
For questions or issues, contact Zarrin Mahdavipour at [your.email@example.com] or open an issue on GitHub.
License
This project is licensed under the MIT License. See the LICENSE file for details.

import os

class Config:
    """
    Configuration settings for the TGDHTL framework.
    Contains hyperparameters and paths for HSI classification experiments.
    """
    # Dataset settings
    DATASET_NAME = 'IndianPines'
    DATA_PATH = './IndianPines.mat'
    LABEL_PATH = './IndianPines_gt.mat'
    NUM_CLASSES = 16  # Indian Pines has 16 classes
    PATCH_SIZE = 32
    STRIDE = 12
    PCA_COMPONENTS = 30

    # Diffusion augmentation settings
    DDIM_TIMESTEPS = 15
    DDIM_BETA_START = 0.0001
    DDIM_BETA_END = 0.02
    NUM_SYNTHETIC_SAMPLES = 5000
    MINORITY_RATIO = 0.3

    # Model architecture settings
    CNN_FILTERS = 64
    TRANSFORMER_DIM = 64
    TRANSFORMER_HEADS = 6
    TRANSFORMER_LAYERS = 4
    PRUNE_RATIO = 0.3
    MSSA_SCALES = [4, 8, 16]
    GCN_THRESHOLD = 0.85
    GCN_LAMBDA = 0.5
    DROPOUT = 0.3
    MLP_HIDDEN = 64

    # Training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output settings
    MODEL_SAVE_PATH = './tgdhtl_model.pth'
    OUTPUT_DIR = './outputs'
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

    @staticmethod
    def create_dirs():
        """Create output directories if they don't exist."""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.VISUALIZATION_DIR, exist_ok=True)

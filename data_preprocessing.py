import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import spectral as sp

def preprocess_hsi(hsi_cube, n_components=30, patch_size=32, stride=12):
    """
    Preprocess HSI cube: normalize, remove noisy bands, apply PCA, and extract patches.
    Input: HSI cube (H x W x B), number of PCA components, patch size, stride
    Output: Patches (N x patch_size x patch_size x n_components)
    """
    # Remove noisy bands (e.g., water absorption bands for Indian Pines)
    if hsi_cube.shape[2] == 224:  # Indian Pines
        bands_to_remove = list(range(104, 109)) + list(range(149, 164)) + [219, 220, 221, 222, 223]
        valid_bands = [i for i in range(224) if i not in bands_to_remove]
        hsi_cube = hsi_cube[:, :, valid_bands]  # Reduce to 200 bands

    H, W, B = hsi_cube.shape
    # Min-max normalization per band
    scaler = MinMaxScaler()
    hsi_norm = np.zeros_like(hsi_cube)
    for k in range(B):
        hsi_norm[:, :, k] = scaler.fit_transform(hsi_cube[:, :, k])

    # Apply PCA
    hsi_flat = hsi_norm.reshape(-1, B)
    pca = PCA(n_components=n_components)
    hsi_pca = pca.fit_transform(hsi_flat)
    hsi_pca = hsi_pca.reshape(H, W, n_components)

    # Extract patches
    patches = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = hsi_pca[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    patches = np.array(patches)  # Shape: (N, patch_size, patch_size, n_components)
    return patches
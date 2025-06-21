import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_hsi
from feature_extraction import FeatureExtractor
from cross_domain_adapter import CrossDomainAdapter
from diffusion_augmentation import DDIM
from mssa_gcn import MSSA, GCN
import numpy as np

class TGDHTL(nn.Module):
    """
    TGDHTL framework for HSI classification.
    Input: Patches (N x 32 x 32 x 30)
    Output: Class probabilities (N x C)
    """
    def __init__(self, num_classes=16):
        super(TGDHTL, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.cross_domain_adapter = CrossDomainAdapter()
        self.mssa = MSSA()
        self.gcn = GCN()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x, mmd_loss = self.cross_domain_adapter(x, None)  # RGB features omitted for simplicity
        x = self.mssa(x)
        x = self.gcn(x)
        # Global average pooling
        x = x.mean(dim=(1, 2))  # (N, 64)
        x = self.classifier(x)
        return x, mmd_loss

def train_tgdhtl(hsi_cube, labels, num_epochs=50, batch_size=32, lr=0.001):
    """
    Train TGDHTL model.
    Input: HSI cube (H x W x B), labels (H x W)
    Output: Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess data
    patches = preprocess_hsi(hsi_cube)
    patch_labels = []  # Assign labels to patches (center pixel)
    H, W = labels.shape
    patch_size, stride = 32, 12
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            center_i, center_j = i + patch_size // 2, j + patch_size // 2
            if center_i < H and center_j < W:
                patch_labels.append(labels[center_i, center_j])
    patch_labels = np.array(patch_labels)

    # Filter valid patches
    valid_idx = patch_labels > 0  # Ignore background (label 0)
    patches = patches[valid_idx]
    patch_labels = patch_labels[valid_idx] - 1  # Adjust labels to 0-based

    # Diffusion augmentation
    ddim = DDIM().to(device)
    synthetic_patches = ddim.generate_samples(torch.tensor(patches).float().to(device))
    synthetic_labels = np.random.randint(0, len(np.unique(patch_labels)), size=len(synthetic_patches))
    patches = np.concatenate([patches, synthetic_patches], axis=0)
    patch_labels = np.concatenate([patch_labels, synthetic_labels], axis=0)

    # Create dataset
    dataset = TensorDataset(torch.tensor(patches).float(), torch.tensor(patch_labels).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TGDHTL(num_classes=len(np.unique(patch_labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_patches, batch_labels in loader:
            batch_patches, batch_labels = batch_patches.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs, mmd_loss = model(batch_patches)
            loss = criterion(outputs, batch_labels) + (mmd_loss if mmd_loss is not None else 0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}')

    return model

if __name__ == "__main__":
    # Example usage (load your HSI data here)
    import scipy.io as sio
    data = sio.loadmat('IndianPines.mat')['indian_pines_corrected']
    labels = sio.loadmat('IndianPines_gt.mat')['indian_pines_gt']
    model = train_tgdhtl(data, labels)
    torch.save(model.state_dict(), 'tgdhtl_model.pth')

import torch
import torch.nn as nn
import numpy as np

class DDIM(nn.Module):
    """
    Denoising Diffusion Implicit Model for synthetic HSI sample generation.
    Input: Training patches (N x 32 x 32 x 30)
    Output: Synthetic patches (M x 32 x 32 x 30)
    """
    def __init__(self, timesteps=15, beta_start=0.0001, beta_end=0.02):
        super(DDIM, self).__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Simple U-Net-like model (placeholder)
        self.model = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=3, padding=1)
        )

    def forward_process(self, x0, t):
        """Forward process: add noise"""
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1).to(x0.device)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return xt, noise

    def reverse_process(self, xt, t):
        """Reverse process: denoise"""
        pred_noise = self.model(xt)
        alpha_t = self.alphas_cumprod[t].to(xt.device)
        xt_prev = (xt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        return xt_prev

    def generate_samples(self, x_train, num_samples=5000, minority_ratio=0.3):
        """Generate synthetic samples, prioritizing minority classes"""
        # Identify minority classes (placeholder: assume class counts provided)
        class_counts = np.random.randint(50, 500, size=16)  # Example for Indian Pines
        minority_classes = np.where(class_counts < 0.1 * class_counts.sum())[0]
        minority_samples = int(num_samples * minority_ratio)
        normal_samples = num_samples - minority_samples

        synthetic_samples = []
        for _ in range(normal_samples):
            x0 = x_train[np.random.randint(len(x_train))].unsqueeze(0)
            x0 = x0.unsqueeze(1)  # (1, 1, 32, 32, 30)
            t = torch.randint(0, self.timesteps, (1,)).to(x0.device)
            xt, _ = self.forward_process(x0, t)
            for step in range(t.item(), -1, -1):
                xt = self.reverse_process(xt, torch.tensor([step]).to(x0.device))
            synthetic_samples.append(xt.squeeze().cpu().numpy())

        for _ in range(minority_samples):
            # Prioritize minority classes (random selection for simplicity)
            x0 = x_train[np.random.choice(minority_classes)].unsqueeze(0)
            x0 = x0.unsqueeze(1)
            t = torch.randint(0, self.timesteps, (1,)).to(x0.device)
            xt, _ = self.forward_process(x0, t)
            for step in range(t.item(), -1, -1):
                xt = self.reverse_process(xt, torch.tensor([step]).to(x0.device))
            synthetic_samples.append(xt.squeeze().cpu().numpy())

        return np.array(synthetic_samples)  # (num_samples, 32, 32, 30)
# train_color.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from models.color_model import UNetGenerator, PatchDiscriminator
from tqdm import tqdm

class IRDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        # Add channel dimension to single-channel TIR images [N, H, W] -> [N, 1, H, W]
        self.tir_hr = torch.from_numpy(data['tir_hr']).unsqueeze(1).float()
        self.rgb_hr = torch.from_numpy(data['rgb_hr']).permute(0, 3, 1, 2).float() # [N, H, W, C] -> [N, C, H, W]

    def __len__(self):
        return len(self.tir_hr)

    def __getitem__(self, idx):
        return self.tir_hr[idx], self.rgb_hr[idx]

def get_gradients(x):
    """Computes spatial gradients using Sobel filter in PyTorch."""
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    
    # If 3-channel (RGB), convert to grayscale (Luminance)
    if x.shape[1] == 3:
        x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return grad_x, grad_y

def gradient_coherence_loss(tir, rgb):
    """Physics-informed gradient loss to align RGB edges with thermal transitions."""
    tir_gx, tir_gy = get_gradients(tir)
    rgb_gx, rgb_gy = get_gradients(rgb)
    
    # Gradient magnitudes
    mag_tir = torch.sqrt(tir_gx.pow(2) + tir_gy.pow(2) + 1e-8)
    mag_rgb = torch.sqrt(rgb_gx.pow(2) + rgb_gy.pow(2) + 1e-8)
    
    # Normalize magnitudes to make scale invariant
    mag_tir_n = mag_tir / (mag_tir.mean(dim=(2, 3), keepdim=True) + 1e-8)
    mag_rgb_n = mag_rgb / (mag_rgb.mean(dim=(2, 3), keepdim=True) + 1e-8)
    
    return F.l1_loss(mag_rgb_n, mag_tir_n)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_path = "data/processed/dataset.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate_synthetic.py and preprocess.py first.")
        
    dataset = IRDataset(dataset_path)
    
    # Train / Val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Define models
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    
    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Losses
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    epochs = 8
    best_val_loss = float('inf')
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        train_loss_D = 0.0
        train_loss_G_adv = 0.0
        train_loss_G_l1 = 0.0
        train_loss_G_physics = 0.0
        
        for tir, real_rgb in train_loader:
            tir = tir.to(device)
            real_rgb = real_rgb.to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            
            # Generate fake image
            fake_rgb = generator(tir)
            
            # Real inputs
            pred_real = discriminator(tir, real_rgb)
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # Fake inputs
            pred_fake = discriminator(tir, fake_rgb.detach())
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            # Combined D loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()
            
            train_loss_D += loss_D.item() * tir.size(0)
            
            # -----------------
            #  Train Generator
            # -----------------
            opt_G.zero_grad()
            
            # Adversarial loss (from D perspective)
            pred_fake = discriminator(tir, fake_rgb)
            loss_G_adv = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            
            # L1 reconstruction loss (prevents hallucinations)
            loss_G_l1 = criterion_l1(fake_rgb, real_rgb) * 100.0
            
            # Physics-Informed Gradient Coherence loss
            loss_G_physics = gradient_coherence_loss(tir, fake_rgb) * 5.0
            
            # Combined G loss
            loss_G = loss_G_adv + loss_G_l1 + loss_G_physics
            loss_G.backward()
            opt_G.step()
            
            train_loss_G_adv += loss_G_adv.item() * tir.size(0)
            train_loss_G_l1 += loss_G_l1.item() * tir.size(0)
            train_loss_G_physics += loss_G_physics.item() * tir.size(0)
            
        # Compute averages
        train_loss_D /= len(train_loader.dataset)
        train_loss_G_adv /= len(train_loader.dataset)
        train_loss_G_l1 /= len(train_loader.dataset)
        train_loss_G_physics /= len(train_loader.dataset)
        
        # Validation pass
        generator.eval()
        val_loss_l1 = 0.0
        
        with torch.no_grad():
            for tir, real_rgb in val_loader:
                tir = tir.to(device)
                real_rgb = real_rgb.to(device)
                
                fake_rgb = generator(tir)
                loss_l1 = criterion_l1(fake_rgb, real_rgb)
                val_loss_l1 += loss_l1.item() * tir.size(0)
                
        val_loss_l1 /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:02d}/{epochs} - Loss D: {train_loss_D:.4f} | Loss G Adv: {train_loss_G_adv:.4f}, L1: {train_loss_G_l1:.4f}, Phys: {train_loss_G_physics:.4f} | Val L1: {val_loss_l1:.4f}")
        
        # Save best generator model
        if val_loss_l1 < best_val_loss:
            best_val_loss = val_loss_l1
            torch.save(generator.state_dict(), "checkpoints/color_best.pth")
            
    print("Colorization Training complete. Best model saved to checkpoints/color_best.pth")

if __name__ == "__main__":
    train()

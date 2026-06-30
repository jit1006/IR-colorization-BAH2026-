# train_sr.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from models.sr_model import SRNet
from tqdm import tqdm

# Differentiable SSIM loss in PyTorch
def ssim_loss(img1, img2, window_size=11):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

class IRDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        # Add channel dimension to single-channel TIR images [N, H, W] -> [N, 1, H, W]
        self.tir_lr = torch.from_numpy(data['tir_lr']).unsqueeze(1).float()
        self.tir_hr = torch.from_numpy(data['tir_hr']).unsqueeze(1).float()
        self.rgb_hr = torch.from_numpy(data['rgb_hr']).permute(0, 3, 1, 2).float() # [N, H, W, C] -> [N, C, H, W]

    def __len__(self):
        return len(self.tir_lr)

    def __getitem__(self, idx):
        return self.tir_lr[idx], self.tir_hr[idx], self.rgb_hr[idx]

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
    
    model = SRNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_l1 = nn.L1Loss()
    
    epochs = 20
    best_val_loss = float('inf')
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_ssim = 0.0
        train_deg = 0.0
        
        for lr_in, hr_tgt, _ in train_loader:
            lr_in = lr_in.to(device)
            hr_tgt = hr_tgt.to(device)
            
            # Forward pass
            sr_out = model(lr_in)
            
            # Loss computation
            loss_l1 = criterion_l1(sr_out, hr_tgt)
            loss_ssim = 1.0 - ssim_loss(sr_out, hr_tgt)
            
            # Physics-Informed Degradation-Consistency Loss
            # Downsample SR prediction back to LR input and compare
            sr_downsampled = F.interpolate(sr_out, scale_factor=0.5, mode='bilinear', align_corners=False)
            loss_deg = criterion_l1(sr_downsampled, lr_in)
            
            # Combined Loss
            loss = loss_l1 + 0.1 * loss_ssim + 1.0 * loss_deg
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * lr_in.size(0)
            train_l1 += loss_l1.item() * lr_in.size(0)
            train_ssim += (1.0 - loss_ssim.item()) * lr_in.size(0)
            train_deg += loss_deg.item() * lr_in.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_l1 /= len(train_loader.dataset)
        train_ssim /= len(train_loader.dataset)
        train_deg /= len(train_loader.dataset)
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for lr_in, hr_tgt, _ in val_loader:
                lr_in = lr_in.to(device)
                hr_tgt = hr_tgt.to(device)
                
                sr_out = model(lr_in)
                loss_l1 = criterion_l1(sr_out, hr_tgt)
                loss_ssim = 1.0 - ssim_loss(sr_out, hr_tgt)
                
                loss = loss_l1 + 0.1 * loss_ssim
                val_loss += loss.item() * lr_in.size(0)
                val_l1 += loss_l1.item() * lr_in.size(0)
                val_ssim += (1.0 - loss_ssim.item()) * lr_in.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_l1 /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, SSIM: {train_ssim:.4f}, Deg: {train_deg:.4f}) | Val Loss: {val_loss:.4f} (L1: {val_l1:.4f}, SSIM: {val_ssim:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/sr_best.pth")
            
    print("SR Training complete. Best model saved to checkpoints/sr_best.pth")

if __name__ == "__main__":
    train()

# models/color_model.py
import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """
    U-Net Generator (TIR -> RGB).
    Takes a 1-channel TIR image (128x128) and translates it to 3-channel RGB (128x128)
    using skip connections.
    """
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()
        
        # --- Encoder Layers ---
        # 128x128 -> 64x64
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64x64 -> 32x32
        self.e2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32x32 -> 16x16
        self.e3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 16x16 -> 8x8
        self.e4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 8x8 -> 4x4 (bottleneck)
        self.e5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Decoder Layers ---
        # 4x4 -> 8x8
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        # 8x8 -> 16x16 (takes concat of d1 and e4 output: 512 + 512 = 1024 channels)
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        # 16x16 -> 32x32 (takes concat of d2 and e3 output: 256 + 256 = 512 channels)
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # 32x32 -> 64x64 (takes concat of d3 and e2 output: 128 + 128 = 256 channels)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 64x64 -> 128x128 (takes concat of d4 and e1 output: 64 + 64 = 128 channels)
        self.d5 = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.e1(x)   # 64x64, 64ch
        x2 = self.e2(x1)  # 32x32, 128ch
        x3 = self.e3(x2)  # 16x16, 256ch
        x4 = self.e4(x3)  # 8x8, 512ch
        x5 = self.e5(x4)  # 4x4, 512ch
        
        # Decoder forward pass with skip connections (concatenations)
        y1 = self.d1(x5)                       # 8x8, 512ch
        y2 = self.d2(torch.cat([y1, x4], 1))   # 16x16, 256ch
        y3 = self.d3(torch.cat([y2, x3], 1))   # 32x32, 128ch
        y4 = self.d4(torch.cat([y3, x2], 1))   # 64x64, 64ch
        y5 = self.d5(torch.cat([y4, x1], 1))   # 128x128, 3ch
        
        return y5

class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Takes concatenated TIR (1 channel) and RGB (3 channels) and classifies patches.
    """
    def __init__(self, in_ch=4, num_filters=64):
        super().__init__()
        self.model = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(in_ch, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 15x15 (stride 1)
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 15x15 -> 14x14 (stride 1)
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, tir, rgb):
        # Concatenate inputs along the channel dimension
        x = torch.cat([tir, rgb], dim=1)
        return self.model(x)

if __name__ == "__main__":
    tir = torch.randn(1, 1, 128, 128)
    rgb = torch.randn(1, 3, 128, 128)
    
    G = UNetGenerator()
    fake_rgb = G(tir)
    
    D = PatchDiscriminator()
    pred = D(tir, fake_rgb)
    
    print("Generator Input Shape:", tir.shape)
    print("Generator Output Shape:", fake_rgb.shape)
    print("Discriminator Prediction Shape:", pred.shape)

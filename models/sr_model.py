# models/sr_model.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Standard residual block with batch normalization."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class SRNet(nn.Module):
    """
    Single-channel thermal infrared super-resolution network.
    Upscales input by 2x (e.g. 64x64 -> 128x128).
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=8):
        super().__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Deep residual feature extractor
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        
        # Post-residual conv
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        # 2x PixelShuffle upsampler
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # Output reconstruction
        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()  # Normalizes output to [0, 1] range

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        feat2 = self.res_blocks(feat1)
        feat2 = self.bn2(self.conv2(feat2))
        
        # Skip connection around the residual body
        feat = feat1 + feat2
        
        # Upscale and reconstruct
        out = self.upsample(feat)
        out = self.sigmoid(self.conv_out(out))
        return out

if __name__ == "__main__":
    # Test shape
    x = torch.randn(1, 1, 64, 64)
    model = SRNet()
    y = model(x)
    print("SR Model Input Shape:", x.shape)
    print("SR Model Output Shape:", y.shape)

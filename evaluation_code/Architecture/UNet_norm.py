import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
        channel_weights = self.sigmoid(self.conv1(avg_pool))
        x_att = x * channel_weights
        return x_att


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm2d(in_channels)  # Batch Normalization

    def forward(self, x):
        x = self.batch_norm(x)  # Apply Batch Normalization
        spatial_weights = self.sigmoid(self.conv1(x))
        x_att = x * spatial_weights
        return x_att
    
class OutputNormalization(nn.Module):
    def forward(self, x):
        min_val = torch.min(x)
        range_val = torch.max(x) - min_val
        return (x - min_val) / (range_val + 1e-8)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.spacial_enc1 = SpatialAttention(64)
        self.spectral_enc1 = SpectralAttention(64)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.spacial_enc2 = SpatialAttention(128)
        self.spectral_enc2 = SpectralAttention(128)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.spacial_enc3 = SpatialAttention(256)
        self.spectral_enc3 = SpectralAttention(256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.spacial_bot = SpatialAttention(256)
        self.spectral_bot = SpectralAttention(256)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Upsampling
        )
        
        self.spacial_dec1 = SpatialAttention(128)
        self.spectral_dec1 = SpectralAttention(128)
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SpectralAttention(128),  # Spectral attention mechanism
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Upsampling
        )
        
        self.spacial_dec2 = SpatialAttention(64)
        self.spectral_dec2 = SpectralAttention(64)
        
        self.output_norm = OutputNormalization()
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
#         print(enc1.shape)
        enc2 = self.enc2(enc1)
#         print(enc2.shape)
        enc3 = self.enc3(enc2)
#         print(enc3.shape)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)
#         print(bottleneck.shape)

        # Decoder
        dec1 = self.dec1(torch.cat([enc3, bottleneck], dim=1))
#         print(dec1.shape)
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
#         print(dec2.shape)
        enc1 = self.spacial_enc1(enc1)
    
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
#         print(dec3.shape)
        dec3 = self.output_norm(dec3)
    
        return dec3
    
    def adjust_channels(self, x, target_channels):
        if x.size(1) == target_channels:
            return x
        else:
            # Move the convolutional layer to the same device as the input tensor
            conv_layer = nn.Conv2d(x.size(1), target_channels, kernel_size=1).to(x.device)
            return conv_layer(x)

# # Hyperparameters
# learning_rate = 0.001

# # Initialize the U-Net model with skip connections, loss function, and optimizer
# model= UNet(in_channels=3, out_channels=31).to(device).float()
# model = model.float()
# # criterion = nn.L1Loss()  # You can use a suitable loss function for your task
# criterion =SAMLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


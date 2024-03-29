

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

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

class FFTInteraction(nn.Module):
    def __init__(self):
        super(FFTInteraction, self).__init__()

    def forward(self, x):
        fft_x = torch.fft.fft2(x)
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        return amplitude, phase
    
class IFFTInteraction(nn.Module):
    def __init__(self):
        super(IFFTInteraction, self).__init__()

    def forward(self, amplitude, phase):
        # Combine amplitude and phase to get complex representation
        complex_representation = amplitude * torch.exp(1j * phase)

        # Perform inverse FFT
        ifft_result = torch.fft.ifft2(complex_representation)

        # Take the real part
        ifft_real = ifft_result.real

        return ifft_real
    
class OutputNormalization(nn.Module):
    def forward(self, x):
        min_val = torch.min(x)
        range_val = torch.max(x) - min_val
        return (x - min_val) / (range_val + 1e-8)


class HyperAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HyperAttention, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.en2_spatial_att = SpatialAttention(128)

        # Additional Residual Blocks for increased complexity
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        
        self.fft_interaction = FFTInteraction()
        self.ifft_interaction = IFFTInteraction()
        
        self.output_norm = OutputNormalization()

        # Decoder with attention
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SpectralAttention(128),  # Spectral attention mechanism
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.downscale_enc1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
#         print(f"Shape after enc1: {enc1.shape}")
        
        enc2 = self.enc2(enc1)
#         print(f"Shape after enc2: {enc2.shape}")
        enc2 = self.en2_spatial_att(enc2)
#         print(f"Shape after spatial attention on enc2: {enc2.shape}")
        
        enc3 = self.enc3(enc2)
#         print(f"Shape after enc3: {enc3.shape}")
        
        enc4 = self.enc4(enc3)
#         print(f"Shape after enc4: {enc4.shape}")
        
        # Use FFTInteraction on the input
        amp, ph_temp = self.fft_interaction(enc3)
#         print(f"Shape after FFTInteraction on enc3: {amp.shape}")
        
        # Use FFTInteraction on the encoded result
        amp_temp, ph = self.fft_interaction(enc4)
#         print(f"Shape after FFTInteraction on enc4: {amp_temp.shape}")
        
        # Use IFFTInteraction to combine amplitude and phase
        ifft_enc = self.ifft_interaction(amp, ph)
#         print(f"Shape after IFFT interaction: {ifft_enc.shape}")
        
        # Combine the IFFT result with the original input
        enc4 = ifft_enc + enc4
#         print(f"Shape after combining IFFT result with enc4: {enc4.shape}")

        enc1_adjusted = self.downscale_enc1(enc1)
#         print(f"#{enc1_adjusted.shape}")
        enc1_adjusted = self.adjust_channels(enc1_adjusted, enc4.size(1))
#         print(f"Shape after adjusting channels in enc1_adjusted: {enc1_adjusted.shape}")
        
        combined_features = enc4 + enc1_adjusted
#         print(f"Shape after combining enc1_adjusted with enc4: {combined_features.shape}")

        # Decoder with attention
        dec1 = self.dec1(combined_features)
#         print(f"Shape after dec1: {dec1.shape}")
        
        dec1 = self.output_norm(dec1)
#         print(f"Shape after output_norm: {dec1.shape}")
        
        return dec1

    def adjust_channels(self, x, target_channels):
        if x.size(1) == target_channels:
            return x
        else:
            # Move the convolutional layer to the same device as the input tensor
            conv_layer = nn.Conv2d(x.size(1), target_channels, kernel_size=1).to(
                x.device
            )
            return conv_layer(x)

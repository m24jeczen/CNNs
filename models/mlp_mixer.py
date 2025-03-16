import torch
import torch.nn as nn

class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_channels=3, dim=256, depth=6, num_classes=10):
        """
        MLP-Mixer implementation for CINIC-10.

        Args:
        - image_size: Size of input image (default: 32x32 for CINIC-10).
        - patch_size: Size of patches the image is divided into (default: 4x4).
        - num_channels: Number of image channels (3 for RGB).
        - dim: Hidden dimension size for MLP layers.
        - depth: Number of Mixer blocks.
        - num_classes: Output class count (10 for CINIC-10).
        """
        super(MLPMixer, self).__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.num_patches = (image_size // patch_size) ** 2  # Total patches (32/4)² = 64
        self.patch_dim = num_channels * (patch_size ** 2)  # Each patch flattened (3×4×4=48)
        
        # Linear embedding for patches
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # Mixer Blocks
        self.mixer_layers = nn.Sequential(*[
            MixerBlock(dim, self.num_patches) for _ in range(depth)
        ])

        # Classifier head
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape input into patches
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)  # Create (patch_size x patch_size) patches
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # Reorder dimensions
        x = x.view(batch_size, self.num_patches, -1)  # Flatten patches (batch, num_patches, patch_dim)

        # Project patches into the hidden dimension
        x = self.patch_embedding(x)  # (batch, num_patches, dim)

        # Mixer blocks
        x = self.mixer_layers(x)

        # Global averaging pooling
        x = x.mean(dim=1)

        # Classification head
        return self.fc(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches):
        """
        A single Mixer block with:
        - Token Mixing MLP
        - Channel Mixing MLP
        """
        super(MixerBlock, self).__init__()

        # Token Mixing (Mixes patches across the image)
        self.token_norm = nn.LayerNorm(dim)  # Normalize across feature dimension
        self.token_mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, num_patches)
        )

        # Channel Mixing (Mixes feature channels)
        self.channel_norm = nn.LayerNorm(dim)  # Normalize across feature dimension
        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # Token mixing (Transpose required for correct mixing)
        x = x + self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)

        # Channel mixing (LayerNorm is correctly applied here)
        x = x + self.channel_mixer(self.channel_norm(x))

        return x


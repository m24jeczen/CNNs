import torch
import torch.nn as nn

class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_channels=3, dim=256, depth=6, num_classes=10):

        super(MLPMixer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.num_patches = (image_size // patch_size) ** 2  
        self.patch_dim = num_channels * (patch_size ** 2)  
        
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        self.mixer_layers = nn.Sequential(*[
            MixerBlock(dim, self.num_patches) for _ in range(depth)
        ])

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.unfold(2, 4, 4).unfold(3, 4, 4) 
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  
        x = x.view(batch_size, self.num_patches, -1) 

        x = self.patch_embedding(x) 
        x = self.mixer_layers(x)
        x = x.mean(dim=1)

        return self.fc(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches):
        super(MixerBlock, self).__init__()

        self.token_norm = nn.LayerNorm(dim) 
        self.token_mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, num_patches)
        )

        self.channel_norm = nn.LayerNorm(dim) 
        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x = x + self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixer(self.channel_norm(x))

        return x


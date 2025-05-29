import torch.nn as nn

class Custom3dAE(nn.Module):
    def __init__(self, latent_dim=128, h=150, w=150, d=150):
        super().__init__()
        self.h, self.w, self.d = h, w, d
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1),  # → 32×75×75×75
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), # → 64×38×38×38
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(h//4)*(w//4)*(d//4), latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*(h//4)*(w//4)*(d//4)),
            nn.Unflatten(1, (64, h//4, w//4, d//4)),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

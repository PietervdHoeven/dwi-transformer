import torch.nn as nn
from resnet import generate_model

class ResNetAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # load pretrained encoder
        self.encoder = generate_model(18, n_input_channels=1, n_classes=1000)
        state = torch.load('pretrained_models/resnet_18_23dataset.pth')
        self.encoder.load_state_dict(state['state_dict'])
        # drop classification head
        in_feats = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        # bottleneck
        self.project = nn.Linear(in_feats, latent_dim)
        # decoder (you’ll need to adjust spatial dims here)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_feats),
            # … then ConvTranspose3d layers …
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.encoder(x)
        z = self.project(feat)
        recon = self.decoder(z)
        return recon, z

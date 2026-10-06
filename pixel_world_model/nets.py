"""Variational visual representation and residual latent dynamics."""
import torch
from torch import nn
from torch.nn import functional as F

FRAME_STACK = 4
FRAME_GAP = 4
IMG_SIZE = 128


def reconstruction_loss(logits, target):
    """Stable BCE on raw logits, with pixel MSE retained as a diagnostic."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    mse = F.mse_loss(logits.sigmoid(), target)
    return bce, {'reconstruction': bce.item(), 'pixel_mse': mse.item()}


class VisualVAE(nn.Module):
    def __init__(self, latent_dim=256, img_size=128):
        super().__init__()
        self.latent_dim, self.img_size = latent_dim, img_size
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Conv2d(128, 128, 4, 2, 1), nn.GELU())
        self.side = img_size // 16
        size = 128 * self.side ** 2
        self.mu = nn.Linear(size, latent_dim)
        self.logvar = nn.Linear(size, latent_dim)
        self.project = nn.Linear(latent_dim, size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 4, 4, 2, 1))

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-20, 10)

    def decode_logits(self, z):
        out = self.decoder(self.project(z).view(-1, 128, self.side, self.side))
        if out.shape[-1] != self.img_size:
            out = F.interpolate(out, (self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return out

    def decode(self, z):
        return self.decode_logits(z).sigmoid()

    def forward_logits(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode_logits(z), mu, logvar

    def forward(self, x, sample=True):
        logits, mu, logvar = self.forward_logits(x, sample)
        return logits.sigmoid(), mu, logvar

    def loss(self, x, beta=0.0001, sample=True):
        logits, mu, logvar = self.forward_logits(x, sample)
        bce, metrics = reconstruction_loss(logits, x)
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
        return bce + beta * kl, {**metrics, 'kl': kl.item()}


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x)


class ActionLatentDynamics(nn.Module):
    def __init__(self, latent_dim=256, hidden_width=256, blocks=3):
        super().__init__()
        self.input = nn.Linear(latent_dim + 10, hidden_width)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden_width) for _ in range(blocks)))
        self.delta = nn.Linear(hidden_width, latent_dim)

    def forward(self, z, action):
        a = F.one_hot(action.long(), 10).to(z.dtype)
        return z + self.delta(self.blocks(self.input(torch.cat((z, a), -1))))

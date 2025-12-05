"""Variational Autoencoder for molecular fingerprint representations."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MoleculeVAE(nn.Module):
    """VAE that encodes binary/float fingerprints and reconstructs them.

    The model is intentionally lightweight so it can be reused by future VAE/GAN/RL
    workflows. Encodes high-dimensional fingerprints through an MLP, samples a latent
    vector using the reparameterization trick, and decodes back to the original space.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)

        encoder_layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        last_dim = self.hidden_dims[-1]
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        decoder_dims = list(self.hidden_dims[::-1]) + [input_dim]
        decoder_layers: List[nn.Module] = []
        in_dim = latent_dim
        for hidden_dim in decoder_dims[:-1]:
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, decoder_dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode fingerprint batch into mean and log-variance tensors."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent vector using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples back into fingerprint logits."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run full VAE forward pass returning reconstruction logits, mean, and logvar."""
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input shape (batch, {self.input_dim}) but got {tuple(x.shape)}")
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    recon_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss (reconstruction + KL divergence)."""
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_weight * recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    input_dim = 2048
    latent_dim = 64

    fingerprints = torch.rand(batch_size, input_dim, device=device)
    model = MoleculeVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    recon, mu, logvar = model(fingerprints)
    loss, recon_loss, kl_loss = vae_loss(recon, fingerprints, mu, logvar)
    print(
        f"Sanity check -> total loss: {loss.item():.2f}, "
        f"recon: {recon_loss.item():.2f}, kl: {kl_loss.item():.2f}"
    )

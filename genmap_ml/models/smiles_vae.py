"""Sequence-based SMILES Variational Autoencoder."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SmilesVAE(nn.Module):
    """Character-level VAE that encodes/decodes SMILES sequences."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode token batch into latent statistics."""
        embedded = self.embedding(x)
        _, h_n = self.encoder_rnn(embedded)
        h_enc = h_n[-1]
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vector using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, x_in: torch.LongTensor) -> torch.Tensor:
        """Decode a latent vector using teacher forcing inputs."""
        hidden = torch.tanh(self.fc_z_to_hidden(z)).unsqueeze(0)
        embedded = self.embedding(x_in)
        outputs, _ = self.decoder_rnn(embedded, hidden)
        logits = self.output_projection(outputs)
        return logits

    def forward(self, x: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run VAE forward pass to obtain logits and latent parameters."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_in = x[:, :-1]
        logits = self.decode(z, x_in)
        return logits, mu, logvar


def smiles_vae_loss(
    logits: torch.Tensor,
    target: torch.LongTensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pad_idx: int,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reconstruction and KL losses for the SMILES-VAE."""
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(batch * seq_len, vocab_size)
    target_flat = target.reshape(-1)
    recon_loss = F.cross_entropy(logits_flat, target_flat, ignore_index=pad_idx)

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

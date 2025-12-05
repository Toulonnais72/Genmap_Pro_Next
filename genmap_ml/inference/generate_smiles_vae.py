"""Inference helpers for sampling new SMILES from a trained SMILES-VAE."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn import functional as F

from genmap_ml.datasets.smiles_tokenizer import (
    SmilesTokenizerConfig,
    decode_indices,
    load_tokenizer_config,
)
from genmap_ml.models.smiles_vae import SmilesVAE


def _resolve_device(dev: str) -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(dev)


def load_smiles_vae_model_and_tokenizer(
    model_path: str | Path,
    tokenizer_path: str | Path,
    device: str = "auto",
) -> Tuple[SmilesVAE, SmilesTokenizerConfig, torch.device]:
    """Load a trained SMILES-VAE and its tokenizer configuration."""
    device_obj = _resolve_device(device)
    tokenizer_cfg = load_tokenizer_config(tokenizer_path)

    load_kwargs = {"map_location": device_obj}
    try:
        checkpoint = torch.load(model_path, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(model_path, **load_kwargs)

    state_dict = checkpoint.get("model_state", checkpoint)
    saved_args = checkpoint.get("config", {})
    model = SmilesVAE(
        vocab_size=len(tokenizer_cfg.vocab),
        embed_dim=int(saved_args.get("embed_dim", 256)),
        hidden_dim=int(saved_args.get("hidden_dim", 512)),
        latent_dim=int(saved_args.get("latent_dim", 64)),
        pad_idx=tokenizer_cfg.pad_idx,
    )
    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()
    return model, tokenizer_cfg, device_obj


def sample_latent(n_samples: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Sample latent vectors from the prior distribution."""
    return torch.randn(n_samples, latent_dim, device=device)


@torch.no_grad()
def generate_smiles_from_latent(
    model: SmilesVAE,
    cfg: SmilesTokenizerConfig,
    z: torch.Tensor,
    max_length: int | None = None,
    temperature: float = 1.0,
) -> List[str]:
    """Decode latent vectors into SMILES strings using autoregressive sampling."""
    max_len = max_length or cfg.max_length
    batch = z.size(0)
    device = z.device

    hidden = torch.tanh(model.fc_z_to_hidden(z)).unsqueeze(0)
    input_token = torch.full((batch, 1), cfg.bos_idx, dtype=torch.long, device=device)
    generated_tokens: List[List[int]] = [[] for _ in range(batch)]
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_len):
        embedded = model.embedding(input_token)
        output, hidden = model.decoder_rnn(embedded, hidden)
        logits = model.output_projection(output.squeeze(1))
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        next_token = torch.where(
            finished, torch.full_like(next_token, cfg.pad_idx), next_token
        )
        for i in range(batch):
            generated_tokens[i].append(int(next_token[i].item()))
        finished = finished | (next_token == cfg.eos_idx)
        if torch.all(finished):
            break
        input_token = next_token.unsqueeze(1)

    smiles_outputs = []
    for tokens in generated_tokens:
        sequence = decode_indices(tokens, cfg, skip_special=True)
        smiles_outputs.append(sequence)
    return smiles_outputs


def generate_smiles_with_trained_vae(
    model_dir: str | Path,
    n_samples: int = 100,
    temperature: float = 1.0,
    max_length: int | None = None,
    device: str = "auto",
) -> List[str]:
    """Load model artifacts and generate SMILES strings."""
    model_dir = Path(model_dir)
    model_path = model_dir / "smiles_vae.pt"
    tokenizer_path = model_dir / "tokenizer.json"
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("Model directory must contain smiles_vae.pt and tokenizer.json")

    model, cfg, device_obj = load_smiles_vae_model_and_tokenizer(
        model_path=model_path, tokenizer_path=tokenizer_path, device=device
    )
    z = sample_latent(n_samples, model.latent_dim, device_obj)
    smiles = generate_smiles_from_latent(
        model=model,
        cfg=cfg,
        z=z,
        max_length=max_length,
        temperature=temperature,
    )
    filtered = [s for s in smiles if s]
    return filtered


if __name__ == "__main__":
    demo_dir = Path("models/smiles_vae_example")
    try:
        outputs = generate_smiles_with_trained_vae(demo_dir, n_samples=5, temperature=0.8)
        for idx, smi in enumerate(outputs, 1):
            print(f"{idx:02d}: {smi}")
    except FileNotFoundError:
        print("SMILES-VAE artifacts not found; please train the model first.")

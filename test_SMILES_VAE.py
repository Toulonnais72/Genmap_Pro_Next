#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Petit script de test pour le SMILES-VAE :
- charge le tokenizer et le modèle entraîné
- échantillonne des vecteurs latents z ~ N(0, I)
- génère des SMILES par décodage auto-régressif (greedy)
"""

from pathlib import Path
import torch
import torch.nn.functional as F

from genmap_ml.models.smiles_vae import SmilesVAE
from genmap_ml.datasets.smiles_tokenizer import (
    load_tokenizer_config,
    decode_indices,
)


# ========= PARAMÈTRES À ADAPTER SI BESOIN =========
MODEL_DIR = Path("models/smiles_vae_druggable")  # dossier du run
MODEL_PATH = MODEL_DIR / "smiles_vae.pt"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

N_SAMPLES = 20          # nombre de SMILES à générer pour le test
TEMPERATURE = 1.0       # 1.0 = greedy (argmax) si on utilise argmax
MAX_GEN_LENGTH = None   # None => use cfg.max_length
# ==================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("✅ Utilisation du GPU (cuda)")
        return torch.device("cuda")
    else:
        print("⚠️ Utilisation du CPU")
        return torch.device("cpu")


def load_model_and_tokenizer(
    model_path: Path,
    tokenizer_path: Path,
    device: torch.device,
) -> tuple[SmilesVAE, object]:
    """
    - Charge le tokenizer
    - Charge le checkpoint PyTorch (model_state + config)
    - Reconstruit le modèle avec les bons hyperparamètres
    """
    # 1) Tokenizer
    cfg = load_tokenizer_config(tokenizer_path)
    print(f"Tokenizer chargé depuis {tokenizer_path}")
    print(f"  - vocab_size = {len(cfg.vocab)}")
    print(f"  - max_length = {cfg.max_length}")

    # 2) Checkpoint modèle
    print(f"Chargement du checkpoint : {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print("Clés du checkpoint :", checkpoint.keys())  # devrait afficher ['model_state', 'config']

    state_dict = checkpoint["model_state"]
    config = checkpoint.get("config", {})

    latent_dim = config.get("latent_dim", 64)
    embed_dim = config.get("embed_dim", 256)
    hidden_dim = config.get("hidden_dim", 512)

    print("Hyperparamètres récupérés / utilisés :")
    print(f"  - latent_dim = {latent_dim}")
    print(f"  - embed_dim  = {embed_dim}")
    print(f"  - hidden_dim = {hidden_dim}")

    # 3) Reconstruction du modèle
    model = SmilesVAE(
        vocab_size=len(cfg.vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        pad_idx=cfg.pad_idx,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("✅ Modèle SMILES-VAE chargé et reconstruit")

    return model, cfg


@torch.no_grad()
def generate_smiles_from_latent(
    model: SmilesVAE,
    cfg,
    z: torch.Tensor,
    max_length: int | None = None,
    temperature: float = 1.0,
) -> list[str]:
    """
    Génération auto-régressive à partir de z.

    - z : [batch, latent_dim]
    - On initialise l'état caché du décodeur à partir de z
    - On commence avec le token <bos>
    - À chaque étape :
        - on passe par decoder_rnn + projection
        - on prend argmax (greedy) ou sampling avec temperature
        - on arrête quand on rencontre <eos> ou qu'on atteint max_length
    """
    device = z.device
    batch_size = z.shape[0]

    if max_length is None:
        max_length = cfg.max_length

    # Initial hidden state du décodeur
    h0 = model.fc_z_to_hidden(z)          # [batch, hidden_dim]
    h0 = torch.tanh(h0)                   # petit non-linéaire
    h0 = h0.unsqueeze(0)                  # [1, batch, hidden_dim]

    # Token <bos> pour démarrer
    cur_tokens = torch.full(
        (batch_size, 1),
        fill_value=cfg.bos_idx,
        dtype=torch.long,
        device=device,
    )

    generated_indices = []

    # On garde la liste des séquences générées (indices)
    seqs = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    for t in range(max_length):
        # embed du token courant
        emb = model.embedding(cur_tokens)  # [batch, 1, embed_dim]

        # passer dans le RNN
        out, h0 = model.decoder_rnn(emb, h0)  # out: [batch, 1, hidden_dim]

        # projection vocab
        logits = model.output_projection(out[:, -1, :])  # [batch, vocab_size]

        if temperature <= 0.0:
            # greedy pur (argmax)
            next_tokens = torch.argmax(logits, dim=-1)  # [batch]
        else:
            # sampling avec température
            probs = F.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]

        # on stocke les indices
        for i in range(batch_size):
            if not finished[i]:
                tok = next_tokens[i].item()
                if tok == cfg.eos_idx:
                    finished[i] = True
                else:
                    seqs[i].append(tok)

        # si toutes les séquences sont finies, on peut sortir
        if all(finished):
            break

        # préparer le token suivant comme entrée
        cur_tokens = next_tokens.unsqueeze(1)  # [batch, 1]

    # décodage indices -> SMILES
    smiles_list = []
    for idx_seq in seqs:
        s = decode_indices(idx_seq, cfg, skip_special=True)
        smiles_list.append(s)

    return smiles_list


def main():
    device = get_device()

    # Charger modèle + tokenizer
    model, tok_cfg = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH, device)

    # Échantillonner des z ~ N(0, I)
    latent_dim = model.fc_mu.out_features  # taille du latent
    print(f"Latent dim détectée depuis le modèle : {latent_dim}")

    z = torch.randn(N_SAMPLES, latent_dim, device=device)

    # Génération
    print(f"Génération de {N_SAMPLES} SMILES...")
    smiles_gen = generate_smiles_from_latent(
        model=model,
        cfg=tok_cfg,
        z=z,
        max_length=MAX_GEN_LENGTH,
        temperature=TEMPERATURE,
    )

    # Affichage
    print("\n=== SMILES générés ===")
    for i, s in enumerate(smiles_gen, start=1):
        print(f"{i:3d}: {s}")


if __name__ == "__main__":
    main()

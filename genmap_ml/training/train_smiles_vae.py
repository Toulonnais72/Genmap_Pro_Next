"""Training script for the character-level SMILES VAE (with optional FDA bias)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from genmap_ml.datasets.smiles_tokenizer import (
    SmilesTokenizerConfig,
    build_tokenizer_from_smiles,
    encode_smiles,
    save_tokenizer_config,
)
from genmap_ml.models.smiles_vae import SmilesVAE, smiles_vae_loss


class SmilesDataset(Dataset):
    """Wraps pre-encoded SMILES tensors for convenient batching."""

    def __init__(self, sequences: torch.Tensor) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SMILES-based VAE.")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to main SMILES CSV."
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="SMILES column name in the main dataset.",
    )

    # Legacy: optional extra dataset (kept for backward compatibility)
    parser.add_argument(
        "--extra-dataset",
        type=Path,
        default=None,
        help="Optional extra SMILES CSV (e.g. FDA-approved drugs).",
    )
    parser.add_argument(
        "--extra-smiles-col",
        type=str,
        default="smiles",
        help="SMILES column name in the extra dataset.",
    )

    # New: FDA-biased training configuration
    parser.add_argument(
        "--fda-dataset",
        type=str,
        default=None,
        help=(
            "Chemin vers FDA_Approved_structures.csv. Si fourni, on mélange ce dataset "
            "aux SMILES globaux et on sur-échantillonne les molécules FDA."
        ),
    )
    parser.add_argument(
        "--fda-smiles-col",
        type=str,
        default="smiles",
        help=(
            "Nom de la colonne contenant les SMILES dans le fichier "
            "FDA_Approved_structures.csv."
        ),
    )
    parser.add_argument(
        "--fda-sampling-weight",
        type=float,
        default=5.0,
        help=(
            "Poids relatif de sur-échantillonnage pour les molécules FDA "
            "(WeightedRandomSampler). 1.0 = pas de biais."
        ),
    )

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Alias for --lr; when set, overrides --lr.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="auto", choices=("auto", "cpu", "cuda")
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional max number of SMILES from the main dataset. "
            "The extra dataset is always fully included."
        ),
    )
    parser.add_argument(
        "--max-train-smiles",
        type=int,
        default=None,
        help="Optional max number of SMILES from the main dataset.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of GRU layers in encoder/decoder (metadata only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU available.")
    return torch.device(arg)


def build_run_directory(base_dir: Path, run_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_smiles_list(
    dataset_path: Path, smiles_col: str, max_samples: Optional[int]
) -> List[str]:
    """Load, clean and optionally subsample SMILES from a CSV."""
    df = pd.read_csv(dataset_path)
    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in dataset '{dataset_path}'. "
            f"Available columns: {list(df.columns)}"
        )

    smiles = (
        df[smiles_col]
        .dropna()
        .map(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if max_samples is not None and len(smiles) > max_samples:
        smiles = random.sample(smiles, max_samples)

    if not smiles:
        raise ValueError(
            f"No valid SMILES strings found in '{dataset_path}' after preprocessing."
        )
    return smiles


def encode_dataset(smiles: List[str], cfg: SmilesTokenizerConfig) -> torch.Tensor:
    encoded = [encode_smiles(s, cfg) for s in smiles]
    return torch.tensor(encoded, dtype=torch.long)


def load_smiles_from_csv(path: str, smiles_col: str) -> List[str]:
    df = pd.read_csv(path)
    smiles = df[smiles_col].dropna().astype(str).tolist()
    return smiles


def main() -> None:
    args = parse_args()

    # Harmonise aliases for compatibility and metadata
    if getattr(args, "learning_rate", None) is None:
        args.learning_rate = args.lr
    else:
        args.lr = args.learning_rate
    if getattr(args, "max_train_smiles", None) is None:
        args.max_train_smiles = args.max_samples

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Training SmilesVAE on device: {device}")

    # -------------------------------------------------------------------------
    # Chargement du dataset principal (global)
    # -------------------------------------------------------------------------
    print(f"Loading training SMILES from: {args.dataset}")
    df_main = pd.read_csv(args.dataset)

    smiles_col = args.smiles_col if args.smiles_col in df_main.columns else "smiles"
    if smiles_col not in df_main.columns:
        raise ValueError(
            f"Training dataset must contain a 'smiles' column (or '{args.smiles_col}'). "
            f"Found columns: {list(df_main.columns)}"
        )

    base_smiles = df_main[smiles_col].dropna().astype(str).tolist()

    # Optionnel : limiter le nombre de SMILES globaux (main dataset uniquement)
    if args.max_train_smiles is not None and args.max_train_smiles > 0:
        base_smiles = base_smiles[: args.max_train_smiles]

    # Legacy: extra dataset is treated as additional non-FDA SMILES
    if args.extra_dataset is not None:
        print(f"Loading extra SMILES from: {args.extra_dataset}")
        extra_smiles = prepare_smiles_list(
            args.extra_dataset, args.extra_smiles_col, max_samples=None
        )
        print(
            f"Extra dataset: {len(extra_smiles)} SMILES after cleaning "
            f"(fully included, no subsampling)."
        )
        base_smiles = base_smiles + extra_smiles

    print(f"Global SMILES (base) : {len(base_smiles)}")

    # -------------------------------------------------------------------------
    # Chargement du dataset FDA si fourni
    # -------------------------------------------------------------------------
    fda_smiles: List[str] = []
    if args.fda_dataset is not None:
        print(f"Loading FDA SMILES from: {args.fda_dataset}")
        df_fda = pd.read_csv(args.fda_dataset)

        if args.fda_smiles_col not in df_fda.columns:
            raise ValueError(
                f"FDA dataset must contain column '{args.fda_smiles_col}', "
                f"found columns: {list(df_fda.columns)}"
            )

        fda_smiles = df_fda[args.fda_smiles_col].dropna().astype(str).tolist()
        print(f"FDA SMILES loaded: {len(fda_smiles)}")
    else:
        print("No FDA dataset provided, training remains agnostic.")

    # -------------------------------------------------------------------------
    # Construction de la liste combinée + flags FDA
    # -------------------------------------------------------------------------
    all_smiles = base_smiles + fda_smiles
    is_fda_flags = [0] * len(base_smiles) + [1] * len(fda_smiles)

    print(f"Total SMILES (global + FDA): {len(all_smiles)}")
    print(
        f"Breakdown: base={len(base_smiles)}, FDA={len(fda_smiles)}, "
        f"FDA ratio={len(fda_smiles) / max(1, len(all_smiles)):.3f}"
    )

    # 4) Construction du tokenizer sur l'ensemble combiné
    tokenizer_cfg = build_tokenizer_from_smiles(
        smiles_list=all_smiles,
        max_length=args.max_length,
    )

    run_dir = build_run_directory(args.output_dir, args.run_name)
    tokenizer_path = run_dir / "tokenizer.json"
    save_tokenizer_config(tokenizer_cfg, tokenizer_path)

    # 5) Encodage du dataset complet
    sequences_tensor = encode_dataset(all_smiles, tokenizer_cfg)
    dataset = SmilesDataset(sequences_tensor)

    # ---------------------------------------------------------------------
    # WeightedRandomSampler pour sur-échantillonner les molécules FDA
    # ---------------------------------------------------------------------
    if len(is_fda_flags) != len(dataset):
        raise RuntimeError(
            f"is_fda_flags length ({len(is_fda_flags)}) != dataset length ({len(dataset)})"
        )

    weights: List[float] = []
    for flag in is_fda_flags:
        if flag == 1:
            weights.append(float(args.fda_sampling_weight))
        else:
            weights.append(1.0)

    weights_tensor = torch.as_tensor(weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),  # un epoch ~ taille du dataset
        replacement=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,  # sampler gère l'ordre
        num_workers=args.num_workers,
        drop_last=True,
    )

    # 6) Modèle VAE
    model = SmilesVAE(
        vocab_size=len(tokenizer_cfg.vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        pad_idx=tokenizer_cfg.pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch in loader:
            x = batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(x)
            # Décalage de la cible (teacher forcing standard)
            target = x[:, 1:]
            loss, recon_loss, kl_loss = smiles_vae_loss(
                logits,
                target,
                mu,
                logvar,
                pad_idx=tokenizer_cfg.pad_idx,
                beta=args.beta,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "recon_loss": avg_recon,
                "kl_loss": avg_kl,
            }
        )
        print(
            f"[Epoch {epoch:03d}] loss={avg_loss:.4f} "
            f"recon={avg_recon:.4f} kl={avg_kl:.4f}"
        )

    # 7) Sauvegarde modèle + métadonnées
    model_path = run_dir / "smiles_vae.pt"
    torch.save({"model_state": model.state_dict(), "config": vars(args)}, model_path)

    metadata: Dict[str, Any] = {
        "run_name": args.run_name,
        "output_dir": str(args.output_dir),
        "dataset": str(args.dataset),
        "fda_dataset": str(args.fda_dataset) if args.fda_dataset is not None else None,
        "num_smiles_base": len(base_smiles),
        "num_smiles_fda": len(fda_smiles),
        "num_smiles_total": len(all_smiles),
        "fda_sampling_weight": float(args.fda_sampling_weight),
        "vocab_size": int(len(tokenizer_cfg.vocab)),
        "max_length": int(tokenizer_cfg.max_length),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "device": str(device),
        "run_dir": str(run_dir),
    }
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete. Artifacts saved to {run_dir}")


# Exemple de commande pour entraîner un VAE biaisé FDA :
# python -m genmap_ml.training.train_smiles_vae \
#   --dataset ./genmap_ml/datasets/vae_train_dataset.csv \
#   --fda-dataset ./genmap_ml/datasets/FDA_Approved_structures.csv \
#   --fda-smiles-col smiles \
#   --fda-sampling-weight 5.0 \
#   --output-dir ./genmap_ml/models/vae_fda_biased \
#   --run-name vae_fda_biased_v1


if __name__ == "__main__":
    main()


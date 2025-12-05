"""Training entry point for MoleculeVAE on SMILES datasets."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator as rFG
from torch.utils.data import DataLoader, Dataset

from genmap_ml.models.vae import MoleculeVAE, vae_loss


class FingerprintDataset(Dataset):
    """Dataset turning SMILES strings into Morgan fingerprints."""

    def __init__(
        self,
        csv_path: Path,
        fp_size: int,
        radius: int,
        max_samples: Optional[int] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0 when provided")

        df = pd.read_csv(self.csv_path)
        if "smiles" not in df.columns:
            raise ValueError("Dataset must contain a 'smiles' column")

        generator = rFG.GetMorganGenerator(radius=radius, fpSize=fp_size)
        fingerprints: List[np.ndarray] = []
        invalid = 0
        processed = 0
        for smiles in df["smiles"]:
            if max_samples is not None and len(fingerprints) >= max_samples:
                break
            processed += 1
            if not isinstance(smiles, str) or not smiles.strip():
                invalid += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid += 1
                continue
            bitvect = generator.GetFingerprint(mol)
            arr = np.zeros(fp_size, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bitvect, arr)
            fingerprints.append(arr)

        if not fingerprints:
            raise ValueError("No valid SMILES found in dataset")

        stacked = np.stack(fingerprints)
        self.fp_tensor = torch.from_numpy(stacked)
        self.metadata = {
            "csv_path": str(self.csv_path),
            "num_rows": int(df.shape[0]),
            "num_processed": processed,
            "num_loaded": len(fingerprints),
            "num_invalid": invalid,
            "fp_size": fp_size,
            "radius": radius,
        }

    def __len__(self) -> int:
        return self.fp_tensor.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.fp_tensor[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MoleculeVAE on SMILES data.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("genmap_ml/datasets/vae_train_dataset.csv"),
        help="CSV file containing a 'smiles' column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/vae_runs"),
        help="Directory where checkpoints and logs are written.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for the run (defaults to timestamp).",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=(1024, 512, 256),
        help="Hidden layer sizes for the encoder (mirrored for decoder).",
    )
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional subset size for quick experiments.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device (auto picks CUDA when available).",
    )
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    """Convert objects (e.g., Path) to JSON-serializable structures."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(value) for value in obj]
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU detected")
    return torch.device(device_arg)


def build_run_dir(base_dir: Path, run_name: Optional[str]) -> Path:
    base_dir = base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    label = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / label
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    run_dir = build_run_dir(args.output_dir, args.run_name)

    dataset = FingerprintDataset(
        csv_path=args.dataset,
        fp_size=args.fp_size,
        radius=args.radius,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MoleculeVAE(
        input_dim=args.fp_size,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    checkpoint_path = run_dir / "molecule_vae.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        for batch in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(
                recon, batch, mu, logvar, recon_weight=args.recon_weight
            )
            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon / len(dataset)
        avg_kl = total_kl / len(dataset)
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "config": vars(args),
                },
                checkpoint_path,
            )

    metadata: Dict[str, Any] = {
        "args": vars(args),
        "device": str(device),
        "dataset_stats": dataset.metadata,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "best_loss": best_loss,
        "checkpoint": str(checkpoint_path),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(metadata), f, indent=2)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training finished. Artifacts stored in {run_dir}")


if __name__ == "__main__":
    train()

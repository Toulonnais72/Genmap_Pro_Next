"""Utility functions to sample the MoleculeVAE latent space and map back to SMILES.

The decoder currently outputs fingerprint logits/probabilities. We approximate an
inverse projection to SMILES by snapping generated fingerprints to the closest
reference fingerprint (Tanimoto NN lookup). A future release can swap in a
SMILES-native decoder without touching the public API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rFG
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from genmap_ml.models.vae import MoleculeVAE


def load_vae_model(
    model_path: str | Path,
    input_dim: int,
    latent_dim: int,
    device: str | None = None,
) -> Tuple[MoleculeVAE, torch.device]:
    """Instantiate MoleculeVAE, load weights, and return the eval model and device."""
    resolved = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    load_kwargs = {"map_location": resolved}
    try:
        checkpoint = torch.load(model_path, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(model_path, **load_kwargs)
    state_dict = checkpoint.get("model_state", checkpoint)
    hidden_dims: Optional[Sequence[int]] = None
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if config:
            hidden_dims = config.get("hidden_dims")
    model = MoleculeVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(state_dict)
    model.to(resolved)
    model.eval()
    return model, resolved


@torch.no_grad()
def sample_latent(
    model: MoleculeVAE,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
) -> np.ndarray:
    """Sample latent vectors ~ N(0, I) and decode them into fingerprint logits."""
    z = torch.randn(n_samples, latent_dim, device=device)
    logits = model.decode(z)
    fingerprints = torch.sigmoid(logits)
    return fingerprints.cpu().numpy()


def _array_to_bitvect(values: np.ndarray) -> ExplicitBitVect:
    """Convert probability vector to ExplicitBitVect via 0.5 thresholding."""
    bitvect = ExplicitBitVect(len(values))
    on_bits = np.where(values >= 0.5)[0]
    for idx in on_bits:
        bitvect.SetBit(int(idx))
    return bitvect


def postprocess_fps_to_smiles(
    generated_fps: np.ndarray,
    reference_df: pd.DataFrame,
    n_neighbors: int = 3,
) -> List[Dict[str, object]]:
    """Snap generated fingerprints to nearest reference fingerprints via Tanimoto."""
    if "fp" not in reference_df.columns or "smiles" not in reference_df.columns:
        raise ValueError("reference_df must contain 'fp' and 'smiles' columns")
    ref_fps = reference_df["fp"].tolist()
    ref_smiles = reference_df["smiles"].tolist()
    if n_neighbors <= 0:
        n_neighbors = 1

    results: List[Dict[str, object]] = []
    for fp_array in generated_fps:
        gen_fp = _array_to_bitvect(fp_array)
        scores = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
        if not scores:
            continue
        score_array = np.asarray(scores)
        top_indices = score_array.argsort()[::-1][:n_neighbors]
        best_idx = int(top_indices[0])
        results.append(
            {
                "source_smiles": ref_smiles[best_idx],
                "tanimoto_similarity": float(score_array[best_idx]),
                "neighbor_smiles": [ref_smiles[int(i)] for i in top_indices],
                "neighbor_scores": [float(score_array[int(i)]) for i in top_indices],
            }
        )
    return results


def _load_reference_fingerprints(
    reference_data_path: str | Path,
    fp_size: int,
    radius: int,
) -> pd.DataFrame:
    """Load a CSV of SMILES and compute Morgan fingerprints for lookup."""
    df = pd.read_csv(reference_data_path)
    if "smiles" not in df.columns:
        raise ValueError("Reference data must include a 'smiles' column")
    generator = rFG.GetMorganGenerator(radius=radius, fpSize=fp_size)

    smiles_list: List[str] = []
    fps: List[ExplicitBitVect] = []
    for raw_smiles in df["smiles"]:
        if not isinstance(raw_smiles, str) or not raw_smiles.strip():
            continue
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            continue
        smiles_list.append(Chem.MolToSmiles(mol))
        fps.append(generator.GetFingerprint(mol))
    return pd.DataFrame({"smiles": smiles_list, "fp": fps})


def generate_molecules_with_vae(
    model_path: str | Path,
    config_path: str | Path,
    reference_data_path: str | Path,
    n_samples: int,
    min_tanimoto: float = 0.4,
    filter_unique: bool = True,
) -> pd.DataFrame:
    """High-level helper to sample molecules, map to SMILES, and filter them."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    args = config.get("args", {})
    fp_size = int(args.get("fp_size") or config["dataset_stats"]["fp_size"])
    latent_dim = int(args.get("latent_dim"))
    radius = int(args.get("radius") or config["dataset_stats"]["radius"])

    model, device = load_vae_model(
        model_path=model_path,
        input_dim=fp_size,
        latent_dim=latent_dim,
        device=args.get("device"),
    )

    reference_df = _load_reference_fingerprints(reference_data_path, fp_size, radius)
    generated = sample_latent(model, n_samples=n_samples, latent_dim=latent_dim, device=device)
    neighbor_info = postprocess_fps_to_smiles(generated, reference_df)

    rows: List[Dict[str, object]] = []
    seen: set[str] = set()
    for entry in neighbor_info:
        smiles = entry["source_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        similarity = float(entry["tanimoto_similarity"])
        if similarity < min_tanimoto:
            continue
        if filter_unique and canonical in seen:
            continue
        seen.add(canonical)
        rows.append(
            {
                "generated_smiles": canonical,
                "source_smiles": smiles,
                "tanimoto_similarity": similarity,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    demo_dir = Path("models/vae_morgan_druggable")
    model_file = demo_dir / "molecule_vae.pt"
    config_file = demo_dir / "config.json"
    reference_file = Path("genmap_ml/datasets/vae_train_dataset.csv")
    if model_file.exists() and config_file.exists() and reference_file.exists():
        df = generate_molecules_with_vae(
            model_path=model_file,
            config_path=config_file,
            reference_data_path=reference_file,
            n_samples=8,
            min_tanimoto=0.45,
        )
        print(df.head())
    else:
        print("VAE artifacts not found; skipping demo generation.")

# genmap_ml/inference/vae_candidates.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski

from .generate_smiles_vae import generate_smiles_with_trained_vae


def _is_lipinski_like(mol) -> bool:
    """Filtre très simple type Lipinski."""
    mw = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)

    if mw > 500:
        return False
    if logp > 5:
        return False
    if hbd > 5:
        return False
    if hba > 10:
        return False
    if rot > 10:
        return False
    return True


def _mol_properties(mol) -> dict:
    """Retourne un dict de propriétés RDKit basiques."""
    mw = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    rings = Lipinski.RingCount(mol)

    return {
        "mw": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rot_bonds": rot,
        "num_rings": rings,
    }


def generate_vae_candidates_dataframe(
    model_dir: str | Path,
    n_valid_target: int = 500,
    batch_size: int = 512,
    temperature: float = 0.7,
    max_length: int | None = None,
    device: str = "auto",
    source_label: str = "smiles_vae_druggable_v1",
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Génére des SMILES avec le VAE jusqu'à obtenir n_valid_target molécules
    valides (RDKit + filtre Lipinski-like), calcule quelques propriétés
    et renvoie un DataFrame prêt à être utilisé dans Genmap.

    Colonnes retournées :
    - smiles
    - source
    - mw, logp, tpsa, hbd, hba, rot_bonds, num_rings
    """
    model_dir = Path(model_dir)

    # Seeds pour un comportement reproductible
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    collected_rows: List[dict] = []

    while len(collected_rows) < n_valid_target:
        smiles_batch = generate_smiles_with_trained_vae(
            model_dir=model_dir,
            n_samples=batch_size,
            temperature=temperature,
            max_length=max_length,
            device=device,
        )

        for s in smiles_batch:
            if not s:
                continue
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue

            if not _is_lipinski_like(mol):
                continue

            props = _mol_properties(mol)
            row = {"smiles": s, "source": source_label, **props}
            collected_rows.append(row)

            if len(collected_rows) >= n_valid_target:
                break

    df = pd.DataFrame(collected_rows)
    return df

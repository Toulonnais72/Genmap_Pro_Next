#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import random
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from genmap_ml.inference.generate_smiles_vae import generate_smiles_with_trained_vae

MODEL_DIR = Path("models/smiles_vae_druggable")  # adapte si besoin
N_SAMPLES = 1000
RANDOM_SEED = 0

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    smiles_list = generate_smiles_with_trained_vae(
        model_dir=MODEL_DIR,
        n_samples=N_SAMPLES,
        temperature=0.8,
        max_length=None,
        device="auto",
    )

    print(f"Générés : {len(smiles_list)} SMILES")

    valid_mols = []
    props = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        valid_mols.append((s, mol))

        mw = Descriptors.ExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        props.append((mw, logp, tpsa))

    print(f"Valides RDKit : {len(valid_mols)}/{len(smiles_list)} "
          f"({100 * len(valid_mols)/max(1, len(smiles_list)):.1f} %)")

    if not props:
        print("Aucune molécule valide, vérifier la génération.")
        return

    mws, logps, tpsas = zip(*props)
    print(f"MW   : mean={np.mean(mws):.1f}, min={min(mws):.1f}, max={max(mws):.1f}")
    print(f"logP : mean={np.mean(logps):.2f}, min={min(logps):.2f}, max={max(logps):.2f}")
    print(f"TPSA : mean={np.mean(tpsas):.1f}, min={min(tpsas):.1f}, max={max(tpsas):.1f}")

    # diversité rapide : Tanimoto sur un sous-échantillon de 100 molécules
    sub = valid_mols[:100]
    fps = []
    for _, mol in sub:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)

    sims = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    if sims:
        print(f"Tanimoto pairwise (100 échantillons) : "
              f"mean={np.mean(sims):.3f}, min={min(sims):.3f}, max={max(sims):.3f}")

if __name__ == "__main__":
    main()

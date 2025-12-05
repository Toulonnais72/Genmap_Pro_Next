"""ADMET prediction utilities.

This module provides a :func:`predict_admet` function that returns a set of
absorption, distribution, metabolism, excretion and toxicity (ADMET) related
properties for a molecule defined by its SMILES string.  When optional third
party predictors such as `pkCSM` or `admetSAR` are available they are used,
otherwise basic descriptors are computed with `RDKit`.

Fields returned by :func:`predict_admet` when falling back to RDKit
implementation:

``logP``
    Crippen octanol/water partition coefficient.
``tpsa``
    Topological polar surface area in Å².
``hba``
    Number of hydrogen bond acceptors.
``hbd``
    Number of hydrogen bond donors.
``mol_wt``
    Exact molecular weight.
``rotatable_bonds``
    Number of rotatable bonds.

The module also exposes :func:`compute_mpo` which returns a simple
multi‑parameter optimisation (MPO) score based on common properties.
"""

from __future__ import annotations

from typing import Dict

try:  # pragma: no cover - optional dependency
    import pkcsm  # type: ignore
    _BACKEND = "pkCSM"
except Exception:  # pragma: no cover - optional dependency
    try:
        import admetSAR  # type: ignore
        _BACKEND = "admetSAR"
    except Exception:  # pragma: no cover - optional dependency
        _BACKEND = "rdkit"

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

__all__ = ["predict_admet", "compute_mpo"]


def predict_admet(smiles: str) -> Dict[str, float]:
    """Predict ADMET properties for a molecule.

    Parameters
    ----------
    smiles:
        Molecule represented as a SMILES string.

    Returns
    -------
    dict
        Mapping of property name to predicted value.  Keys are described in the
        module level documentation.  When a third party predictor is available
        its result is returned directly.
    """

    if _BACKEND == "pkCSM":  # pragma: no cover - optional
        return pkcsm.predict(smiles)  # type: ignore
    if _BACKEND == "admetSAR":  # pragma: no cover - optional
        return admetSAR.predict(smiles)  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    properties = {
        "logP": Crippen.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "mol_wt": Descriptors.ExactMolWt(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
    }
    return properties


def _triangular_score(value: float, low: float, high: float) -> float:
    """Return a triangular score between 0 and 1.

    The score is 0 at the ``low`` and ``high`` boundaries and 1 in the middle.
    """

    if value <= low or value >= high:
        return 0.0
    mid = (low + high) / 2.0
    if value <= mid:
        return (value - low) / (mid - low)
    return (high - value) / (high - mid)


def compute_mpo(smiles: str) -> float:
    """Compute a simplified multi‑parameter optimisation score.

    The score is the sum of triangular contributions from physicochemical
    properties and ranges between 0 and 4 in this implementation.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.ExactMolWt(mol)
    hbd = Lipinski.NumHDonors(mol)

    scores = [
        _triangular_score(logp, 0, 6),
        _triangular_score(tpsa, 0, 140),
        _triangular_score(mw, 200, 600),
        _triangular_score(hbd, 0, 3),
    ]
    return float(sum(scores))

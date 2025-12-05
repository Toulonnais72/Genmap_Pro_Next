"""Visualization utilities using UMAP, Plotly and RDKit.

This module offers small helper functions to create a UMAP scatter plot with
Plotly and to render molecules with RDKit.  All heavy dependencies are optional
and the functions will raise :class:`ImportError` when they are unavailable.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Optional

# Optional imports ---------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import numpy as np
    import umap  # type: ignore
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    umap = None  # type: ignore
    px = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
    from rdkit.Chem import Draw
except Exception:  # pragma: no cover - optional dependency
    Chem = None  # type: ignore
    Draw = None  # type: ignore

__all__ = ["plot_umap", "mol_to_image"]


def plot_umap(
    features: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Project ``features`` to 2‑D using UMAP and return a Plotly figure.

    Parameters
    ----------
    features:
        Iterable of feature vectors.  Will be converted to a NumPy array.
    labels:
        Optional labels to display with each point.
    n_neighbors:
        Number of neighbors used by UMAP.
    min_dist:
        Minimum distance parameter for UMAP.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive scatter plot of the UMAP projection.
    """

    if np is None or umap is None or px is None:  # pragma: no cover - optional
        raise ImportError("plot_umap requires numpy, umap-learn and plotly")

    embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(
        np.asarray(list(features))
    )
    fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], text=labels)
    fig.update_traces(textposition="top center")
    fig.update_layout(xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    return fig


def mol_to_image(smiles: str, size: Tuple[int, int] = (300, 300)):
    """Return a PIL image of the molecule represented by ``smiles``.

    Parameters
    ----------
    smiles:
        Molecule represented as a SMILES string.
    size:
        Size of the generated image in pixels.
    """

    if Chem is None or Draw is None:  # pragma: no cover - optional
        raise ImportError("mol_to_image requires RDKit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    return Draw.MolToImage(mol, size=size)

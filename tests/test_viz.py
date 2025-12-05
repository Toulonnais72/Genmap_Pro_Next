import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("numpy")
pytest.importorskip("rdkit")
pytest.importorskip("umap")
pytest.importorskip("plotly")

import numpy as np

from modules.viz import mol_to_image, plot_umap


def test_plot_umap_returns_figure():
    data = np.random.rand(5, 3)
    fig = plot_umap(data)
    assert hasattr(fig, "data")


def test_mol_to_image_returns_image():
    img = mol_to_image("CCO")
    assert img.size[0] > 0 and img.size[1] > 0

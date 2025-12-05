import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("rdkit")

from modules.admet import predict_admet, compute_mpo

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


def test_predict_admet_keys():
    props = predict_admet(ASPIRIN)
    expected = {"logP", "tpsa", "hba", "hbd", "mol_wt", "rotatable_bonds"}
    assert expected.issubset(props.keys())


def test_compute_mpo_range():
    score = compute_mpo(ASPIRIN)
    assert 0.0 <= score <= 4.0

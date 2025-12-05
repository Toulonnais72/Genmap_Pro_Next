from __future__ import annotations

from typing import List, Tuple, Dict, Any
from functools import lru_cache
import base64

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Draw
from rdkit import DataStructs
import networkx as nx
from networkx.algorithms import community as nx_community
from pyvis.network import Network
import tempfile
from pathlib import Path
from streamlit.components.v1 import html as st_html

try:
    # Reuse existing warning system when available
    from Genmap_modules.status_manager import add_warning
except Exception:  # pragma: no cover - fallback outside Streamlit app
    def add_warning(message: str) -> None:
        """Fallback warning logger when Genmap status manager is unavailable."""
        try:
            st.warning(message)
        except Exception:
            # As a last resort, just ignore
            pass


@st.cache_data(show_spinner=False)
def compute_molecule_fingerprints(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    fp_type: str = "ECFP4",
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[List[Chem.Mol | None], List[DataStructs.ExplicitBitVect | None]]:
    """
    Convert SMILES in df[smiles_col] to RDKit Mol objects and compute fingerprints.

    fp_type currently supports "ECFP4" (Morgan) and "MACCS".
    Returns a tuple (mols, fps) with the same length as df.
    Invalid SMILES are skipped in the similarity computation (fingerprint = None)
    and reported via the Genmap warning system when available.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{smiles_col}' not found in DataFrame.")

    mols: List[Chem.Mol | None] = []
    fps: List[DataStructs.ExplicitBitVect | None] = []

    invalid_count = 0
    fp_type_norm = (fp_type or "ECFP4").upper()

    for smi in df[smiles_col].astype(str).tolist():
        smi = smi.strip()
        if not smi:
            mols.append(None)
            fps.append(None)
            invalid_count += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mols.append(None)
            fps.append(None)
            invalid_count += 1
            continue

        if fp_type_norm == "MACCS":
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

        mols.append(mol)
        fps.append(fp)

    if invalid_count:
        add_warning(
            f"{invalid_count} molecule(s) had invalid or empty SMILES and were excluded from the similarity network."
        )

    return mols, fps


@st.cache_data(show_spinner=False)
def compute_tanimoto_similarity_matrix(
    fps: List[DataStructs.ExplicitBitVect | None],
) -> np.ndarray:
    """
    Compute a symmetric Tanimoto similarity matrix (n_mols x n_mols)
    from a list of RDKit fingerprints.

    Entries corresponding to molecules with missing fingerprints (None)
    are set to 0.0 and the diagonal for valid fingerprints is set to 1.0.
    """
    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=float)

    # Indices with valid fingerprints
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    for idx_i, i in enumerate(valid_indices):
        fp_i = fps[i]
        if fp_i is None:
            continue
        sim_matrix[i, i] = 1.0
        for j in valid_indices[idx_i + 1 :]:
            fp_j = fps[j]
            if fp_j is None:
                continue
            s = DataStructs.TanimotoSimilarity(fp_i, fp_j)
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    return sim_matrix


@st.cache_data(show_spinner=False)
def build_similarity_graph(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    id_col: str = "molecule_id",
    similarity_threshold: float = 0.6,
) -> nx.Graph:
    """
    Build a NetworkX graph where nodes are molecules from df and
    edges connect molecules whose Tanimoto similarity is >= similarity_threshold.

    Node attributes include:
    - id_col (if present)
    - SMILES / smiles (when present)
    - all other columns from df as generic attributes.
    """
    if sim_matrix.shape[0] != len(df) or sim_matrix.shape[1] != len(df):
        raise ValueError(
            "Similarity matrix shape does not match DataFrame length."
        )

    G = nx.Graph()

    def _opt(row: pd.Series, col: str) -> Any:
        return row[col] if col in df.columns else None

    # Prepare node ids in DataFrame order
    node_ids: List[Any] = []
    for idx, row in df.iterrows():
        node_id = row[id_col] if id_col in df.columns else idx
        node_ids.append(node_id)

        attrs: Dict[str, Any] = row.to_dict()
        attrs["index"] = int(idx)
        attrs["molecule_id"] = node_id
        if id_col in df.columns:
            attrs[id_col] = row[id_col]

        # Ensure we keep a canonical SMILES attribute when present
        if "SMILES" in df.columns:
            attrs["SMILES"] = row["SMILES"]
        if "smiles" in df.columns:
            attrs["smiles"] = row["smiles"]

        # Optional scores / AD fields
        for col_name in ["score", "activity", "mpo_score", "ad_label", "ad_score"]:
            val = _opt(row, col_name)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                attrs[col_name] = val

        G.add_node(node_id, **attrs)

    # Add edges based on similarity threshold
    n = len(node_ids)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if s >= float(similarity_threshold):
                G.add_edge(
                    node_ids[i],
                    node_ids[j],
                    weight=s,
                    similarity=s,
                )

    # Populate cluster IDs by default
    cluster_similarity_graph(G)

    return G


def cluster_similarity_graph(G: nx.Graph) -> Dict[Any, int]:
    """
    Compute communities (clusters) on the similarity graph G.

    Uses networkx.algorithms.community.greedy_modularity_communities
    when the graph has at least one edge. For graphs with no edges,
    assigns a unique cluster to each node.

    Returns a dict {node_id: cluster_id} and writes cluster_id
    as a node attribute in G.
    """
    cluster_map: Dict[Any, int] = {}

    if G.number_of_nodes() == 0:
        return cluster_map

    if G.number_of_edges() == 0:
        # No structure: each node is its own cluster
        for idx, node in enumerate(G.nodes()):
            G.nodes[node]["cluster_id"] = idx
            cluster_map[node] = idx
        return cluster_map

    communities = nx_community.greedy_modularity_communities(G)
    for cid, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["cluster_id"] = cid
            cluster_map[node] = cid

    return cluster_map


@lru_cache(maxsize=2048)
def _smiles_to_data_url(smiles: str, size: int = 320) -> str | None:
    """
    Convert a SMILES string to a PNG data-URL for 2D depiction.

    The image is intentionally small so that many nodes remain readable
    at once; zooming in the PyVis view scales it up for better legibility.
    """
    try:
        smi = (smiles or "").strip()
        if not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(size, size))
        # Rendre le fond blanc transparent pour une meilleure intégration visuelle
        try:
            img = img.convert("RGBA")
            datas = img.getdata()
            new_data = []
            for r, g, b, a in datas:
                if r > 250 and g > 250 and b > 250:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append((r, g, b, a))
            img.putdata(new_data)
        except Exception:
            # En cas de problème avec la transparence, on garde l'image originale
            pass
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def render_similarity_network_streamlit(
    G: nx.Graph,
    height: str = "600px",
    width: str = "100%",
    notebook: bool = False,
    show_2d_images: bool = True,
    color_mode: str = "cluster",
    ad_label_palette: Dict[Any, str] | None = None,
) -> None:
    """
    Render the similarity graph G as an interactive network in Streamlit.

    Uses pyvis.Network, exports to an HTML file, then embeds it with
    st.components.v1.html.
    Node color reflects cluster_id when available.
    Node tooltip shows:
      - molecule ID
      - SMILES
      - activity / score / MPO when present.
    """
    if G.number_of_nodes() == 0:
        st.info("No molecules available for the similarity network.")
        return

    net = Network(height=height, width=width, notebook=notebook, directed=False)
    net.barnes_hut()

    if ad_label_palette is None:
        ad_label_palette = {
            "in_domain": "#2ecc71",
            "borderline": "#f1c40f",
            "out_of_domain": "#e74c3c",
            None: "#95a5a6",
        }

    for node_id, data in G.nodes(data=True):
        smiles = data.get("SMILES") or data.get("smiles") or ""

        def _first(keys: List[str]) -> Any:
            for k in keys:
                if k in data:
                    return data.get(k)
            return None

        # Collect tooltip lines with key attributes (short, human-readable)
        def _fmt(val: Any, decimals: int = 3) -> str:
            try:
                return f"{float(val):.{decimals}f}"
            except Exception:
                return str(val)

        title_lines: List[str] = []
        title_lines.append(f"ID: {data.get('molecule_id', node_id)}")

        # Prediction / activity / scores
        pred_val = _first(["prediction", "Pr\u00e9diction"])
        if pred_val is not None:
            title_lines.append(f"Prediction: {pred_val}")

        activity_val = _first(["activity", "Probabilit\u00e9 activit\u00e9", "probabilite", "probability"])
        if activity_val is not None:
            title_lines.append(f"Activity: {_fmt(activity_val)}")

        score_val = _first(["score"])
        if score_val is not None:
            title_lines.append(f"Score: {_fmt(score_val)}")

        mpo_val = _first(["mpo_score", "MPO_score"])
        if mpo_val is not None:
            title_lines.append(f"MPO: {_fmt(mpo_val)}")

        # AD info
        ad_label_val = data.get("ad_label")
        ad_score_val = data.get("ad_score")
        if ad_label_val is not None:
            if ad_score_val is not None:
                title_lines.append(f"AD: {ad_label_val} (score {_fmt(ad_score_val)})")
            else:
                title_lines.append(f"AD: {ad_label_val}")

        # Key ADMET / tox proxies when present
        logp_val = _first(["logp", "LogP"])
        if logp_val is not None:
            title_lines.append(f"logP: {_fmt(logp_val)}")
        tpsa_val = _first(["tpsa", "TPSA"])
        if tpsa_val is not None:
            title_lines.append(f"tPSA: {_fmt(tpsa_val)}")
        sol_val = _first(["solubility"])
        if sol_val is not None:
            title_lines.append(f"Solubility: {_fmt(sol_val)}")
        herg_val = _first(["herg_prob"])
        if herg_val is not None:
                title_lines.append(f"hERG risk: {_fmt(herg_val)}")
        ames_val = _first(["ames_prob"])
        if ames_val is not None:
            title_lines.append(f"Ames risk: {_fmt(ames_val)}")

        summary_val = _first(["Summary", "summary", "plain_summary", "Warnings", "warnings"])
        if summary_val:
            summary_text = str(summary_val).strip()
            summary_short = summary_text if len(summary_text) <= 220 else summary_text[:217] + "..."
            title_lines.append(f"Summary: {summary_short}")

        # Wrap for nicer spacing
        # Use plain text with newlines so PyVis renders it as a readable tooltip
        title = "\n".join(title_lines)

        group = data.get("cluster_id")
        color = None
        if color_mode == "ad_label":
            ad_lbl = data.get("ad_label")
            color = ad_label_palette.get(ad_lbl, ad_label_palette.get(None))

        # Slightly larger base depiction to remain readable; still scales with zoom.
        data_url = _smiles_to_data_url(smiles) if (show_2d_images and smiles) else None

        node_kwargs: Dict[str, Any] = {
            "label": str(node_id),
            "title": title,
            "group": group,
            # Larger font so ChemBL ID (or molecule_id) is clearly readable
            "font": {"size": 60},
        }
        if color_mode == "ad_label" and color:
            node_kwargs["color"] = color
        # If we have a 2D depiction, use it as the node image;
        # the size here is a base radius in pixels and scales with zoom.
        if data_url is not None:
            node_kwargs["shape"] = "image"
            node_kwargs["image"] = data_url
            # Larger base size; still scales with zoom in the viewer.
            node_kwargs["size"] = 400

        net.add_node(node_id, **node_kwargs)

    for u, v, edata in G.edges(data=True):
        weight = float(edata.get("similarity", edata.get("weight", 0.0)))
        net.add_edge(u, v, value=weight)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "similarity_network.html"
        net.write_html(str(out_path), notebook=notebook)
        html_content = out_path.read_text(encoding="utf-8")

    st_html(html_content, height=int(height.replace("px", "")) if height.endswith("px") else 600, scrolling=True)

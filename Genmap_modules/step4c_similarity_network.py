import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from uuid import uuid4
import json

from Genmap_modules.status_manager import add_warning
from similarity_network import (
    compute_molecule_fingerprints,
    compute_tanimoto_similarity_matrix,
    build_similarity_graph,
    cluster_similarity_graph,
    render_similarity_network_streamlit,
)

import streamlit.components.v1 as components

try:
    import py3Dmol
    _HAS_3D = True
except Exception:
    _HAS_3D = False


def _get_results_dataframe() -> pd.DataFrame | None:
    """
    Retrieve the main prediction results DataFrame from session state.

    Falls back to the ADMET-ready subset if needed.
    """
    df = st.session_state.get("prediction_results")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    df_admet = st.session_state.get("prediction_results_for_admet")
    if isinstance(df_admet, pd.DataFrame) and not df_admet.empty:
        return df_admet

    return None


def _render_3d_from_smiles(smiles: str, title: str | None = None) -> None:
    """Display a 3D representation from a SMILES string using a 3Dmol.js embed."""
    smi = (smiles or "").strip()
    if not smi:
        st.error("No SMILES available for 3D rendering.")
        return

    # Prepare source model
    molblock = _smiles_to_3d_molblock(smi)
    model_format = "mol" if molblock else "smi"
    model_data = molblock if molblock else smi

    width, height = 640, 420
    div_id = f"view-{uuid4().hex}"
    js_model = json.dumps(model_data)
    js_title = json.dumps(title or "")

    html = f"""
    <div id="{div_id}" style="width:{width}px; height:{height}px; position: relative;"></div>
    <script src="https://3dmol.org/build/3Dmol.js"></script>
    <script>
      (function() {{
        const element = document.getElementById("{div_id}");
        if (!element || typeof $3Dmol === "undefined") {{
          return;
        }}
        const viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
        viewer.addModel({js_model}, "{model_format}");
        viewer.setStyle({{stick:{{radius:0.18}}, sphere:{{scale:0.25, colorscheme:'Jmol'}}}});
        viewer.zoomTo();
        viewer.render();
        const legend = {js_title};
        if (legend) {{
          viewer.addLabel(legend, {{backgroundColor:'white', fontColor:'black', fontSize: 12}});
        }}
      }})();
    </script>
    """
    st.caption(f"3D view SMILES (truncated): {smi[:60]}{'…' if len(smi) > 60 else ''}")
    components.html(html, height=height + 40, width=width + 20)


def _smiles_to_3d_molblock(smiles: str) -> str | None:
    """Generate a 3D MolBlock from SMILES using RDKit; return None on failure."""
    try:
        # If RDKit or force-field is unavailable, fallback is handled by caller.
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass  # geometry is already embedded; proceed with current coords
        mol3d = Chem.RemoveHs(mol)
        return Chem.MolToMolBlock(mol3d)
    except Exception:
        return None


def run() -> None:
    """Interactive similarity network view for molecules predicted in Step 4."""
    st.header("Similarity Network & Applicability Domain")

    df = _get_results_dataframe()
    if df is None or df.empty:
        st.info("No prediction results available. Run Step 4 first.")
        return

    df_results = df.reset_index(drop=True).copy()

    # If ADMET/Tox results are available from step 4.5, merge key columns
    admet_df = st.session_state.get("admet_results")
    if isinstance(admet_df, pd.DataFrame) and not admet_df.empty:
        key = None
        if "SMILES" in df_results.columns and "SMILES" in admet_df.columns:
            key = "SMILES"
        elif "ChEMBL_ID" in df_results.columns and "ChEMBL_ID" in admet_df.columns:
            key = "ChEMBL_ID"
        if key is not None:
            # Keep only core ADMET columns to avoid clutter (but include summaries)
            main_cols = [c for c in [
                key,
                "logp", "tpsa", "solubility",
                "herg_prob", "ames_prob",
                "MPO_score", "Warnings", "Summary",
            ] if c in admet_df.columns]
            if len(main_cols) > 1:
                admet_sub = admet_df[main_cols].drop_duplicates(subset=[key])
                df_results = df_results.merge(admet_sub, on=key, how="left")

    st.write(f"{len(df_results)} molecule(s) available for similarity analysis.")

    # Display thresholds carried over from Step 4, if any
    ic50_thr = st.session_state.get("ic50_threshold")
    prob_thr = st.session_state.get("step4_admet_prob_threshold")
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"IC50 training threshold (Step 4): {ic50_thr} nM" if ic50_thr is not None else
                   "IC50 training threshold (Step 4): not set")
    with col2:
        if isinstance(prob_thr, (int, float)):
            st.caption(f"Activity probability threshold (Step 4): {prob_thr:.2f}")
        else:
            st.caption("Activity probability threshold (Step 4): not set")

    if len(df_results) < 3:
        st.info("Need at least 3 molecules to build a similarity network.")
        return

    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        sim_threshold = st.slider(
            "Tanimoto similarity threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
        )
    with col_settings2:
        color_mode = st.radio(
            "Color nodes by",
            options=["Cluster", "Applicability Domain"],
            index=0,
            help="Cluster: communities in the similarity graph; Applicability Domain: in-domain / borderline / out-of-domain.",
        )

    # Activity filter toggle
    act_filter_mode = st.radio(
        "Activity filter",
        options=["All molecules", "Active only"],
        index=0,
        horizontal=True,
        help="Choose whether to display all molecules or only those predicted active.",
    )

    # AD filters
    st.markdown("**Filter by Applicability Domain:**")
    col_ad1, col_ad2, col_ad3, col_ad4 = st.columns(4)
    with col_ad1:
        show_in_domain = st.checkbox("In-domain", value=True)
    with col_ad2:
        show_borderline = st.checkbox("Borderline", value=True)
    with col_ad3:
        show_out_of_domain = st.checkbox("Out-of-domain", value=True)
    with col_ad4:
        show_unknown_ad = st.checkbox("No AD info", value=True)

    mode = st.radio(
        "Network visualisation mode",
        options=["2D in network", "Labels + 3D viewer"],
        horizontal=True,
    )
    show_2d_images = mode == "2D in network"

    # Pick a SMILES column
    smiles_col = None
    for candidate in ["SMILES", "smiles"]:
        if candidate in df_results.columns:
            smiles_col = candidate
            break

    if smiles_col is None:
        st.error("No SMILES column found in prediction results.")
        return

    df_network = df_results.copy()
    if "ad_label" in df_network.columns:
        mask = pd.Series(False, index=df_network.index)
        if show_in_domain:
            mask |= df_network["ad_label"] == "in_domain"
        if show_borderline:
            mask |= df_network["ad_label"] == "borderline"
        if show_out_of_domain:
            mask |= df_network["ad_label"] == "out_of_domain"
        if show_unknown_ad:
            mask |= df_network["ad_label"].isna()
        if mask.any():
            df_network = df_network[mask]
        else:
            st.warning("No AD categories selected; showing all molecules.")
    else:
        st.info("AD information not available for this run; defaulting to all molecules.")

    # Optionally restrict to actives using prediction label
    if act_filter_mode == "Active only":
        pred_col = None
        for c in df_network.columns:
            if str(c).lower().startswith("pr"):
                pred_col = c
                break
        if pred_col is not None:
            mask_actives = df_network[pred_col].astype(str).str.contains("Actif", case=False, na=False) | \
                           df_network[pred_col].astype(str).str.contains("Active", case=False, na=False)
            df_network = df_network[mask_actives]
        else:
            st.warning("No prediction column found to filter actives; showing all molecules.")

    # Ensure a simple molecule identifier is available
    if "molecule_id" not in df_network.columns:
        name_col = None
        for cand in ["Nom de la mol\u00e9cule", "Name"]:
            if cand in df_network.columns:
                name_col = cand
                break

        ids = []
        has_chembl = "ChEMBL_ID" in df_network.columns
        for i, (_, row) in enumerate(df_network.iterrows(), start=1):
            cid = ""
            if has_chembl:
                val = row["ChEMBL_ID"]
                if pd.notna(val):
                    cid = str(val).strip()
            if cid:
                ids.append(cid)
                continue

            nm = ""
            if name_col is not None:
                valn = row[name_col]
                if pd.notna(valn):
                    nm = str(valn).strip()
            if nm:
                ids.append(nm)
            else:
                ids.append(f"Mol_{i:03d}")

        df_network["molecule_id"] = ids

    if len(df_network) < 3:
        st.warning("Not enough molecules after AD filtering to build the network.")
        return

    try:
        _, fps = compute_molecule_fingerprints(df_network, smiles_col=smiles_col)
        sim_matrix = compute_tanimoto_similarity_matrix(fps)
        G = build_similarity_graph(
            df_network,
            sim_matrix,
            id_col="molecule_id",
            similarity_threshold=sim_threshold,
        )
        cluster_map = cluster_similarity_graph(G)

        # Optionally expose cluster IDs in a small summary
        df_network["cluster_id"] = df_network["molecule_id"].map(cluster_map)

        with st.expander("Cluster summary", expanded=False):
            st.dataframe(
                df_network[["molecule_id", "cluster_id"]].value_counts("cluster_id").reset_index(name="size"),
                use_container_width=True,
            )

        color_mode_str = "cluster" if color_mode == "Cluster" else "ad_label"

        render_similarity_network_streamlit(
            G,
            show_2d_images=show_2d_images,
            color_mode=color_mode_str,
        )

        st.markdown("""
**Applicability Domain (AD) – Legend**
- ✅ **In-domain**: molecule lies within the dense training region → predictions more reliable.  
- ⚠️ **Borderline**: molecule near the edge of the training domain → interpret with caution.  
- ❌ **Out-of-domain**: molecule outside the training domain → predictions are extrapolations and may be unreliable.
""")

        st.subheader("3D visualisation and side-by-side comparison")
        if not _HAS_3D:
            st.info("3D visualisation is not available (py3Dmol not installed).")
        else:
            max_viewers = st.slider("Number of 3D viewers", 1, 4, 3, help="How many molecules to display in parallel.")

            # Prefill selection from the largest cluster when available
            prefill: list[str] = []
            if "cluster_id" in df_network.columns and df_network["cluster_id"].notna().any():
                cluster_sizes = df_network["cluster_id"].value_counts()
                if not cluster_sizes.empty:
                    top_cluster = cluster_sizes.index[0]
                    prefill = df_network[df_network["cluster_id"] == top_cluster]["molecule_id"].astype(str).head(max_viewers).tolist()
            cluster_label = st.selectbox(
                "Focus on cluster (prefills the selection)",
                options=[str(c) for c in df_network["cluster_id"].dropna().unique()],
                index=0 if df_network["cluster_id"].notna().any() else None,
                help="Cluster IDs originate from network community detection.",
            ) if df_network["cluster_id"].notna().any() else None
            if cluster_label is not None:
                cluster_prefill = df_network[df_network["cluster_id"].astype(str) == cluster_label]["molecule_id"].astype(str).head(max_viewers).tolist()
                if cluster_prefill:
                    prefill = cluster_prefill

            all_ids = df_network["molecule_id"].astype(str).tolist()
            selected_ids = st.multiselect(
                "Pick molecules to render in 3D (SMILES must be present)",
                options=all_ids,
                default=prefill[:max_viewers] if prefill else all_ids[:max_viewers],
            )

            if len(selected_ids) > max_viewers:
                st.warning(f"Showing the first {max_viewers} molecules.")
                selected_ids = selected_ids[:max_viewers]

            if not selected_ids:
                st.info("Select at least one molecule to render.")
            else:
                cols = st.columns(len(selected_ids))
                for col, mid in zip(cols, selected_ids):
                    with col:
                        st.markdown(f"**{mid}**")
                        row = df_network[df_network["molecule_id"].astype(str) == str(mid)].head(1)
                        if not row.empty:
                            smi_val = row.iloc[0][smiles_col]
                            _render_3d_from_smiles(str(smi_val), title=str(mid))
                        else:
                            st.error("SMILES not found for this molecule.")

            # Quick single-view dropdown below for detailed focus
            st.markdown("---")
            st.markdown("##### Quick focus on one molecule")
            choices = df_network["molecule_id"].astype(str).tolist()
            sel_id = st.selectbox("Choose a molecule for 3D view", choices, key="single_3d_viewer")
            row = df_network[df_network["molecule_id"].astype(str) == sel_id].head(1)
            if not row.empty:
                smi_val = row.iloc[0][smiles_col]
                _render_3d_from_smiles(str(smi_val), title=str(sel_id))
            else:
                st.info("No matching molecule found for 3D rendering.")
    except Exception as e:
        add_warning(f"Could not build similarity network: {e}")
        st.error("An error occurred while building the similarity network. See warnings for details.")

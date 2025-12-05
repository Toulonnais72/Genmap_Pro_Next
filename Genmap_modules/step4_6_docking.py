# -*- coding: utf-8 -*-
"""
step4_6_docking.py
Robust docking page with tool/file checks, explicit error surfacing, and 3D viewer.
Requires: autodock-vina (or smina/gnina), openbabel, py3Dmol
"""
import os, shutil, tempfile
import streamlit as st
from Genmap_modules.status_manager import add_warning
import pandas as pd
import py3Dmol
from pathlib import Path

# ===== utils (no external import)
def which_or_msg(bin_name: str):
    p = shutil.which(bin_name)
    return p

def parse_vina_scores_from_text(text: str):
    scores = []
    for line in text.splitlines():
        if "REMARK VINA RESULT" in line:
            parts = line.strip().split()
            try:
                scores.append(float(parts[3]))
            except Exception:
                pass
    return scores

def parse_vina_scores_from_pdbqt(pdbqt_text: str):
    return parse_vina_scores_from_text(pdbqt_text or "")

def run_cmd(cmd, cwd=None):
    import subprocess
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def smiles_to_pdbqt_obabel(smiles: str, out_pdbqt: Path, ph: float = 7.4):
    out_pdbqt = Path(out_pdbqt)
    with tempfile.TemporaryDirectory() as td:
        sdf = Path(td) / "lig.sdf"
        code, out = run_cmd(["obabel", "-:", smiles, "-O", str(sdf), "-p", f"{ph}", "--gen3d"])
        if code != 0 or not sdf.exists():
            raise RuntimeError(f"OpenBabel SDF generation failed:\n{out}")
        code, out = run_cmd(["obabel", str(sdf), "-O", str(out_pdbqt)])
        if code != 0 or not out_pdbqt.exists():
            raise RuntimeError(f"OpenBabel PDBQT conversion failed:\n{out}")
    return str(out_pdbqt)

def dock_with_vina_single(smiles: str, receptor_pdbqt: str, center: dict, size: dict,
                          exhaustiveness: int = 8, num_modes: int = 9, vina_bin: str = "vina"):
    with tempfile.TemporaryDirectory() as td:
        lig_pdbqt = Path(td) / "lig.pdbqt"
        out_pdbqt = Path(td) / "out.pdbqt"
        log_txt   = Path(td) / "log.txt"

        smiles_to_pdbqt_obabel(smiles, lig_pdbqt)

        cmd = [
            vina_bin,
            "--receptor", str(receptor_pdbqt),
            "--ligand",   str(lig_pdbqt),
            "--center_x", str(center["x"]), "--center_y", str(center["y"]), "--center_z", str(center["z"]),
            "--size_x",   str(size["x"]),   "--size_y",   str(size["y"]),   "--size_z",   str(size["z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--out", str(out_pdbqt),
            "--log", str(log_txt)
        ]
        code, out = run_cmd(cmd)
        # Try to get scores from all possible places
        all_scores = parse_vina_scores_from_text(out)
        if log_txt.exists():
            all_scores += parse_vina_scores_from_text(log_txt.read_text())
        docked_txt = out_pdbqt.read_text() if out_pdbqt.exists() else None
        all_scores += parse_vina_scores_from_pdbqt(docked_txt or "")

        best = min(all_scores) if all_scores else None
        if code != 0 and best is None:
            raise RuntimeError(f"{vina_bin} failed:\n{out}")

        return {"best_score_kcal_per_mol": best,
                "all_pose_scores": sorted(set(all_scores)) if all_scores else [],
                "docked_pdbqt": docked_txt,
                "raw_out": out}

def _get_pred_dataframe():
    df = st.session_state.get("prediction_results_for_admet")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        df = st.session_state.get("prediction_results")
    return df

def _viewer_3d(receptor_pdb_path: str, docked_pdbqt_text: str, width: int = 820, height: int = 540):
    view = py3Dmol.view(width=width, height=height)
    if receptor_pdb_path and Path(receptor_pdb_path).exists():
        with open(receptor_pdb_path, "r") as fh:
            view.addModel(fh.read(), "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": "lightgrey"}})
    if docked_pdbqt_text:
        view.addModel(docked_pdbqt_text, "pdbqt")
        # Model indices: receptor=0 (if present), ligand=1
        view.setStyle({"model": 1 if receptor_pdb_path and Path(receptor_pdb_path).exists() else 0}, {"stick": {}})
    view.zoomTo()
    return view

def run():
    st.header("Step 4.6 — Docking (AutoDock Vina)")

    # ==== Tool & file checks
    missing = []
    if which_or_msg("obabel") is None:
        missing.append("Open Babel (`obabel`)")
    # default engine: vina
    default_engine = "vina" if which_or_msg("vina") else ("smina" if which_or_msg("smina") else ("gnina" if which_or_msg("gnina") else None))
    if default_engine is None:
        missing.append("AutoDock Vina / smina / gnina")
    if missing:
        st.error("Missing tools: " + ", ".join(missing))
        st.stop()

    dfpred = _get_pred_dataframe()
    if dfpred is None or len(dfpred) == 0 or "SMILES" not in dfpred.columns:
        st.info("No molecules available (Step 4 results with a 'SMILES' column are required).")
        st.stop()

    # ==== Filter to predicted actives if the label is present
    only_actives = st.checkbox("Use only predicted actives", value=("Prédiction" in dfpred.columns))
    if only_actives and "Prédiction" in dfpred.columns:
        dfpred = dfpred[dfpred["Prédiction"].astype(str).str.contains("Actif", case=False, na=False)]
        if dfpred.empty:
            add_warning("No predicted actives after filtering.")
            st.info("Adjust filtering parameters or ensure predictions are available.")
            st.stop()

    st.subheader("Docking setup")
    # Allow upload OR path for receptor files
    up_pdbqt = st.file_uploader("Upload receptor .pdbqt (or leave empty and provide a path below)", type=["pdbqt"])
    up_pdb   = st.file_uploader("Upload receptor .pdb (optional, for cartoon view)", type=["pdb"])

    tmp_dir = tempfile.TemporaryDirectory()
    receptor_pdbqt_path = ""
    receptor_pdb_path = ""

    if up_pdbqt is not None:
        receptor_pdbqt_path = str(Path(tmp_dir.name) / "uploaded_receptor.pdbqt")
        open(receptor_pdbqt_path, "wb").write(up_pdbqt.read())
    else:
        receptor_pdbqt_path = st.text_input("Receptor PDBQT path", "receptor.pdbqt")

    if up_pdb is not None:
        receptor_pdb_path = str(Path(tmp_dir.name) / "uploaded_receptor.pdb")
        open(receptor_pdb_path, "wb").write(up_pdb.read())
    else:
        receptor_pdb_path = st.text_input("Receptor PDB path (optional for cartoon view)", "receptor.pdb")

    if not Path(receptor_pdbqt_path).exists():
        st.error(f"Receptor PDBQT not found: {receptor_pdbqt_path}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        cx = st.number_input("center_x", value=10.0, step=0.5)
        sx = st.number_input("size_x",   value=20.0, step=0.5)
    with col2:
        cy = st.number_input("center_y", value=10.0, step=0.5)
        sy = st.number_input("size_y",   value=20.0, step=0.5)
    with col3:
        cz = st.number_input("center_z", value=10.0, step=0.5)
        sz = st.number_input("size_z",   value=20.0, step=0.5)
    center = {"x": cx, "y": cy, "z": cz}
    size   = {"x": sx, "y": sy, "z": sz}

    vina_bin = st.selectbox("Docking engine", [e for e in ["vina", "smina", "gnina"] if which_or_msg(e)], index=0)
    exhaustiveness = st.slider("Exhaustiveness", 1, 64, 8, 1)
    num_modes = st.slider("Num modes", 1, 20, 9, 1)

    # ==== Select how many to dock
    if "Probabilité activité" in dfpred.columns:
        dfpred = dfpred.sort_values("Probabilité activité", ascending=False).reset_index(drop=True)
    top_n = st.number_input("Dock top-N molecules", min_value=1, max_value=len(dfpred), value=min(30, len(dfpred)))
    work = dfpred.head(int(top_n)).reset_index(drop=True)

    st.write(f"Ready to dock {len(work)} molecules.")
    errors = []

    if st.button("Run docking now"):
        scores, poses = [], []
        for i, row in work.iterrows():
            smi = str(row["SMILES"])
            try:
                res = dock_with_vina_single(
                    smiles=smi,
                    receptor_pdbqt=receptor_pdbqt_path,
                    center=center,
                    size=size,
                    exhaustiveness=int(exhaustiveness),
                    num_modes=int(num_modes),
                    vina_bin=vina_bin,
                )
                scores.append(res["best_score_kcal_per_mol"])
                poses.append(res["docked_pdbqt"])
                if res["best_score_kcal_per_mol"] is None:
                    errors.append((i, "No score parsed", res.get("raw_out", "")))
            except Exception as e:
                scores.append(None)
                poses.append(None)
                errors.append((i, str(e), ""))

        work["Docking score (kcal/mol)"] = scores
        work["Docked pose (pdbqt)"] = poses
        st.session_state["docking_results"] = work

        if errors:
            with st.expander("Docking issues (click to expand)"):
                for i, msg, raw in errors[:10]:
                    st.markdown(f"**Row {i}** — {msg}")
                    if raw:
                        st.code(raw, language="text")
                if len(errors) > 10:
                    st.write(f"... and {len(errors)-10} more.")

        if not errors or any(s is not None for s in scores):
            st.success("Docking finished.")
        else:
            st.error("Docking finished with errors and no scores. Check tool paths, receptor/box, and logs above.")

    dock = st.session_state.get("docking_results")
    if isinstance(dock, pd.DataFrame) and not dock.empty:
        show_cols = [c for c in ["Nom de la molécule", "SMILES", "Probabilité activité", "Docking score (kcal/mol)"] if c in dock.columns]
        st.dataframe(dock[show_cols], use_container_width=True, hide_index=True)

        st.subheader("3D viewer")
        idx = st.number_input("Row index to view in 3D", min_value=0, max_value=len(dock)-1, value=0)
        pose = dock.iloc[int(idx)]["Docked pose (pdbqt)"]
        if pose:
            view = _viewer_3d(receptor_pdb_path=receptor_pdb_path, docked_pdbqt_text=pose)
            st.components.v1.html(view._make_html(), height=560, scrolling=False)
        else:
            st.info("No docked pose available for the selected row (check errors above).")

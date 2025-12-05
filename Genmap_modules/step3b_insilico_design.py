# Genmap_modules/step3b_insilico_design.py
import streamlit as st
import random
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem import QED
from rdkit.Chem import Draw
from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem import rdChemReactions

from genmap_ml.inference import generate_vae_candidates_dataframe

# Prépare le catalogue PAINS une seule fois au niveau du module
try:
    _pains_params = FilterCatalogParams()
    _pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    _pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    _pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    PAINS_CAT = FilterCatalog.FilterCatalog(_pains_params)
except Exception:
    PAINS_CAT = None

# ---------- Utils ----------



def _mol(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def _canon(smi):
    try:
        m = _mol(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except Exception:
        return None

def _sascorer(mol):
    # SA score approximation (Ertl) — copie simplifiée pour éviter dépendances
    # On utilise la version RDKit contrib si dispo ; ici, heuristique minimaliste:
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
        rings = rdmd.CalcNumRings(mol)
        heavy = mol.GetNumHeavyAtoms()
        score = 1.0 + 0.02*heavy + 0.15*max(0, rings-1) + 0.05*len(frags)
        return float(score)
    except Exception:
        return 10.0

def _lipinski_ok(m):
    try:
        mw = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        hbd = Descriptors.NumHDonors(m)
        hba = Descriptors.NumHAcceptors(m)
        return (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
    except Exception:
        return False

def _pains_flag(m):
    try:
        return PAINS_CAT.HasMatch(m)
    except Exception:
        return True  # si erreur : pénalise

def _admet_proxy(m):
    # proxies légers : QED haut, TPSA modéré, MW raisonnable
    try:
        qed = float(QED.qed(m))
    except Exception:
        qed = 0.0
    try:
        tpsa = rdmd.CalcTPSA(m)
    except Exception:
        tpsa = 200.0
    try:
        mw = Descriptors.MolWt(m)
    except Exception:
        mw = 800.0
    # fenêtre douce : TPSA 20–120 / MW < 550
    tpsa_term = 1.0 - min(1.0, abs((tpsa-80.0))/100.0)
    mw_term = 1.0 - min(1.0, max(0.0, (mw-550.0)/250.0))
    return max(0.0, 0.5*qed + 0.25*tpsa_term + 0.25*mw_term)

def _fp(m):
    return rdmd.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)

def _clf_score_from_session(m):
    """Utilise le classifieur Step4 si disponible (proba d'activité)."""
    clf = st.session_state.get("clf")
    selected = st.session_state.get("selected_descriptors", [])
    # recompute minimal feature set si on peut
    if clf is None or not selected:
        return None
    try:
        # Calcule les descripteurs sélectionnés dans Step4 (subset)
        desc_vals = []
        from Genmap_modules.step4_prediction import DESC_OPTIONS  # réutilise mapping
        for d in selected:
            col, func = DESC_OPTIONS[d]
            desc_vals.append(func(m))
        # Morgan FP
        arr = np.zeros((1, 1024), dtype=int)
        bv = _fp(m)
        DataStructs.ConvertToNumpyArray(bv, arr[0])
        X = [desc_vals + list(arr[0])]
        proba = float(clf.predict_proba(X)[0][1])
        return proba
    except Exception:
        return None

def _surrogate_activity(m, actives_fps):
    """Si pas de clf, score de similarité Tanimoto vs actives (proxy activité)."""
    try:
        if not actives_fps:
            return 0.0
        fp_m = _fp(m)
        sims = [DataStructs.TanimotoSimilarity(fp_m, f) for f in actives_fps]
        return float(np.mean(sims))
    except Exception:
        return 0.0

def _score(m, actives_fps, w_act=0.5, w_admet=0.3, w_dl=0.2, pains_penalty=0.3):
    if m is None:
        return -1.0
    proba = _clf_score_from_session(m)
    if proba is None:
        proba = _surrogate_activity(m, actives_fps)
    admet = _admet_proxy(m)
    dl = 1.0 if _lipinski_ok(m) else 0.0
    if _pains_flag(m):
        admet *= (1.0 - pains_penalty)
        dl *= (1.0 - pains_penalty)
    # pénalise synthèse difficile
    sa = _sascorer(m)
    sa_term = max(0.0, 1.0 - (sa-2.0)/6.0)  # ~1 bon, ~0 mauvais
    # combine
    base = w_act*proba + w_admet*admet + w_dl*dl
    return 0.8*base + 0.2*sa_term

def _seed_library_from_session():
    """Return cleaned seed data (names + SMILES) and refresh actives_fps."""
    seeds = st.session_state.get("seeds_in") or []
    seeds_names = st.session_state.get("seeds_in_names") or []
    cleaned = []
    fps = []
    for idx, smi in enumerate(seeds):
        smi = (smi or "").strip()
        if not smi:
            continue
        m = _mol(smi)
        if not m:
            continue
        fps.append(_fp(m))
        try:
            canon = Chem.MolToSmiles(m, canonical=True)
        except Exception:
            canon = smi
        nm = seeds_names[idx] if idx < len(seeds_names) else f"Seed_{idx+1:03d}"
        cleaned.append({"Name": nm, "SMILES": canon, "Original": smi})
    if fps:
        st.session_state["actives_fps"] = fps
    return cleaned, fps

# ---------- Streamlit UI ----------


# ---------- Streamlit UI ----------
def run():
    st.header("🧬 3b. Elevage in-silico de nouvelles molécules")

    # paramètres récupérés de la session (scores, activité)
    seed_records, seeds_fps = _seed_library_from_session()
    actives_fps = seeds_fps or st.session_state.get("actives_fps", [])
    w_act = st.session_state.get("w_act", 0.6)
    w_adm = st.session_state.get("w_adm", 0.3)
    w_dl  = st.session_state.get("w_dl", 0.1)
    pains_pen = st.session_state.get("pains_penalty", 1.0)

    with st.expander("✨ Seeds coming from Step 4 (optional)", expanded=False):
        if seed_records:
            st.caption(f"{len(seed_records)} seed(s) received from predictions. They are used as similarity anchors.")
            st.dataframe(pd.DataFrame(seed_records), use_container_width=True)
        else:
            st.info("No seed molecule has been sent yet. Use Step 4 → “Send seeds to Step 3b”.")

    st.subheader("🧬 SMILES generation with VAE (in silico design)")

    with st.expander("Configure VAE generator", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            n_valid_target = st.number_input(
                "Number of valid VAE candidates to generate",
                min_value=50,
                max_value=5000,
                value=500,
                step=50,
            )
            batch_size = st.number_input(
                "Batch size (internal VAE generation)",
                min_value=128,
                max_value=2048,
                value=512,
                step=128,
            )
        with col2:
            temperature = st.slider(
                "Sampling temperature",
                min_value=0.3,
                max_value=1.5,
                value=0.7,
                step=0.1,
            )
            device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

        model_dir = st.text_input(
            "VAE model directory",
            value="models/smiles_vae_druggable",
        )

        if st.button("Generate VAE candidates"):
            with st.spinner("Generating SMILES with VAE, please wait..."):
                df_vae = generate_vae_candidates_dataframe(
                    model_dir=model_dir,
                    n_valid_target=int(n_valid_target),
                    batch_size=int(batch_size),
                    temperature=float(temperature),
                    max_length=None,
                    device=device,
                    source_label="smiles_vae_druggable_v1",
                    random_seed=0,
                )

            st.success(f"Generated {len(df_vae)} VAE candidates.")
            st.dataframe(df_vae.head(20))

            st.session_state["vae_candidates_df"] = df_vae

    if "vae_candidates_df" not in st.session_state:
        st.info("Générez des candidats avec le VAE ci-dessus pour construire l'espace de conception.")
        return

    df_vae = st.session_state["vae_candidates_df"]
    # Construire la liste de SMILES à partir des candidats VAE uniquement
    smiles_list = [str(s).strip() for s in df_vae.get("smiles", []).tolist()]
    pop = []
    for smi in smiles_list:
        if not smi:
            continue
        c = _canon(smi)
        if c:
            pop.append(c)
    pop = list(dict.fromkeys(pop))

    if not pop:
        st.error("Aucun candidat VAE exploitable (SMILES invalides ou vides).")
        return

    # --- Final scoring détaillé ---
    rows = []
    for smi in pop:
        m = _mol(smi)
        if not m:
            continue

        # probabilité d’activité
        proba = _clf_score_from_session(m)
        if proba is None:
            proba = _surrogate_activity(m, actives_fps)

        try:
            qed = float(QED.qed(m))
        except Exception:
            qed = 0.0

        try:
            tpsa = rdmd.CalcTPSA(m)
        except Exception:
            tpsa = float("nan")

        try:
            mw = Descriptors.MolWt(m)
        except Exception:
            mw = float("nan")

        try:
            sa = _sascorer(m)
        except Exception:
            sa = 10.0

        try:
            pains = _pains_flag(m)
        except Exception:
            pains = True

        try:
            dl = _lipinski_ok(m)
        except Exception:
            dl = False

        score_val = _score(m, actives_fps, w_act=w_act, w_admet=w_adm, w_dl=w_dl, pains_penalty=pains_pen)
        if score_val is None:
            score_val = -1.0

        seed_sim = _surrogate_activity(m, actives_fps) if actives_fps else None

        rows.append({
            "Name": f"Design_{len(rows)+1:03d}",
            "SMILES": smi,
            "Score (global)": round(float(score_val), 3),
            "Proba activité": round(float(proba), 3),
            "QED": round(float(qed), 3),
            "TPSA": round(float(tpsa), 1) if tpsa == tpsa else "",
            "MW": round(float(mw), 1) if mw == mw else "",
            "SA_score(~low=best)": round(float(sa), 2),
            "PAINS_flag": bool(pains),
            "Lipinski_OK": bool(dl),
            "Seed similarity": round(float(seed_sim), 3) if seed_sim is not None else None,
            "Origin": "In silico design (VAE)",
        })

    df = pd.DataFrame(rows)



    if df.empty:
        st.error("Aucun candidat généré à partir du VAE.")
        st.info("Astuce : ajustez les paramètres du générateur VAE (température, nombre de candidats valides).")
        return

    sort_col = "Score (global)" if "Score (global)" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    st.session_state["insilico_candidates"] = df
    st.session_state["design_df"] = df
    n_total = len(df)

    # === Réinjection dans Step 3 ======================================
    st.subheader("📥 Envoyer des candidats vers la sélection principale (Step 3)")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        top_step3 = int(
            st.slider(
                "Nombre de molécules à pousser dans Step 3",
                min_value=1,
                max_value=n_total,
                value=min(30, n_total),
                step=5,
                key="step3b_to_step3_top",
            )
        )
    with col_s2:
        replace_pool = st.checkbox(
            "Remplacer les molécules Step 3b existantes",
            value=False,
            help="Sinon, les nouvelles molécules seront ajoutées au pool existant (sans doublon SMILES).",
            key="step3b_to_step3_replace",
        )

    if st.button(f"📥 Ajouter {top_step3} candidat(s) au Step 3", key="step3b_push_to_step3"):
        subset = df.head(int(top_step3)).copy()
        payload = []
        for i, row in subset.iterrows():
            smi = str(row.get("SMILES", "")).strip()
            if not smi:
                continue
            payload.append({
                "Name": str(row.get("Name") or row.get("Nom") or f"Design_{i+1:03d}"),
                "SMILES": smi,
                "ChEMBL_ID": row.get("ChEMBL_ID"),
                "Source": "In silico design",
                "DesignScore": row.get("Score (global)"),
                "DesignProbability": row.get("Proba activité"),
                "DesignSeedSimilarity": row.get("Seed similarity"),
                "DesignQED": row.get("QED"),
                "DesignTPSA": row.get("TPSA"),
                "DesignMW": row.get("MW"),
            })

        if not payload:
            st.error("Aucune molécule exploitable à envoyer.")
        else:
            st.session_state.setdefault("insilico_entries", [])
            if replace_pool:
                st.session_state["insilico_entries"] = payload
                added = len(payload)
            else:
                existing = {
                    str(entry.get("SMILES", "")).strip()
                    for entry in st.session_state.get("insilico_entries", [])
                    if isinstance(entry, dict)
                }
                added = 0
                for record in payload:
                    key = record["SMILES"]
                    if not key or key in existing:
                        continue
                    st.session_state["insilico_entries"].append(record)
                    existing.add(key)
                    added += 1
            st.success(f"{added} molécule(s) prêtes dans Step 3.")
            st.caption("Ouvrez l'étape 3 pour les inclure / exclure dans le tableau principal.")

    # === Envoi vers Step 4 (Prédiction) ================================

    st.subheader("➡️ Envoyer les candidats vers la prédiction (Step 4)")

    # Options de sélection rapide
    colA, colB, colC = st.columns(3)
    with colA:
        top_n = int(
            st.slider(
                "Nombre de molécules à envoyer au Step 4",
                min_value=1,
                max_value=n_total,
                value=min(50, n_total),
                step=10,
            )
        )
    with colB:
        # Seuil sur la proba d’activité si dispo
        if "Proba activité" in df.columns:
            min_proba = st.slider("Seuil proba activité (min)", 0.0, 1.0, 0.0, 0.01)
        else:
            min_proba = 0.0
            st.caption("Seuil proba : non disponible (colonne absente).")
    with colC:
        dedupe_on_smiles = st.checkbox("Éviter les doublons (SMILES)", value=True)

    # Filtre: tri & top-N
    df2send = df.copy()
    if "Proba activité" in df2send.columns:
        df2send = df2send[df2send["Proba activité"] >= min_proba]
        df2send = df2send.sort_values("Proba activité", ascending=False)
    elif "Score (global)" in df2send.columns:
        df2send = df2send.sort_values("Score (global)", ascending=False)

    df2send = df2send.head(int(top_n)).reset_index(drop=True)

    # Bouton d'envoi
    if st.button(f"➕ Ajouter {len(df2send)} candidat(s) au Step 4"):
        # Prépare les conteneurs Step 4
        st.session_state.setdefault("to_predict", [])
        st.session_state.setdefault("to_predict_names", [])
        st.session_state.setdefault("to_predict_ids", [])  # peut rester None
        st.session_state.setdefault("to_predict_sources", [])
        # Aligne la longueur pour éviter les décallages lors d'un ajout manuel précédent
        current_len = len(st.session_state["to_predict_sources"])
        if current_len < len(st.session_state["to_predict"]):
            st.session_state["to_predict_sources"].extend([None] * (len(st.session_state["to_predict"]) - current_len))
        existing = set(st.session_state["to_predict"]) if dedupe_on_smiles else set()

        added = 0
        skipped = 0
        for i, row in df2send.iterrows():
            smi = str(row["SMILES"]).strip()
            if not smi:
                continue
            if dedupe_on_smiles and smi in existing:
                skipped += 1
                continue
            name = str(row.get("Nom") or row.get("Name") or f"Design_{i + 1:03d}")
            st.session_state["to_predict"].append(smi)
            st.session_state["to_predict_names"].append(name)
            st.session_state["to_predict_ids"].append(row.get("ChEMBL_ID") or None)
            st.session_state["to_predict_sources"].append("In silico design")
            existing.add(smi)
            added += 1

        st.success(f"{added} candidat(s) ajouté(s) au Step 4. {skipped} doublon(s) ignoré(s).")
        # Petit récap
        st.info(f"Total actuel à prédire : {len(st.session_state['to_predict'])} molécules.")

    st.success(f"{len(df)} candidats générés.")
    st.dataframe(df, use_container_width=True)
    show_structures = st.checkbox(
        "Afficher les structures 2D",
        key="show_structures",
    )
    if show_structures:
        for _, row in df.head(24).iterrows():
            smi = str(row.get("SMILES", ""))
            name = str(row.get("Nom") or row.get("Name") or "")
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    img = Draw.MolToImage(mol, size=(200, 200))
                    st.image(img, caption=f"2D - {name}")
            except Exception:
                continue
    st.download_button("💾 Télécharger (CSV)", df.to_csv(index=False).encode("utf-8"), "insilico_candidates.csv")

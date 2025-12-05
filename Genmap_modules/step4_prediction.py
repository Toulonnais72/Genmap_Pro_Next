import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import requests
from functools import lru_cache
from urllib.parse import quote_plus
import re

from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator as rFG
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import requests
import matplotlib.pyplot as plt
from Genmap_modules.status_manager import add_warning
from modules.applicability_domain import compute_knn_applicability_domain
import time
from contextlib import contextmanager
import streamlit.components.v1 as components

RDLogger.DisableLog('rdApp.*')
pd.set_option("styler.render.max_elements", 999999)


# --- Chrono simple, utilisable comme context manager ---
@contextmanager
def timer(label: str, store_key: str = "timings", show: bool = True):
    t0 = time.perf_counter()
    # zone d'affichage (ligne en cours)
    ph = st.empty() if show else None
    if ph:
        ph.info(f"⏱️ {label}…")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        # stocke la durée
        st.session_state.setdefault(store_key, [])
        st.session_state[store_key].append({"Step": label, "Duration(s)": round(dt, 3)})
        if ph:
            ph.success(f"✅ {label} – {dt:.2f} s")


@st.cache_data(show_spinner=False)
def _cached_applicability_domain(
    X_train: np.ndarray,
    X_new: np.ndarray,
    n_neighbors: int = 5,
    quantiles: tuple[float, float] = (0.80, 0.95),
):
    """Cached wrapper for applicability-domain computation."""
    return compute_knn_applicability_domain(
        X_train, X_new, n_neighbors=n_neighbors, quantiles=quantiles
    )

@lru_cache(maxsize=20000)
def chembl_id_from_smiles(smi: str) -> str | None:
    """Retourne l'ID ChEMBL d'un SMILES par match exact (si trouvé), sinon None."""
    if not smi:
        return None
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
            params={"molecule_structures__canonical_smiles": smi, "limit": 1},
            timeout=15,
        )
        if r.ok:
            items = r.json().get("molecules", [])
            if items:
                return items[0].get("molecule_chembl_id")
    except Exception:
        pass
    return None
# ================= Optional 3D viz (no stmol hard dep) =================
try:
    import py3Dmol
    _HAS_STMOL = True
    def showmol(view, height=400, width=400):
        html = view._make_html()
        st.components.v1.html(html, height=height, width=width)
except Exception:
    _HAS_STMOL = False

# ================= RDKit descriptors (hardened) =================
from rdkit.Chem import Descriptors
try:
    from rdkit.Chem import Descriptors3D
except Exception:
    Descriptors3D = None
from rdkit.Chem import rdMolDescriptors as rdmd
try:
    from rdkit.Chem import QED as QEDmod
except Exception:
    QEDmod = None

# ---- Helpers ----
def count_num_aromatic_atoms(mol):
    try:
        return rdmd.CalcNumAromaticAtoms(mol)
    except Exception:
        return sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

def safe_labute_asa(mol):
    try:
        return rdmd.CalcLabuteASA(mol)[0]
    except Exception:
        return float('nan')

# Descriptor dictionary (conditional availability)
DESC_OPTIONS_RAW = {
    "Poids moléculaire":           ("MolWt", Descriptors.MolWt),
    "Poids moléculaire exact":     ("ExactMolWt", Descriptors.ExactMolWt),
    "LogP":                        ("MolLogP", Descriptors.MolLogP),
    "TPSA":                        ("TPSA", Descriptors.TPSA),
    "Donneurs H":                  ("NumHDonors", Descriptors.NumHDonors),
    "Accepteurs H":                ("NumHAcceptors", Descriptors.NumHAcceptors),
    "Liaisons rotatables":         ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    "Fraction Csp3":               ("FractionCSP3", Descriptors.FractionCSP3),
    "Atomes lourds":               ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    "Cycles (total)":              ("RingCount", Descriptors.RingCount),
    "Cycles aromatiques":          ("NumAromaticRings", Descriptors.NumAromaticRings),
    "Cycles aliphatiques":         ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    "Cycles saturés":              ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    "Nb. d'atomes aromatiques":    ("NumAromaticAtoms", count_num_aromatic_atoms),
    "Hétéroatomes":                ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    "Électrons de valence":        ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    "Électrons radicaux":          ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    "Hall–Kier Alpha":             ("HallKierAlpha", Descriptors.HallKierAlpha),
    "Kappa1":                      ("Kappa1", Descriptors.Kappa1),
    "Kappa2":                      ("Kappa2", Descriptors.Kappa2),
    "Kappa3":                      ("Kappa3", Descriptors.Kappa3),
    "Bertz CT":                    ("BertzCT", Descriptors.BertzCT),
    "Indice de réfraction (MR)":   ("MolMR", Descriptors.MolMR),
    "Labute ASA":                  ("LabuteASA", safe_labute_asa),
    "Proportion aromatique":       ("AromaticProportion", lambda m: (sum(a.GetIsAromatic() for a in m.GetAtoms())/m.GetNumAtoms()) if m.GetNumAtoms() else 0.0),
}

# Add QED if available
if hasattr(rdmd, "CalcQED"):
    DESC_OPTIONS_RAW["QED"] = ("QED", rdmd.CalcQED)
elif QEDmod is not None and hasattr(QEDmod, "qed"):
    DESC_OPTIONS_RAW["QED"] = ("QED", QEDmod.qed)

# Add 3D if available
if Descriptors3D is not None and hasattr(Descriptors3D, "MolVol"):
    DESC_OPTIONS_RAW["Volume moléculaire"] = ("MolVolume", Descriptors3D.MolVol)


def filter_available_descriptors(desc_dict):
    test_mol = Chem.MolFromSmiles("CCO")
    available = {}
    for label, (col, func) in desc_dict.items():
        try:
            v = func(test_mol)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                continue
            available[label] = (col, func)
        except Exception:
            pass
    return available

DESC_OPTIONS = filter_available_descriptors(DESC_OPTIONS_RAW)

# ================= Data acquisition (ChEMBL) =================
BASE = "https://www.ebi.ac.uk"

def get_activities_via_api(chembl_id, max_records=2000):
    url = f"{BASE}/chembl/api/data/activity.json?limit=100&offset=0&target_chembl_id={chembl_id}&standard_type=IC50"
    activities = []
    while url and len(activities) < max_records:
        if url.startswith("/"):
            url = BASE + url
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            break
        data = r.json()
        acts = data.get("activities", [])
        for entry in acts:
            smiles = entry.get("canonical_smiles")
            if not smiles:
                ms = entry.get("molecule_structures")
                if ms:
                    smiles = ms.get("canonical_smiles")
            entry["_smiles"] = smiles
        activities += acts
        url = data.get("page_meta", {}).get("next")
    return activities


def build_training_df(target_id, seuil, source_df=None, max_records=2000):
    """Build a clean training dataframe.
    If source_df is provided, it must have columns: SMILES, IC50(nM) in nM; otherwise fetch from API and filter.
    Returns columns: smiles, ic50, active, mol
    """
    if source_df is not None and isinstance(source_df, pd.DataFrame) and not source_df.empty:
        df = (
            source_df.rename(columns={"SMILES": "smiles", "IC50(nM)": "ic50"})
            .dropna(subset=["smiles", "ic50"]).copy()
        )
        df = df.groupby("smiles", as_index=False)["ic50"].min()
    else:
        acts = get_activities_via_api(target_id, max_records=max_records)
        rows = []
        for a in acts:
            smi = a.get("_smiles")
            val = a.get("standard_value")
            units = (a.get("standard_units") or "").lower()
            rel = (a.get("standard_relation") or "=")
            if not smi or val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            if units and units != "nm":
                continue
            if rel and rel != "=":
                continue
            rows.append({"smiles": smi, "ic50": v})
        if not rows:
            return pd.DataFrame(columns=["smiles", "ic50", "active", "mol"])
        df = pd.DataFrame(rows)
        df = df.groupby("smiles", as_index=False)["ic50"].min()

    df["active"] = (df["ic50"] < seuil).astype(int)
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()].reset_index(drop=True)
    return df


def compute_features(df, selected_labels):
    desc_cols = [DESC_OPTIONS[d][0] for d in selected_labels]
    for d in selected_labels:
        colname, func = DESC_OPTIONS[d]
        df[colname] = df["mol"].apply(func)
    mg = rFG.GetMorganGenerator(radius=2, fpSize=1024)
    df["fp"] = df["mol"].apply(lambda m: mg.GetFingerprintAsNumPy(m).tolist())
    X = [list(d) + f for d, f in zip(df[desc_cols].values, df["fp"])]
    return X, desc_cols

def predict_table(
    smiles_list,
    names_list,
    chembl_ids,
    clf,
    selected_labels,
    X_train=None,
    ad_neighbors: int = 5,
    ad_quantiles: tuple[float, float] = (0.80, 0.95),
    sources_list=None,
):
    """Retourne un DataFrame des prédictions pour un lot de molécules (un seul timer autour, côté run())."""
    mg = rFG.GetMorganGenerator(radius=2, fpSize=1024)

    # 1) Construire mols valides + features en une passe
    mols, names_eff, ids_eff, feats, desc_vals_all, smiles_eff, sources_eff = [], [], [], [], [], [], []
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mols.append(m)
        smiles_eff.append(smi)
        names_eff.append(names_list[i] if names_list and len(names_list) == len(smiles_list) else smi)
        ids_eff.append(chembl_ids[i] if chembl_ids and len(chembl_ids) == len(smiles_list) else None)
        src_val = None
        if sources_list:
            if len(sources_list) == len(smiles_list):
                src_val = sources_list[i]
            elif i < len(sources_list):
                src_val = sources_list[i]
        sources_eff.append(src_val)
        # descripteurs
        desc_vals = [DESC_OPTIONS[d][1](m) for d in selected_labels]
        desc_vals_all.append(desc_vals)
        # empreinte
        fp = mg.GetFingerprintAsNumPy(m).tolist()
        feats.append(desc_vals + fp)

    if not feats:
        cols = [
            "Nom de la molécule",
            "SMILES",
            "ChEMBL_ID",
            "Prédiction",
            "Probabilité activité",
            "ad_score",
            "ad_label",
        ] + selected_labels
        return pd.DataFrame(columns=cols)

    # 2) Prédictions en lot (une seule passe)
    X = np.asarray(feats, dtype=float)
    y_hat = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]  # numérique

    # 2b) Applicability domain
    ad_scores = np.full(len(X), np.nan)
    ad_labels = np.array(["unknown"] * len(X), dtype=object)
    if X_train is not None:
        try:
            ad_scores, ad_labels = _cached_applicability_domain(
                np.asarray(X_train, dtype=float),
                X,
                n_neighbors=ad_neighbors,
                quantiles=ad_quantiles,
            )
        except Exception as exc:
            add_warning(f"Applicability Domain computation failed: {exc}")

    # 3) Construction du DataFrame
    rows = []
    for i in range(len(smiles_eff)):
        row = {
            "Nom de la molécule": names_eff[i],
            "SMILES": smiles_eff[i],
            "ChEMBL_ID": ids_eff[i],
            "Prédiction": "🟩 Actif" if int(y_hat[i]) == 1 else "🟥 Inactif",
            "Probabilité activité": float(proba[i]),  # numérique
            "ad_score": float(ad_scores[i]) if not np.isnan(ad_scores[i]) else np.nan,
            "ad_label": str(ad_labels[i]) if ad_labels is not None else None,
        }
        if sources_eff[i]:
            row["Source"] = sources_eff[i]
        for j, d in enumerate(selected_labels):
            row[d] = desc_vals_all[i][j]
        rows.append(row)

    dfpred = pd.DataFrame(rows)
    return dfpred



# ========================= Streamlit UI =========================

def run():
    st.header("4. Molecular Activity Prediction — automatic training from Step 3")
    show_timer = st.checkbox("Display the time of the steps", value=True)
    st.session_state["show_timer"] = show_timer
    # reset optionnel à chaque passage si tu veux des chiffres frais
    if st.button("🔁 Reinitialize timers"):
        st.session_state["timings"] = []
        st.toast("Timers reinitialized", icon="⏱️")

    # --- Descriptor selection ---
    st.markdown("#### Selection of molecular descriptors")
    all_descriptors = list(DESC_OPTIONS.keys())
    default_desc = [d for d in ["Poids molǸculaire", "LogP", "TPSA", "Donneurs H", "Accepteurs H", "Liaisons rotatables"] if d in DESC_OPTIONS]

    # Persistent selection used as single source of truth (separate from widget key)
    if "step4_selected_descriptors" not in st.session_state:
        st.session_state["step4_selected_descriptors"] = default_desc.copy()

    # Bulk controls to select / deselect all descriptors
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("Select all descriptors", key="step4_select_all_desc"):
            st.session_state["step4_selected_descriptors"] = all_descriptors.copy()
    with c2:
        if st.button("Deselect all descriptors", key="step4_deselect_all_desc"):
            st.session_state["step4_selected_descriptors"] = []
    default_desc = [d for d in ["Poids moléculaire", "LogP", "TPSA", "Donneurs H", "Accepteurs H", "Liaisons rotatables"] if d in DESC_OPTIONS]
    # Ensure widget state is initialised from our persistent selection
    if "step4_selected_descriptors" not in st.session_state:
        st.session_state["step4_selected_descriptors"] = default_desc.copy()

    selected = st.multiselect(
        "Chose the descriptors to be used in the model:",
        list(DESC_OPTIONS.keys()),
        key="step4_selected_descriptors",
    )
    if not selected:
        add_warning("Please choose at least one descriptor.")
        st.info("Select at least one descriptor to build the model.")
        st.stop()

    # Persist for Step 5 or re-use (and for Step 3b surrogate scoring)
    st.session_state["selected_descriptors"] = selected
    st.session_state["selected_desc"] = selected
    st.session_state["desc_options"] = DESC_OPTIONS

    target_id = st.session_state.get('selected_target_chembl_id', None)
    if not target_id:
        st.error("No ChemBL target selectr (cf step 2).")
        return

    # IC50 threshold is now controlled from the main sidebar (genmap2.py)
    seuil = st.session_state.get("ic50_threshold", 1000)
    try:
        seuil = int(seuil)
    except (TypeError, ValueError):
        seuil = 1000
        st.session_state["ic50_threshold"] = seuil

    # >>> NEW: Try using the dataframe provided by Step 3 if present
    source_df = None
    for k in ("chembl_df_for_training", "df_chembl", "chembl_df_cached"):
        if isinstance(st.session_state.get(k), pd.DataFrame) and not st.session_state[k].empty:
            source_df = st.session_state[k]
            break

    # Build clean training df (from Step 3 DF if available, else API)
    df = build_training_df(target_id, seuil, source_df=source_df, max_records=2000)
    nb_mols = len(df)
    if nb_mols == 0:
        st.error("No usable molecule IC50 (nM, relation '=').")
        return

    # Distribution IC50
    st.markdown("#### IC50 (nM) distribution for this target:")
    st.write(df["ic50"].describe())
    fig = plt.figure(figsize=(25,20))
    plt.hist(df["ic50"], bins=30)
    plt.xlabel("IC50 (nM)")
    plt.ylabel("Count")
    plt.title("IC50 Distribution")
    st.pyplot(fig)

    # Compute features (numpy for downstream AD distances)
    X, desc_cols = compute_features(df, selected)
    X_array = np.asarray(X, dtype=float)
    y = df["active"].to_numpy()

    # Distribution des classes
    classes, counts = np.unique(y, return_counts=True)
    st.write(f"**Classes distribution:** {dict(zip(classes, counts))}")

    if len(classes) < 2:
        st.error("All molecules are either all active or all inactive depending on the threshold chosen. "
                 "Cannot train the model. Try another target or adjust the threshold.")
        st.session_state.clf = None
        return

    min_count = counts.min()

    # Heuristique pour gérer le cas minoritaire < 2
    if min_count < 2:
        add_warning("Minority class with only 1 sample: training without split (no test accuracy). Load more CheMBL data or adjust the IC50 (nm) threshold to balance.")
        st.info("Training on the full dataset; no hold-out accuracy will be reported.")
        X_train, y_train = X_array, y
        X_test = np.empty((0, X_array.shape[1]))
        y_test = np.array([])
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y, test_size=0.2, random_state=42, stratify=y
        )

    # Pondération de la classe positive (si déséquilibré)
    pos = (np.array(y_train) == 1).sum()
    neg = (np.array(y_train) == 0).sum()
    scale_pos = max(1.0, neg / pos) if pos > 0 else 1.0

    clf_ = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist',
        eval_metric='logloss',
        scale_pos_weight=scale_pos,
        n_jobs=0,
    )
    clf_.fit(X_train, y_train)

    # Affichage des métriques selon la présence d’un test set

    if len(y_test) > 0:
        acc = clf_.score(X_test, y_test)
        metric_text = f"Accuracy (test): {acc:.2f}"
    else:
        acc_train = clf_.score(X_train, y_train)
        metric_text = f"Accuracy (train): {acc_train:.2f} (no test set)"

    st.success(f"Model trained on {len(X_train)} molecules | {metric_text}")

    st.session_state.clf = clf_
    st.session_state.df = df
    st.session_state["ad_train_features"] = X_train
    ad_quantiles = (0.80, 0.95)
    dfpred = pd.DataFrame()

    ###################### === P  R  E  D  I  C  T  I  O  N  S on Step 3 molecules (with names) === ########################

    if st.session_state.get("clf") is not None and st.session_state.get("to_predict"):
        names = st.session_state.get("to_predict_names", [])
        ids = st.session_state.get("to_predict_ids", [])
        smiles_in = st.session_state["to_predict"]

        # Un seul timer, autour du **lot entier**
        with timer(f"Predictions ({len(smiles_in)} molecules)", show=show_timer):
            dfpred = predict_table(
                smiles_in,
                names,
                ids,
                st.session_state.clf,
                selected,
                X_train=X_train,
                ad_neighbors=5,
                ad_quantiles=ad_quantiles,
                sources_list=st.session_state.get("to_predict_sources"),
            )

        # --- Récupération rapide des IDs ChEMBL déjà présents dans la colonne "Nom de la molécule"
        if "Molecule name" in dfpred.columns:
            mask_missing = dfpred["ChEMBL_ID"].isna() | (dfpred["ChEMBL_ID"].astype(str).str.strip() == "")
            for idx in dfpred[mask_missing].index:
                nom = str(dfpred.at[idx, "Molecule name"]).strip()
                if re.fullmatch(r"CHEMBL\d+", nom, flags=re.IGNORECASE):
                    dfpred.at[idx, "ChEMBL_ID"] = nom.upper()

        # --- Enrichir les molécules issues du CSV : retrouver les IDs ChEMBL manquants ---
        resolve_ids = st.checkbox("🔗 Try to get the ChemBL IDs for the molecules imported (exact SMILES)",
                                  value=True)
        max_lookup = st.number_input("Search limit (security)", min_value=0, max_value=50000, value=2000,
                                     step=100)

        if resolve_ids and "ChEMBL_ID" in dfpred.columns:
            missing_mask = dfpred["ChEMBL_ID"].isna() | (dfpred["ChEMBL_ID"].astype(str).str.strip() == "")
            to_resolve_idx = dfpred[missing_mask].head(int(max_lookup)).index
            if len(to_resolve_idx) > 0:
                ph = st.empty()
                for k, idx in enumerate(to_resolve_idx, start=1):
                    smi = dfpred.at[idx, "SMILES"]
                    cid = chembl_id_from_smiles(smi)
                    if cid:
                        dfpred.at[idx, "ChEMBL_ID"] = cid
                    if k % 100 == 0 or k == len(to_resolve_idx):
                        ph.info(f"ChEMBL Resolution : {k}/{len(to_resolve_idx)} processed…")
                ph.success("ChemBL resolution finalized.")

        ################################# --- RENDU DU RÉCAP --- ###############################################

        # Colonnes clés + tri
        df_show = dfpred.copy()
        if "Probabilité activité" in df_show.columns:
            df_show = df_show.sort_values("Probabilité activité", ascending=False)

        # URL ChemBL (si ID présent)
        def _chembl_url(x):
            x = "" if x is None else str(x).strip()
            return f"https://www.ebi.ac.uk/chembl/compound_report_card/{x}/" if x else None

        df_show["ChEMBL_URL"] = df_show["ChEMBL_ID"].apply(_chembl_url)

        # SMILES : on garde le complet en interne, on ajoute un **aperçu** pour l'affichage "résumé"
        df_show["SMILES (complet)"] = df_show["SMILES"]
        df_show["SMILES (aperçu)"] = df_show["SMILES"].astype(str).str.slice(0, 90) + \
                                     df_show["SMILES"].astype(str).map(lambda s: "…" if len(s) > 90 else "")

        # Applicability Domain formatting
        ad_badges = {
            "in_domain": "🟢 In-domain",
            "borderline": "🟠 Borderline",
            "out_of_domain": "🔴 Out-of-domain",
        }
        if "ad_label" in df_show.columns:
            df_show["AD label"] = df_show["ad_label"].map(ad_badges).fillna(df_show["ad_label"])
        if "ad_score" in df_show.columns:
            df_show["AD score"] = df_show["ad_score"]

        st.markdown(
            """
            **Applicability Domain (AD)**  
            - ✅ **In-domain**: molecule lies within the dense training region → predictions are more reliable.  
            - ⚠️ **Borderline**: molecule near the edge of the training domain → interpret with caution.  
            - ❌ **Out-of-domain**: molecule outside the training domain → predictions are extrapolations and may be unreliable.
            """
        )
        ad_col1, ad_col2 = st.columns(2)
        show_in_domain_only = ad_col1.checkbox("Show only in-domain molecules", value=False, key="ad_only_in_domain")
        highlight_ood = ad_col2.checkbox("Highlight out-of-domain predictions", value=True, key="ad_highlight_out_domain")

        df_filtered = df_show.copy()
        if show_in_domain_only and "ad_label" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["ad_label"] == "in_domain"].reset_index(drop=True)

        def _style_ad_rows(row):
            if not highlight_ood:
                return [""] * len(row)
            label = str(row.get("AD label", "")).lower()
            color = ""
            if "out-of-domain" in label:
                color = "#fde2e1"
            elif "borderline" in label:
                color = "#fff4e5"
            elif "in-domain" in label:
                color = "#d1f2d9"
            return [f"background-color: {color}"] * len(row) if color else [""] * len(row)

        # ————— TABLEAU RÉSUMÉ (lisible) —————
        summary_cols = [c for c in [
            "Nom de la molécule", "SMILES (aperçu)",
            "Prédiction", "Probabilité activité", "AD label", "AD score", "ChEMBL_URL"
        ] if c in df_filtered.columns]
        summary = df_filtered[summary_cols].reset_index(drop=True)
        summary_display = summary
        if not summary.empty:
            summary_display = summary.style.apply(_style_ad_rows, axis=1)
            if "AD score" in summary.columns:
                summary_display = summary_display.background_gradient(subset=["AD score"], cmap="Greens")

        st.subheader("Summary table of the predictions")
        st.dataframe(
            summary_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Nom de la molécule": st.column_config.TextColumn("Name"),
                "SMILES (aperçu)": st.column_config.TextColumn("SMILES"),
                "Prédiction": st.column_config.TextColumn("Prediction"),
                "Probabilité activité": st.column_config.NumberColumn("Activity probability", format="%.3f"),
                "AD score": st.column_config.NumberColumn("AD score", format="%.3f", help="0=close to training set, 1=farther"),
                "AD label": st.column_config.TextColumn("AD label"),
                "ChEMBL_URL": st.column_config.LinkColumn("ChEMBL", display_text="Card"),
            },
        )

        # ————— TABLEAU COMPLET (toutes colonnes) —————
        # 1) on supprime l'ancien 'SMILES' pour éviter les doublons
        df_tmp = df_filtered.drop(columns=["SMILES"], errors="ignore").copy()
        # 2) on renomme la colonne 'SMILES (complet)' en 'SMILES'
        df_tmp = df_tmp.rename(columns={"SMILES (complet)": "SMILES"})

        # Colonnes de base (dans l'ordre voulu)
        base_cols = ["Nom de la molécule", "SMILES", "ChEMBL_ID", "ChEMBL_URL",
                     "Prédiction", "Probabilité activité"]
        ad_cols = [c for c in ["ad_score", "ad_label", "AD score", "AD label"] if c in df_tmp.columns]
        base_cols = base_cols + ad_cols

        # Colonnes auxiliaires / à ignorer
        drop_aux = {"SMILES (aperçu)"}  # on a déjà 'SMILES' complet maintenant

        # Descripteurs = tout le reste sauf base + aux
        desc_cols = [c for c in df_tmp.columns if c not in set(base_cols) | drop_aux]

        # DataFrame final sans noms de colonnes dupliqués
        full_df = df_tmp[base_cols + desc_cols].reset_index(drop=True)
        full_display = full_df
        if len(full_df):
            full_display = full_df.style.apply(_style_ad_rows, axis=1)
            subset_cols = [c for c in ["AD score", "ad_score"] if c in full_df.columns]
            if subset_cols:
                full_display = full_display.background_gradient(subset=subset_cols, cmap="Greens")

        st.expander("🔎 Voir toutes les colonnes / descripteurs").dataframe(
            full_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probabilité activité": st.column_config.NumberColumn("Activity probability", format="%.3f"),
                "AD score": st.column_config.ProgressColumn("AD score", min_value=0.0, max_value=1.0, format="%.3f"),
                "ChEMBL_URL": st.column_config.LinkColumn("ChEMBL", display_text="Card"),
            },
        )

        # Téléchargements (clés uniques)
        st.download_button(
            "Download (summary) CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"prediction_summary_{len(summary)}.csv",
            mime="text/csv",
            key=f"dl_summary_{len(summary)}",
        )
        st.download_button(
            "Download (full) CSV",
            data=full_df.to_csv(index=False).encode("utf-8"),
            file_name=f"prediction_full_{len(full_df)}.csv",
            mime="text/csv",
            key=f"dl_full_{len(full_df)}",
        )

        # ============================================================
        # ↩️  Renvoyer des candidats vers Step 3b comme graines
        # ============================================================

        st.subheader("↩️ Send candidates back to Step 3b (BRICS seeds)")

        if 'prediction_results' in st.session_state and isinstance(st.session_state['prediction_results'],
                                                                   pd.DataFrame):
            df_all = st.session_state['prediction_results'].copy()
        else:
            df_all = dfpred.copy()

        # Sélection des sources (résumé ou complet)
        source_choice = st.radio(
            "Source",
            options=["Top visibles (résumé)", "Table complète"],
            horizontal=True,
            key="step4_to_3b_source",
        )

        # On part de la table correspondante
        if source_choice == "Top visibles (résumé)":
            # si tu as un DataFrame 'summary' utilisable, sinon on prend un top N du dfpred
            base_for_seeds = df_all.head(100).copy() if len(df_all) > 0 else df_all.copy()
        else:
            base_for_seeds = df_all.copy()

        # Paramètres de filtrage
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            top_n_seeds = st.number_input("Top-N to send", min_value=1, max_value=max(1, len(base_for_seeds)),
                                          value=min(50, max(1, len(base_for_seeds))), step=10, key="step4_to_3b_topn")
        with col2:
            has_proba = "Probabilité activité" in base_for_seeds.columns
            seed_thr = st.slider("Activity threshold min", 0.0, 1.0, 0.5 if has_proba else 0.0, 0.01, disabled=not has_proba,
                                 key="step4_to_3b_thr")
        with col3:
            only_pred_actives = st.checkbox("Keep only the actives", value=True, key="step4_to_3b_onlyact")
        with col4:
            replace_mode = st.checkbox("Replace existing seeds", value=False,
                                       help="Otherwise added to current seeds", key="step4_to_3b_replace")

        # Préparer la table triée/filtrée
        work = base_for_seeds.copy()
        # Normaliser le nom de colonne proba si besoin (au cas où)
        if "Probabilité activité" in work.columns:
            work = work.sort_values("Probabilité activité", ascending=False)
            work = work[work["Probabilité activité"] >= seed_thr]

        if only_pred_actives and "Prédiction" in work.columns:
            # accepte "🟩 Actif" ou "Actif"
            work = work[work["Prédiction"].astype(str).str.contains("Actif", case=False, na=False)]

        # Sélection Top-N
        work = work.head(int(top_n_seeds)).reset_index(drop=True)

        # Bouton d'envoi
        def _safe_smiles(x):
            s = str(x).strip()
            return s if s and Chem.MolFromSmiles(s) is not None else None

        if st.button(f"➕ Send {len(work)} molecule(s) as seeds to Step 3b", key=f"btn_to3b_{len(work)}"):
            if work.empty or "SMILES" not in work.columns:
                st.error("No SMILES column detected in the table selected.")
            else:
                # Construire la liste unique de SMILES valides
                smiles_list = []
                seen = set()
                for smi in work["SMILES"].tolist():
                    smi = _safe_smiles(smi)
                    if not smi or smi in seen:
                        continue
                    smiles_list.append(smi)
                    seen.add(smi)

                # Nom optionnel (au cas où Step 3b affiche les graines)
                names_list = work["Nom de la molécule"].astype(
                    str).tolist() if "Nom de la molécule" in work.columns else []

                if replace_mode:
                    st.session_state["seeds_in"] = smiles_list
                    st.session_state["seeds_in_names"] = names_list[:len(smiles_list)]
                else:
                    st.session_state.setdefault("seeds_in", [])
                    st.session_state.setdefault("seeds_in_names", [])
                    existing = set(st.session_state["seeds_in"])
                    added = 0
                    for i, smi in enumerate(smiles_list):
                        if smi in existing:
                            continue
                        st.session_state["seeds_in"].append(smi)
                        nm = names_list[i] if i < len(names_list) else f"Seed_{len(st.session_state['seeds_in'])}"
                        st.session_state["seeds_in_names"].append(nm)
                        existing.add(smi)
                        added += 1
                    st.info(f"{added} new seed(s) added.")

                st.success(f"✅ {len(smiles_list)} seeds ready for Step 3b.")
                st.caption(
                    "Tip : go to Step 3b and start a generation with bigger population size / more generations if needed.")

        st.session_state["prediction_results"] = dfpred
        prob_col = "Probabilit\u00e9 activit\u00e9"
        prob_threshold = None
        if prob_col in dfpred.columns:
            default_thr = st.session_state.get("step4_admet_prob_threshold", 0.5)
            try:
                default_thr = float(default_thr)
            except (TypeError, ValueError):
                default_thr = 0.5
            default_thr = max(0.0, min(1.0, default_thr))
            prob_threshold = st.slider(
                "Minimum activity threshold to send to 4.5 (ADMET/Tox)",
                min_value=0.0,
                max_value=1.0,
                value=default_thr,
                step=0.01,
                help="Molecules below this threshold will not be assessed.",
                key="step4_admet_threshold",
            )
            st.session_state["step4_admet_prob_threshold"] = prob_threshold
        else:
            st.session_state["step4_admet_prob_threshold"] = None
        prob_mask = pd.Series(True, index=dfpred.index)
        if prob_threshold is not None and prob_col in dfpred.columns:
            prob_values = pd.to_numeric(dfpred[prob_col], errors="coerce")
            prob_mask = prob_values > float(prob_threshold)
        act_mask = pd.Series(True, index=dfpred.index)
        if "Pr?diction" in dfpred.columns:
            act_mask = dfpred["Pr?diction"].astype(str).str.contains("Actif", case=False, na=False)
        selected_for_admet = dfpred[prob_mask & act_mask].reset_index(drop=True)
        st.session_state["prediction_results_for_admet"] = selected_for_admet
        if prob_threshold is not None:
            st.caption(f"{len(selected_for_admet)} molecule(s) kept for Step 4.5 (proba > {prob_threshold:.2f}).")
        else:
            st.caption(f"{len(selected_for_admet)} molecule(s) kept for Step 4.5.")


    # ===== Améliorations d'affichage Step 4 =====

    dfpred_display = df_filtered.copy() if "df_filtered" in locals() else dfpred.copy()

    # Tri par probabilité d'activité décroissante si présent
    if "Probabilité activité" in dfpred_display.columns:
        dfpred_display = dfpred_display.sort_values("Probabilité activité", ascending=False).reset_index(drop=True)

    # Vue compact / étendue
    with st.expander("👁️ Display option (Step 4)", expanded=False):
        compact4 = st.checkbox("Compact display", value=True)
        show_ic50 = st.checkbox("Display IC50 column (if avail.)", value=False)

    # Colonnes d'affichage selon la vue
    base_cols = [c for c in ["Nom de la molécule", "SMILES", "Probabilité activité", "AD label", "AD score", "ad_label", "ad_score"] if c in dfpred_display.columns]
    extra_cols = [c for c in ["IC50 (nM)", "Score", "Target", "Notes"] if c in dfpred_display.columns]
    if compact4:
        display_cols_4 = base_cols + ([] if not show_ic50 else [c for c in extra_cols if "IC50" in c])
    else:
        display_cols_4 = base_cols + extra_cols + [c for c in dfpred_display.columns if c not in base_cols + extra_cols]

    # Déduplication & présence
    _seen = set()
    display_cols_4 = [c for c in display_cols_4 if (c in dfpred_display.columns and not (c in _seen or _seen.add(c)))]

    df_show4 = dfpred_display[display_cols_4].copy()

    # Tronquer les SMILES pour lisibilité
    if "SMILES" in df_show4.columns:
        def _truncate(s, n=64):
            return (s[:n-1] + "…") if isinstance(s, str) and len(s) > n else s
        df_show4["SMILES"] = df_show4["SMILES"].map(lambda s: _truncate(s, 64))

    # Column config
    colcfg4 = {}
    if "Probabilité activité" in df_show4.columns:
        colcfg4["Probabilité activité"] = st.column_config.ProgressColumn(
            "Probabilité activité", format="%.0f%%", min_value=0, max_value=1
        )
    if "AD score" in df_show4.columns:
        colcfg4["AD score"] = st.column_config.ProgressColumn(
            "AD score", min_value=0.0, max_value=1.0, format="%.3f"
        )
    if "ad_score" in df_show4.columns and "AD score" not in df_show4.columns:
        colcfg4["ad_score"] = st.column_config.ProgressColumn(
            "AD score", min_value=0.0, max_value=1.0, format="%.3f"
        )

    st.dataframe(df_show4, use_container_width=True, hide_index=True, column_config=colcfg4)

    # Exports
    csv_all4 = dfpred_display.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV Export (Step 4 — all)", csv_all4, file_name="step4_predictions_all.csv", mime="text/csv")

    df_for_admet = st.session_state.get("prediction_results_for_admet")
    if isinstance(df_for_admet, type(dfpred_display)) and len(df_for_admet):
        csv_for_admet = df_for_admet.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV Export (ready for ADMET/Tox)", csv_for_admet, file_name="step4_predictions_for_admet.csv", mime="text/csv")

    else:
        st.info("No molecule selected in Step 3. Go back to step 3 to identify some.")

import streamlit as st
from Genmap_modules.status_manager import add_warning
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
# Silence RDKit error messages
RDLogger.DisableLog('rdApp.error')
import io, gzip

# ================= Optional 3D rendering (no hard dependency) =================
try:
    import py3Dmol
    _HAS_3D = True
    def _showmol(view, height=320, width=320):
        html = view._make_html()
        st.components.v1.html(html, height=height, width=width)
except Exception:
    _HAS_3D = False

# ================= Optional ChEMBL client (fallback to REST) =================
try:
    from chembl_webresource_client.new_client import new_client as _chembl_client
except Exception:
    _chembl_client = None

# --------------------------- Helpers ---------------------------

def _pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_columns(src_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise les colonnes vers: Name, SMILES, IC50(nM), ChEMBL_ID quand dispo.
       - supprime BOM, espaces insécables
       - gère les alias 'isomeric smiles' (avec espace), 'canonicalsmiles' (sans espace)
       - déduit une colonne SMILES si DF a une seule colonne sans header utile
    """
    if src_df is None or src_df.empty:
        return pd.DataFrame(columns=["Name", "SMILES", "IC50(nM)"])

    # --- Nettoyage des entêtes ---
    raw_cols = list(src_df.columns)
    colmap = {}
    for c in raw_cols:
        k = str(c).replace("\ufeff", "")            # BOM
        k = k.strip().lower().replace("\u00a0", " ") # espace insécable
        k = k.replace("\t", " ")
        k = " ".join(k.split())                      # collapse espaces
        colmap[c] = k
    df = src_df.rename(columns=colmap).copy()

    # --- Aliases étendus ---
    smiles_aliases = {
        "smiles", "canonical smiles", "canonicalsmiles",
        "isomericsmiles", "isomeric_smiles", "isomeric smiles",
        "smiles string", "smiles_string",
        "structure", "structure_smiles", "mol_smiles"
    }
    name_aliases = {
        "name","molname","molecule","compoundname","compound name","id","compound id",
        "molecule_name","molecule name","molecule id",
        "pref_name","pref name","nom"
    }
    chembl_aliases = {
        "chembl_id","chembl id","molecule_chembl_id",
    }
    ic50_aliases = {
        "ic50","ic50(nm)","standard_value","standard value","ic50_nm","ic50 (nm)"
    }

    def _pick_alias(dfcols, aliases):
        for a in aliases:
            if a in dfcols:
                return a
        return None

    smi_col = _pick_alias(set(df.columns), smiles_aliases)
    nm_col  = _pick_alias(set(df.columns), name_aliases)
    ic_col  = _pick_alias(set(df.columns), ic50_aliases)
    chembl_col = _pick_alias(set(df.columns), chembl_aliases)

    # --- Cas '1 seule colonne' → deviner SMILES ---
    if smi_col is None and df.shape[1] == 1:
        only_col = df.columns[0]
        # Heuristique : >60% des 200 premières lignes contiennent majoritairement les caractères SMILES
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=#()[]+-@\\/.,:*")
        sample = df[only_col].astype(str).head(200)
        def looks_like_smiles(s: str) -> bool:
            s = s.strip()
            if not s or len(s) < 4:  # trop court
                return False
            bad = sum(ch not in allowed for ch in s)
            return (bad / max(1, len(s))) < 0.1
        ratio = (sample.apply(looks_like_smiles).sum()) / max(1, len(sample))
        if ratio >= 0.6:
            smi_col = only_col

    # --- Construction normalisée ---
    if smi_col is None:
        return pd.DataFrame(columns=["Name", "SMILES", "IC50(nM)"])

    out = pd.DataFrame()
    out["SMILES"] = df[smi_col].astype(str).map(lambda s: s.replace("\ufeff","").strip()).replace({"": None})
    if nm_col is not None:
        out["Name"] = df[nm_col].astype(str).map(lambda s: s.strip()).replace({"": None})
    else:
        out["Name"] = [f"Mol_{i+1:03d}" for i in range(len(df))]
    if ic_col is not None:
        out["IC50(nM)"] = pd.to_numeric(df[ic_col], errors="coerce")
    if chembl_col is not None:
        out["ChEMBL_ID"] = df[chembl_col].astype(str).map(lambda s: s.strip()).replace({"": None})

    out = out.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    return out
# --------------------------- Name → SMILES ---------------------------

from rdkit import Chem
import requests

def get_smiles_from_name_or_smiles(text: str | None):
    """Resolve a text entry (SMILES, ChEMBL ID, or drug name) into a SMILES and optional ChEMBL ID.
    
    Returns
    -------
    tuple[str|None, str|None]
        (smiles, chembl_id) or (None, None) if not resolved.
    """
    s = (text or "").strip()
    if not s:
        return None, None

    # Already a SMILES?
    try:
        if Chem.MolFromSmiles(s) is not None:
            return s, None
    except Exception:
        pass

    # ChEMBL: search by name or ID
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json",
            params={"q": s, "limit": 5}, timeout=15
        )
        if r.ok:
            data = r.json()
            for m in data.get("molecules", []):
                ms = m.get("molecule_structures") or {}
                smi = ms.get("canonical_smiles")
                chembl_id = m.get("molecule_chembl_id")
                if smi:
                    return smi, chembl_id
    except Exception as e:
        print(f"[WARN] ChEMBL lookup failed for {s}: {e}")

    # PubChem: PUG REST by name
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{s}/property/IsomericSMILES/JSON",
            timeout=15,
        )
        if r.ok:
            js = r.json()
            props = js.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES"), None
    except Exception as e:
        print(f"[WARN] PubChem lookup failed for {s}: {e}")

    return None, None

def get_smiles_from_name_or_smiles_old(text: str | None):
    s = (text or "").strip()
    if not s:
        return None, None
    # Already a SMILES?
    if Chem.MolFromSmiles(s) is not None:
        return s, None
    # ChEMBL search by name or ID
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json",
            params={"q": s, "limit": 5}, timeout=15
        )
        if r.ok:
            data = r.json()
            for m in data.get("molecules", []):
                ms = m.get("molecule_structures") or {}
                smi = ms.get("canonical_smiles")
                chembl_id = m.get("molecule_chembl_id")
                if smi:
                    return smi, chembl_id
    except Exception:
        pass
    # PubChem PUG REST (by name)
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{s}/property/IsomericSMILES/JSON",
            timeout=15,
        )
        if r.ok:
            js = r.json()
            props = js.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES"), None
    except Exception:
        pass
    return None, None


# --------------------------- ChEMBL activities ---------------------------

def _get_chembl_activities_rest(chembl_target_id: str, limit_total=800) -> pd.DataFrame:
    """Fetch activities via REST and keep IC50 in nM with relation '=' when possible; return Name/SMILES/IC50(nM)."""
    BASE = "https://www.ebi.ac.uk"
    url = f"{BASE}/chembl/api/data/activity.json?limit=200&offset=0&target_chembl_id={chembl_target_id}&standard_type=IC50"
    rows, seen = [], set()
    while url and len(rows) < limit_total:
        if url.startswith('/'):
            url = BASE + url
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            break
        data = r.json()
        for a in data.get("activities", []):
            smi = a.get("canonical_smiles") or (a.get("molecule_structures") or {}).get("canonical_smiles")
            if not smi:
                continue
            units = (a.get("standard_units") or "").lower()
            rel = a.get("standard_relation") or "="
            try:
                ic50 = float(a.get("standard_value")) if a.get("standard_value") is not None else None
            except Exception:
                ic50 = None
            # Keep only nM and '=' when available
            if ic50 is None or (units and units != "nm") or (rel and rel != "="):
                continue
            mol_id = a.get("molecule_chembl_id") or (a.get("molecule") or {}).get("molecule_chembl_id")
            mol_name = (a.get("molecule") or {}).get("pref_name") or a.get("molecule_pref_name") or mol_id or "?"
            key = (smi, ic50)
            if key not in seen:
                rows.append({"Name": str(mol_name), "SMILES": str(smi), "IC50(nM)": ic50, "ChEMBL_ID": mol_id})
                seen.add(key)
        url = data.get("page_meta", {}).get("next")
    df = pd.DataFrame(rows)
    if not df.empty:
        # Deduplicate by SMILES: keep best potency (min IC50)
        df = df.sort_values("IC50(nM)").drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    return df


def _get_chembl_activities_client(chembl_target_id: str, limit=800) -> pd.DataFrame:
    if _chembl_client is None:
        return _get_chembl_activities_rest(chembl_target_id, limit_total=limit)
    rows = []
    act = _chembl_client.activity.filter(
        target_chembl_id=chembl_target_id,
        standard_type="IC50",
        standard_relation="=",
        standard_units="nM",
        limit=limit,
    )
    for entry in act:
        smi = entry.get("canonical_smiles")
        if not smi:
            continue
        try:
            ic50 = float(entry.get("standard_value")) if entry.get("standard_value") is not None else None
        except Exception:
            ic50 = None
        if ic50 is None:
            continue
        mol_id = entry.get("molecule_chembl_id")
        mol_name = entry.get("molecule_name") or mol_id or "?"
        rows.append({"Name": str(mol_name), "SMILES": str(smi), "IC50(nM)": ic50, "ChEMBL_ID": mol_id})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("IC50(nM)").drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    return df


# --------------------------- UI Logic ---------------------------

def _on_add_manual():
    ref = (st.session_state.get("manual_name_input") or "").strip()
    text = (st.session_state.get("manual_smiles_input") or "").strip()
    smi, chembl_id = get_smiles_from_name_or_smiles(text)
    if smi is None:
        st.session_state["manual_add_error"] = "Entrée invalide : ni SMILES valide ni nom reconnu."
        return
    st.session_state.setdefault("manual_entries", [])
    name = ref if ref else text
    st.session_state["manual_entries"].append({"Name": name, "SMILES": smi, "ChEMBL_ID": chembl_id})
    st.session_state["manual_add_error"] = ""
    st.session_state["manual_name_input"] = ""
    st.session_state["manual_smiles_input"] = ""

def _render_3d(smiles, width=300, height=300):
    if not _HAS_3D:
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    mb = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(mb, "mol")
    view.setStyle({"stick": {}})
    view.setBackgroundColor("white")
    view.zoomTo()
    _showmol(view, height=height, width=width)

def _read_uploaded_table(up) -> pd.DataFrame:
    """
    Lecture robuste d'un fichier tabulaire provenant de st.file_uploader.
    - Gère .csv, .tsv, .txt, .gz
    - Détecte le séparateur (sep=None + engine='python')
    - Essaie UTF-8 puis latin-1
    - Saute les lignes illisibles (on_bad_lines='skip')
    """
    name = (getattr(up, "name", "") or "").lower()

    # Récupère les octets (et réinitialise un buffer indépendant)
    raw = up.read() if hasattr(up, "read") else bytes(up)
    bio = io.BytesIO(raw)

    # Décompression .gz si besoin
    if name.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=bio, mode="rb") as gz:
                raw = gz.read()
            bio = io.BytesIO(raw)
        except Exception:
            # si pas du vrai .gz, on retombe sur le raw
            bio = io.BytesIO(raw)

    # Essais de parsing — ordre du plus permissif au plus strict
    trials = [
        dict(sep=None, engine="python", encoding="utf-8", on_bad_lines="skip"),
        dict(sep=None, engine="python", encoding="latin-1", on_bad_lines="skip"),
        dict(sep=",",   engine="python", encoding="utf-8", on_bad_lines="skip"),
        dict(sep="\t",  engine="python", encoding="utf-8", on_bad_lines="skip"),
        dict(sep=";",   engine="python", encoding="utf-8", on_bad_lines="skip"),
    ]

    last_err = None
    for kw in trials:
        try:
            bio.seek(0)
            df = pd.read_csv(bio, **kw)
            # Cas piégeux : une seule colonne mais séparateurs présents → réessaie en TSV
            if df.shape[1] == 1:
                col0 = df.columns[0]
                sample = "\n".join(map(str, df[col0].head(5).tolist()))
                if "\t" in sample:
                    bio.seek(0)
                    df = pd.read_csv(bio, sep="\t", engine="python", encoding=kw.get("encoding","utf-8"), on_bad_lines="skip")
            df.columns = [str(c).replace("\ufeff", "") for c in df.columns]
            return df
        except Exception as e:
            last_err = e

    # Si tout a échoué
    raise RuntimeError(f"Impossible de lire le fichier : {last_err}")

def run():
    st.header("3. Sélection des molécules")

    # ------- 1) Saisie manuelle -------
    st.session_state.setdefault("manual_entries", [])
    with st.expander("➕ Manually enter a molecule", expanded=True):
        st.text_input("Reference (Optional)", key="manual_name_input", help="Ex: BN80915, Molecule_7, MyCompound_0011")
        st.text_input("SMILES or Name", key="manual_smiles_input", help="Ex: CC(=O)Oc1ccccc1C(=O)O or aspirin")
        st.button("Add to list", type="primary", on_click=_on_add_manual)
        if st.session_state.get("manual_add_error"):
            st.error(st.session_state["manual_add_error"])

    # ------- 1bis) Pool coming from Step 3b (in-silico design) -------
    raw_insilico = st.session_state.get("insilico_entries") or []
    if isinstance(raw_insilico, dict):
        raw_insilico = [raw_insilico]
    insilico_df = pd.DataFrame(raw_insilico) if raw_insilico else pd.DataFrame(columns=["Name", "SMILES"])
    if not insilico_df.empty:
        insilico_df = insilico_df.dropna(subset=["SMILES"])
        insilico_df["SMILES"] = insilico_df["SMILES"].astype(str).str.strip()
        insilico_df = insilico_df[insilico_df["SMILES"] != ""]
        insilico_df = insilico_df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
        if "Name" not in insilico_df.columns:
            insilico_df["Name"] = [f"Design_{i+1:03d}" for i in range(len(insilico_df))]
        else:
            mask = insilico_df["Name"].astype(str).str.strip().eq("")
            if mask.any():
                repl = [f"Design_{i+1:03d}" for i in range(mask.sum())]
                insilico_df.loc[mask, "Name"] = repl
        default_source = "In silico design"
        if "Source" not in insilico_df.columns:
            insilico_df["Source"] = default_source
        else:
            insilico_df["Source"] = insilico_df["Source"].fillna(default_source).replace({"": default_source})
    nb_insilico_ok = len(insilico_df)

    with st.expander("✨ Molecules coming from Step 3b (optional)", expanded=False):
        if nb_insilico_ok:
            st.caption(f"{nb_insilico_ok} candidate(s) received from Step 3b (VAE / in-silico design).")
            st.dataframe(insilico_df, use_container_width=True)
            if st.button("Clear Step 3b pool", key="step3_clear_insilico"):
                st.session_state["insilico_entries"] = []
                st.success("In-silico pool cleared. Re-run Step 3b to add new molecules.")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        else:
            st.info("No molecule from Step 3b yet. Go to “Elevage in-silico” to design new ones.")

    # ------- 2) Import ChEMBL & CSV -------
    chembl_df = None
    csv_df = None

    target_id = st.session_state.get("selected_target_chembl_id")
    auto_load = st.checkbox("Automatic download from ChemBL (if a target is selected)", value=True)
    if target_id and auto_load and "chembl_df_cached" not in st.session_state:
        try:
            chembl_df = _get_chembl_activities_client(target_id, limit=800)
        except Exception:
            chembl_df = _get_chembl_activities_rest(target_id, limit_total=800)
        st.session_state["chembl_df_cached"] = chembl_df
    else:
        chembl_df = st.session_state.get("chembl_df_cached")

    with st.expander("📡 ChemBL import (manual)", expanded=not auto_load):
        if target_id:
            st.info(f"ChemBL target: **{target_id}**")
            if st.button("Reimport from ChemBL"):
                try:
                    chembl_df = _get_chembl_activities_client(target_id, limit=800)
                except Exception:
                    chembl_df = _get_chembl_activities_rest(target_id, limit_total=800)
                st.session_state["chembl_df_cached"] = chembl_df
                st.success(f"{len(chembl_df)} records found.")
        else:
            st.info("No target selected in step 2.")

    with st.expander("📥 Import a CSV (optional)", expanded=False):
        up = st.file_uploader("CSV/TSV with columns: SMILES (compulsory), Name (optional)",
                              type=["csv", "tsv", "txt", "gz"])
        if up is not None:
            try:
                csv_df = _read_uploaded_table(up)
                st.success(f"File imported: {len(csv_df)} lines.")
                # Affiche un aperçu
                st.dataframe(csv_df.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"Cannot read the file: {e}")
                st.info("Tip: check the separator (tab/comma/semicolon), "
                        "the encoding, or export CSV from Excel - standard (UTF-8).")

    # ------- 3) Fusion des sources & nettoyage -------
    parts = []

    if isinstance(chembl_df, pd.DataFrame) and not chembl_df.empty:
        chembl_norm = _normalize_columns(chembl_df)
        if not chembl_norm.empty:
            chembl_norm = chembl_norm.copy()
            chembl_norm["Source"] = "ChEMBL"
            parts.append(chembl_norm)

    manual_df = pd.DataFrame(st.session_state["manual_entries"]) if st.session_state["manual_entries"] else pd.DataFrame(columns=["Name","SMILES","ChEMBL_ID"])
    if not manual_df.empty:
        manual_norm = _normalize_columns(manual_df)
        if not manual_norm.empty:
            manual_norm = manual_norm.copy()
            manual_norm["Source"] = "Manual"
            parts.append(manual_norm)

    if isinstance(csv_df, pd.DataFrame) and not csv_df.empty:
        csv_norm = _normalize_columns(csv_df)
        if not csv_norm.empty:
            csv_norm = csv_norm.copy()
            csv_norm["Source"] = "CSV import"
            parts.append(csv_norm)

    if not insilico_df.empty:
        design_part = insilico_df.copy()
        design_part["Source"] = design_part["Source"].fillna("In silico design")
        parts.append(design_part)

    if not parts:
        add_warning("No molecule available. Add some manually, import a CSV or import from ChEMBL.")
        st.info("No molecules currently available. Use the controls above to add entries.")
        return

    base_df = pd.concat(parts, ignore_index=True)
    base_df = base_df.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    base_df["Name"] = base_df["Name"].fillna("")
    mask = base_df["Name"].str.strip().eq("")
    if mask.any():
        seq = [f"Mol_{i+1:03d}" for i in range(mask.sum())]
        base_df.loc[mask, "Name"] = seq
    if "Source" not in base_df.columns:
        base_df["Source"] = ""
    else:
        base_df["Source"] = base_df["Source"].fillna("")

    st.subheader("Verification of sources for the prediction.")
    nb_chembl_ok = 0
    if isinstance(chembl_df, pd.DataFrame) and not chembl_df.empty:
        nb_chembl_ok = len(_normalize_columns(chembl_df))

    nb_manual_ok = len(_normalize_columns(pd.DataFrame(st.session_state["manual_entries"]))) if st.session_state.get(
        "manual_entries") else 0
    nb_csv_ok = len(_normalize_columns(csv_df)) if isinstance(csv_df, pd.DataFrame) and not csv_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Valid SMILES (ChEMBL)", nb_chembl_ok)
    with c2:
        st.metric("Valid SMILES (Manual)", nb_manual_ok)
    with c3:
        st.metric("Valid SMILES (CSV)", nb_csv_ok)
    with c4:
        st.metric("Valid SMILES (In silico)", nb_insilico_ok)

    # ------- 4) Récapitulatif des sources -------
    st.subheader("Summary of the sources")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        nb_chembl = len(chembl_df) if isinstance(chembl_df, pd.DataFrame) else 0
        st.metric("ChemBL molecules (IC50)", nb_chembl)
    with colB:
        st.metric("Molecules from manual entry", len(manual_df))
    with colC:
        nb_csv = len(csv_df) if isinstance(csv_df, pd.DataFrame) else 0
        st.metric("Imported molecules (CSV)", nb_csv)
    with colD:
        st.metric("In silico design (Step 3b)", nb_insilico_ok)

    with st.expander("🧪 ChEMBL Molecules known to be active (after filtering IC50 nM =)", expanded=False):
        if isinstance(chembl_df, pd.DataFrame) and not chembl_df.empty:
            prev = _normalize_columns(chembl_df)
            if "IC50(nM)" in prev.columns:
                prev = prev.sort_values("IC50(nM)", ascending=True, na_position="last")
            st.dataframe(prev, use_container_width=True)
        else:
            st.info("No ChemBL data loaded.")

    with st.expander("✍️ Molecules added by the user", expanded=False):
        if not manual_df.empty:
            st.dataframe(manual_df, use_container_width=True)
        else:
            st.info("No molecule entered.")

    if isinstance(csv_df, pd.DataFrame) and not csv_df.empty:
        with st.expander("📄 Molecules imported (CSV)", expanded=False):
            st.dataframe(_normalize_columns(csv_df), use_container_width=True)

    # ------- 5) Sélection finale (un seul tableau) -------
    st.subheader("Final selection of the molecules to predict (ChEMBL + Manual + CSV + Step 3b)")
    st.caption("Untick to exclude. Manual entries are included by default.")

    # Base table (ordered, without Include) used to synchronise the editor content
    base_sorted = base_df.copy()
    if "IC50(nM)" in base_sorted.columns:
        base_sorted = base_sorted.sort_values(by="IC50(nM)", ascending=True, na_position="last")
    base_sorted = base_sorted.reset_index(drop=True)

    editor_key = "step3_editor"       # widget key
    df_state_key = "step3_editor_df"  # DataFrame actually edited by the widget

    # Initialise or resynchronise the editor DF based on the current base_sorted
    if df_state_key not in st.session_state:
        df_state = base_sorted.copy()
        df_state.insert(0, "Inclure", True)
        st.session_state[df_state_key] = df_state
    else:
        df_prev = st.session_state[df_state_key]
        try:
            # Map previous inclusion choices by SMILES
            prev_inc = {
                str(row.SMILES): bool(row.Inclure)
                for row in df_prev.itertuples()
                if hasattr(row, "Inclure")
            }
        except Exception:
            prev_inc = {}

        df_state = base_sorted.copy()
        smiles_series = df_state["SMILES"].astype(str)
        incl_col = smiles_series.map(prev_inc)
        # New molecules default to True, others keep their previous choice
        df_state.insert(0, "Inclure", incl_col.fillna(True))
        st.session_state[df_state_key] = df_state

    # Bulk actions acting directly on the stored DF
    bulk_col1, bulk_col2, _ = st.columns([1, 1, 6])
    with bulk_col1:
        if st.button("Select all", key="step3_include_select_all", help="Include all molecules for prediction"):
            st.session_state[df_state_key]["Inclure"] = True
    with bulk_col2:
        if st.button("Deselect all", key="step3_include_deselect_all",
                     help="Exclude all molecules (you can then re-include specific ones)"):
            st.session_state[df_state_key]["Inclure"] = False

    edited = st.data_editor(
        st.session_state[df_state_key],
        use_container_width=True,
        column_config={
            "Inclure": st.column_config.CheckboxColumn("Include", help="Include this molecule for the prediction"),
            "Name": st.column_config.TextColumn("Name"),
            "SMILES": st.column_config.TextColumn("SMILES"),
            "IC50(nM)": st.column_config.NumberColumn("IC50 (nM)", format="%.2f"),
            "ChEMBL_ID": st.column_config.TextColumn("ChEMBL_ID"),
            "Source": st.column_config.TextColumn("Source", help="Origin of the molecule", disabled=True),
        },
        num_rows="fixed",
        key=editor_key,
    )

    # Persist edits from the widget back into our DF, then compute final selection
    st.session_state[df_state_key] = edited

    df_final = edited[edited["Inclure"] == True].copy()
    total, kept = len(edited), len(df_final)
    st.success(f"{kept} / {total} molecules selected for step 4.")

    if df_final.empty:
        add_warning("No molecule selected.")
        st.info("Select at least one molecule to continue.")
        return

    # ------- 6) Passage aux étapes suivantes -------
    st.session_state["to_predict"] = df_final["SMILES"].astype(str).tolist()
    st.session_state["to_predict_names"] = df_final["Name"].astype(str).tolist()
    if "Source" in df_final.columns:
        st.session_state["to_predict_sources"] = df_final["Source"].astype(str).tolist()
    else:
        st.session_state["to_predict_sources"] = [None] * len(df_final)
    if "ChEMBL_ID" in df_final.columns:
        st.session_state["to_predict_ids"] = df_final["ChEMBL_ID"].where(df_final["ChEMBL_ID"].notnull(), None).tolist()
    else:
        st.session_state["to_predict_ids"] = [None] * len(df_final)

    # Très important : fournir un DF d'entraînement propre au Step 4
    # (uniquement les molécules ChEMBL avec IC50, si dispo)
    if isinstance(chembl_df, pd.DataFrame) and not chembl_df.empty:
        train_df = _normalize_columns(chembl_df)
        train_df = train_df.dropna(subset=["SMILES", "IC50(nM)"])
        train_df = train_df.sort_values("IC50(nM)").drop_duplicates(subset=["SMILES"], keep="first")
        st.session_state["chembl_df_for_training"] = train_df.reset_index(drop=True)

    # ------- 7) Aperçu 2D/3D (sélection finale uniquement) -------
    with st.expander("🔬 Visual display of selected molecules (2D/3D)", expanded=False):
        from rdkit.Chem import Draw
        N = min(24, len(df_final))
        for idx, row in df_final.head(N).iterrows():
            smi = row["SMILES"]
            name = row.get("Name", f"Mol_{idx+1:03d}")
            c1, c2 = st.columns(2)
            with c1:
                try:
                    mol2d = Chem.MolFromSmiles(smi)
                    if mol2d is not None:
                        st.image(Draw.MolToImage(mol2d, size=(300, 300)), caption=f"2D - {name}")
                except Exception:
                    pass
            with c2:
                try:
                    _render_3d(smi, width=300, height=300)
                except Exception:
                    st.info("3D unavailable for this molecule.")

    st.info("Go to Step 4 to train the AI model and run the predictions.")

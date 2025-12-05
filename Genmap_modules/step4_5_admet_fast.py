
# -*- coding: utf-8 -*-
"""
Step 4.5 — ADMET/Tox (fast + richer proxies)
Fixed: RDKit FilterCatalogParams import/usage for broader compatibility.
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
from Genmap_modules.status_manager import add_warning

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import Crippen  # for RDKit logP fallback
from rdkit.Chem import FilterCatalog  # important: use nested FilterCatalogParams
from rdkit.Chem import Draw
import base64
from io import BytesIO

from joblib import Parallel, delayed
import requests

# Your existing ADMET model API (must be available in your project)
# predict_admet(smi) -> object with attributes .logP and .tpsa
try:
    from modules.admet import predict_admet  # type: ignore
except Exception as _e:
    predict_admet = None  # will be guarded at runtime


# -------------------------
# Desirability functions
# -------------------------

def _d_gauss(x, mu, sigma, lo=None, hi=None):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return np.nan
    x = float(x)
    if lo is not None and x < lo:
        return math.exp(-((lo - x) / sigma) ** 2)
    if hi is not None and x > hi:
        return math.exp(-((x - hi) / sigma) ** 2)
    return math.exp(-((x - mu) / sigma) ** 2)

def _desir_logp(logp):        # target ~2, tol ~1.2
    return _d_gauss(logp, 2.0, 1.2, lo=-1.0, hi=5.0)

def _desir_tpsa(tpsa):        # window 40–90
    return _d_gauss(tpsa, 65.0, 20.0, lo=20.0, hi=140.0)

def _desir_solubility(logp, tpsa):
    # proxy mixing logP (inverse) and tPSA (bonus)
    if logp is None or tpsa is None:
        return np.nan
    base = 1.0 / (1.0 + math.exp(float(logp) - 1.0))
    bonus = min(max(float(tpsa) / 120.0, 0.0), 1.0) * 0.25
    return float(min(max(base + bonus, 0.0), 1.0))

def _proxy_herg(logp, arom):
    if logp is None or arom is None:
        return np.nan
    return float(min(max((float(logp) / 6.0) + (float(arom) / 10.0), 0.0), 1.0))


# -------------------------
# Alerts catalogs (cached once)
# -------------------------

@st.cache_resource(show_spinner=False)
def _get_filter_catalog():
    """Create PAINS/Brenk FilterCatalog if available in this RDKit build.
    Falls back to None if not available.
    """
    try:
        params = FilterCatalog.FilterCatalogParams()
        # Some builds have PAINS_A/B/C; we start with A. Add Brenk if present.
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        # Brenk may not be present in very old builds; guard in try/except.
        try:
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        except Exception:
            pass
        return FilterCatalog.FilterCatalog(params)
    except Exception:
        return None


# -------------------------
# Fast batch ADMET core
# -------------------------

@st.cache_data(show_spinner=False)
def _batch_admet_fast(smiles_list: tuple[str, ...]) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    # 1) Parallel call to user's predict_admet when available
    def _one_pred(smi):
        try:
            res = predict_admet(smi)
            # Accept both dict and attribute-based APIs
            if isinstance(res, dict):
                lp = res.get("logP")
                if lp is None:
                    lp = res.get("logp")
                tp = res.get("tpsa")
                if tp is None:
                    tp = res.get("TPSA")
                return (lp, tp)
            lp = getattr(res, "logP", None)
            if lp is None:
                lp = getattr(res, "logp", None)
            tp = getattr(res, "tpsa", None)
            if tp is None:
                tp = getattr(res, "TPSA", None)
            return (lp, tp)
        except Exception:
            return (None, None)

    if predict_admet is not None:
        logp_tpsa = Parallel(n_jobs=-1, backend="threading")(delayed(_one_pred)(s) for s in smiles_list)
        logp = [lt[0] for lt in logp_tpsa]
        tpsa = [lt[1] for lt in logp_tpsa]
    else:
        # Robust RDKit-only fallback: ensure we still compute logP/tPSA
        logp = [Crippen.MolLogP(m) if m is not None else None for m in mols]
        tpsa = [rdmd.CalcTPSA(m) if m is not None else None for m in mols]

    # 2) RDKit descriptors (vector-ish, single pass on cached mols)
    arom = [rdmd.CalcNumAromaticRings(m) if m else None for m in mols]
    hbd  = [rdmd.CalcNumHBD(m) if m else None for m in mols]
    hba  = [rdmd.CalcNumHBA(m) if m else None for m in mols]
    rb   = [rdmd.CalcNumRotatableBonds(m) if m else None for m in mols]
    mw   = [rdmd.CalcExactMolWt(m) if m else None for m in mols]

    # 3) Alerts: PAINS/Brenk (safe fallback)
    catalog = _get_filter_catalog()
    def _alerts(m):
        if m is None or catalog is None:
            return []
        matches = []
        e = catalog.GetFirstMatch(m)
        while e is not None:
            try:
                matches.append(e.GetFilter().GetName())
            except Exception:
                break
            e = catalog.GetNextMatch(m, e)
        return matches

    alerts = [_alerts(m) for m in mols]
    pains_brenk = [1.0 if len(a) > 0 else 0.0 for a in alerts]

    # 4) Proxies & desirabilities
    herg = [_proxy_herg(lp, ar) if lp is not None and ar is not None else np.nan
            for lp, ar in zip(logp, arom)]
    sol  = [_desir_solubility(lp, ps) for lp, ps in zip(logp, tpsa)]
    d_logp = [_desir_logp(lp) for lp in logp]
    d_tpsa = [_desir_tpsa(ps) for ps in tpsa]

    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    has_nitro = [int(m.HasSubstructMatch(nitro)) if m else 0 for m in mols]
    ames_prob = [float(min(max(0.2 + 0.6 * (ni or pb), 0.0), 1.0))
                 for ni, pb in zip(has_nitro, pains_brenk)]

    # 5) Rules (for warnings)
    lip_fail = [int((lp is not None and lp > 5) or (mw_ is not None and mw_ > 500)
                    or (hbd_ is not None and hbd_ > 5) or (hba_ is not None and hba_ > 10))
                for lp, mw_, hbd_, hba_ in zip(logp, mw, hbd, hba)]
    veber_ok = [int((psa is not None and psa <= 140) and (rb_ is not None and rb_ <= 10))
                for psa, rb_ in zip(tpsa, rb)]

    return pd.DataFrame({
        "logp": logp,
        "tpsa": tpsa,
        "aromatic_rings": arom,
        "HBD": hbd, "HBA": hba, "RB": rb, "MW": mw,
        "solubility": sol,            # 0..1 desirability-like
        "herg_prob": herg,            # 0..1 risk proxy
        "ames_prob": ames_prob,       # 0..1 risk proxy
        "d_logp": d_logp,             # desirabilities for MPO
        "d_tpsa": d_tpsa,
        "pains_brenk": pains_brenk,
        "lipinski_fail": lip_fail,
        "veber_ok": veber_ok,
        "alerts": alerts,             # list[str]
    })


# -------------------------
# ChEMBL Max Phase helper
# -------------------------

@st.cache_data(show_spinner=False)
def _fetch_max_phase_for_ids(chembl_ids: tuple[str, ...]) -> dict[str, str]:
    """
    Fetch max_phase for a list of ChEMBL molecule IDs using the public REST API.
    Returns a mapping {chembl_id: phase_label}.
    """
    phase_map: dict[str, str] = {}
    ids = sorted({str(cid).strip() for cid in chembl_ids if cid})
    if not ids:
        return phase_map

    # ChEMBL convention: 0=Preclinical, 1=Phase I, 2=Phase II, 3=Phase III, 4=Approved
    phase_labels = {
        0: "Preclinical",
        1: "Phase I",
        2: "Phase II",
        3: "Phase III",
        4: "Approved",
    }

    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

    # Query in small chunks to avoid overlong URLs
    chunk_size = 50
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        try:
            params = {
                "molecule_chembl_id__in": ",".join(chunk),
                "limit": len(chunk),
            }
            resp = requests.get(base_url, params=params, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for entry in data.get("molecules", []):
                cid = str(entry.get("molecule_chembl_id") or "").strip()
                if not cid:
                    continue
                mp = entry.get("max_phase")
                # ChEMBL web UI treats missing/None max_phase as "Preclinical"
                if mp is None:
                    mp_int = 0
                else:
                    try:
                        mp_int = int(mp)
                    except Exception:
                        continue
                label = phase_labels.get(mp_int)
                if label is not None:
                    phase_map[cid] = label
        except Exception:
            # Fail silently: missing phases are simply left blank
            continue

    return phase_map


# -------------------------
# MPO (vectorized)
# -------------------------

def _mpo_np(act, d_logp, d_tpsa, sol, herg, ames, w):
    # benefits ↑ (act, d_logp, d_tpsa, sol), risks ↓ (herg, ames)
    # herg/ames transformed to (1 - risk)
    parts = [
        w["activity_prob"] * (0.0 if act is None or np.isnan(act) else float(act)),
        w["logp"]          * (0.0 if d_logp is None or np.isnan(d_logp) else float(d_logp)),
        w["tpsa"]          * (0.0 if d_tpsa is None or np.isnan(d_tpsa) else float(d_tpsa)),
        w["solubility"]    * (0.0 if sol   is None or np.isnan(sol)   else float(sol)),
        w["herg_prob"]     * (1.0 - (0.5 if herg is None or np.isnan(herg) else float(herg))),
        w["ames_prob"]     * (1.0 - (0.5 if ames is None or np.isnan(ames) else float(ames))),
    ]
    wsum = sum([w[k] for k in w])
    return float(min(max(sum(parts) / max(wsum, 1e-6), 0.0), 1.0))


def _plain_english_summary(row) -> str:
    """Return a short, clinician‑friendly summary for a single molecule."""
    msgs: list[str] = []

    def _risk_level(x):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return None
        try:
            xv = float(x)
        except Exception:
            return None
        if xv >= 0.7:
            return "high"
        if xv >= 0.4:
            return "moderate"
        return "low"

    # Cardio‑toxicity proxy (hERG)
    herg = row.get("herg_prob")
    h_level = _risk_level(herg)
    if h_level == "high":
        msgs.append("Potential cardio-toxicity: high hERG risk proxy.")
    elif h_level == "moderate":
        msgs.append("Possible cardio-toxicity signal: moderate hERG risk proxy.")

    # Genotoxicity proxy (Ames)
    ames = row.get("ames_prob")
    a_level = _risk_level(ames)
    if a_level == "high":
        msgs.append("Potential mutagenicity / genotoxicity: high Ames risk proxy.")
    elif a_level == "moderate":
        msgs.append("Possible mutagenicity signal: moderate Ames risk proxy.")

    # Solubility
    sol = row.get("solubility")
    try:
        sol_v = float(sol) if sol is not None else None
    except Exception:
        sol_v = None
    if sol_v is not None and not np.isnan(sol_v):
        if sol_v < 0.3:
            msgs.append("Low solubility proxy: may limit systemic exposure.")
        elif sol_v > 0.7:
            msgs.append("Solubility proxy favourable for oral exposure.")

    # MPO score (global ADMET desirability)
    mpo = row.get("MPO_score")
    try:
        mpo_v = float(mpo) if mpo is not None else None
    except Exception:
        mpo_v = None
    if mpo_v is not None and not np.isnan(mpo_v):
        if mpo_v >= 0.7:
            msgs.append("Overall ADMET/MPO profile globally favourable.")
        elif mpo_v >= 0.4:
            msgs.append("Overall ADMET/MPO profile intermediate; some trade-offs.")
        else:
            msgs.append("Overall ADMET/MPO profile suboptimal; consider optimisation.")

    # Rule-of-five / Veber / PAINS
    if bool(row.get("pains_brenk")):
        msgs.append("Contains structural alerts (PAINS/Brenk): possible assay artefacts or toxicity.")
    if bool(row.get("lipinski_fail")):
        msgs.append("Outside typical oral drug-like space (Lipinski rule-of-five).")
    veber_ok = row.get("veber_ok")
    if veber_ok is False:
        msgs.append("Veber rules not satisfied: oral bioavailability may be reduced.")

    if not msgs:
        return "No major ADMET/Tox concern identified with the current proxy scores."
    return "\n".join(msgs)


# -------------------------
# Warnings
# -------------------------

def _warn_row(pb, lp_fail, veber, herg, ames, alerts):
    flags = []
    if herg is not None and not np.isnan(herg) and herg > 0.5:
        flags.append("hERG")
    if ames is not None and not np.isnan(ames) and ames > 0.5:
        flags.append("Ames")
    if pb:
        flags.append("PAINS/Brenk")
    if lp_fail:
        flags.append("Lipinski")
    if not veber:
        flags.append("Veber")
    if alerts:
        flags.extend([f"ALERT:{a}" for a in list(alerts)[:2]])
    return "⚠️ " + ", ".join(flags) if flags else ""


# -------------------------
# SMILES → 2D image (HTML)
# -------------------------

def _smiles_to_img_tag(smiles: str, size: tuple[int, int] = (200, 200)) -> str:
    """Return an HTML <img> tag with a 2D RDKit depiction for the given SMILES."""
    if not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""
        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        # title attribute keeps the original SMILES accessible on hover
        return f'<img src="data:image/png;base64,{data}" width="{size[0]}" height="{size[1]}" title="{smiles}"/>'
    except Exception:
        return ""


# -------------------------
# Streamlit entry point
# -------------------------

def run():
    st.header("Step 4.5 — ADMET / Toxicology")

    # Pull predictions from session
    dfpred = st.session_state.get("prediction_results_for_admet")
    if not isinstance(dfpred, pd.DataFrame) or dfpred.empty:
        dfpred = st.session_state.get("prediction_results")

    if not isinstance(dfpred, pd.DataFrame) or dfpred.empty:
        st.info("No predictions available from Step 4.")
        return

    dfpred = dfpred.copy()
    prob_col = "Probabilit\u00e9 activit\u00e9"
    prob_threshold = st.session_state.get("step4_admet_prob_threshold")
    try:
        threshold_value = float(prob_threshold) if prob_threshold is not None else None
    except (TypeError, ValueError):
        threshold_value = None
    if threshold_value is not None and prob_col in dfpred.columns:
        proba_values = pd.to_numeric(dfpred[prob_col], errors="coerce")
        dfpred = dfpred[proba_values > threshold_value].reset_index(drop=True)
        if dfpred.empty:
            add_warning("No molecules exceed the probability threshold selected in Step 4.")
            st.info("Adjust the probability threshold in Step 4 or review the molecule list.")
            return

    # We coerce necessary columns
    if prob_col not in dfpred.columns:
        add_warning("Column 'Activity probability' is absent: MPO will be calculated without it (Update in step 4).")
        st.info("Add the activity probability column in Step 4 to include it in MPO.")

    # We allow compact vs extended display
    with st.expander("⚙️ Display options & score options", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            compact = st.checkbox("Compact display (hide advanced column)", value=True)
        with colB:
            st.caption("MPO score ponderation:")

        # Sliders for weights
        w_act  = st.slider("Activity weight", 0.0, 1.0, 0.30, 0.05)
        w_logp = st.slider("logP weight (desirability)", 0.0, 1.0, 0.15, 0.05)
        w_tpsa = st.slider("tPSA weight (desirability)", 0.0, 1.0, 0.15, 0.05)
        w_sol  = st.slider("Solubility weight (proxy)", 0.0, 1.0, 0.15, 0.05)
        w_herg = st.slider("hERG weight (risk↓)", 0.0, 1.0, 0.15, 0.05)
        w_ames = st.slider("Ames weight (risk↓)", 0.0, 1.0, 0.10, 0.05)

    smiles = tuple(dfpred["SMILES"].astype(str))
    with st.spinner("ADMET/Tox calculation..."):
        admet_df = _batch_admet_fast(smiles)

    work = pd.concat([dfpred.reset_index(drop=True), admet_df], axis=1)
    def _format_chembl_value(value):
        if value is None:
            return ""
        text_val = str(value).strip()
        if not text_val or text_val.lower() == "nan":
            return ""
        if text_val.startswith("http"):
            return text_val
        if text_val.upper().startswith("CHEMBL"):
            return f"https://www.ebi.ac.uk/chembl/compound_report_card/{text_val.upper()}/"
        return ""

    work["ChEMBL_URL"] = work.apply(
        lambda row: _format_chembl_value(row.get("ChEMBL_ID")) or _format_chembl_value(row.get("ChEMBL_URL")),
        axis=1,
    )

    # Add Max Phase (development stage) from ChEMBL when possible
    if "ChEMBL_ID" in work.columns:
        ids_for_phase = tuple(
            str(x).strip()
            for x in work["ChEMBL_ID"].tolist()
            if pd.notna(x) and str(x).strip()
        )
        phase_map = _fetch_max_phase_for_ids(ids_for_phase) if ids_for_phase else {}
        # Always create the column so it is available in the display options,
        # even if we could not retrieve any phase information.
        work["Max Phase"] = work["ChEMBL_ID"].astype(str).map(phase_map).fillna("")


    weights = {
        "activity_prob": w_act,
        "logp": w_logp, "tpsa": w_tpsa,
        "solubility": w_sol, "herg_prob": w_herg, "ames_prob": w_ames,
    }

    # Vectorized MPO
    act_col = work[prob_col] if prob_col in work.columns else [None] * len(work)
    work["MPO_score"] = [
        _mpo_np(a, dl, dt, s, h, am, weights)
        for a, dl, dt, s, h, am in zip(
            act_col,
            work["d_logp"], work["d_tpsa"], work["solubility"],
            work["herg_prob"], work["ames_prob"]
        )
    ]

    # Warnings
    work["Warnings"] = [
        _warn_row(pb, lp, vb, h, am, al)
        for pb, lp, vb, h, am, al in zip(
            work["pains_brenk"], work["lipinski_fail"], work["veber_ok"],
            work["herg_prob"], work["ames_prob"], work["alerts"]
        )
    ]

    # Plain‑English summary for clinicians / non‑specialists
    work["Summary"] = [_plain_english_summary(row) for _, row in work.iterrows()]

    # Save to session for later steps
    st.session_state["admet_results"] = work.copy()

    # Display configuration (HTML table with column selection)
    if compact:
        default_cols = [
            "Nom de la molécule", "ChEMBL_URL", "SMILES",
            "logp", "tpsa", "solubility", "herg_prob", "ames_prob",
            "MPO_score", "Warnings", "Summary",
        ]
    else:
        default_cols = [
            "Nom de la molécule", "ChEMBL_URL", "SMILES",
            "logp", "tpsa", "solubility", "herg_prob", "ames_prob",
            "MPO_score", "Warnings", "Summary",
            "MW", "HBD", "HBA", "RB", "aromatic_rings",
            "pains_brenk", "lipinski_fail", "veber_ok", "alerts",
        ]

    all_cols = list(work.columns)

    cols_state_key = "step45_display_columns"          # internal selection state
    cols_widget_key = "step45_display_columns_widget"  # widget key (separate to avoid Streamlit warnings)

    if cols_state_key not in st.session_state:
        st.session_state[cols_state_key] = [c for c in default_cols if c in all_cols]

    # Bulk controls to select / deselect all columns (update internal state only)
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("Select all columns", key="step45_select_all_cols"):
            st.session_state[cols_state_key] = all_cols.copy()
    with c2:
        if st.button("Deselect all columns", key="step45_deselect_all_cols"):
            st.session_state[cols_state_key] = []

    cols_to_show = st.multiselect(
        "Columns to display",
        options=all_cols,
        default=st.session_state[cols_state_key],
        key=cols_widget_key,
    )

    # Persist widget choice back into our internal state
    st.session_state[cols_state_key] = cols_to_show
    if not cols_to_show:
        st.info("Select at least one column to display.")
        return

    df_show = work[cols_to_show].copy()
    if "ChEMBL_URL" in df_show.columns:
        df_show["ChEMBL_URL"] = df_show["ChEMBL_URL"].replace({"": None})
        def _chembl_as_link(url: object) -> str:
            if url is None:
                return ""
            text_val = str(url).strip()
            if not text_val:
                return ""
            return f'<a href="{text_val}" target="_blank">ChEMBL</a>'
        df_show["ChEMBL_URL"] = df_show["ChEMBL_URL"].apply(_chembl_as_link)
    if "SMILES" in df_show.columns:
        df_show["SMILES"] = df_show["SMILES"].astype(str).apply(_smiles_to_img_tag)
        df_show = df_show.rename(columns={"SMILES": "2D molecule"})

    html_table = df_show.to_html(escape=False, index=False)
    html = """
<style>
.admet-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.90rem;
}
.admet-table th, .admet-table td {
    border: 1px solid #ddd;
    padding: 0.45rem 0.6rem;
    vertical-align: top;
    text-align: left;
}
.admet-table th {
    background-color: #f5f5f5;
    font-weight: 600;
}
.admet-table td {
    max-width: 480px;
    word-wrap: break-word;
    white-space: pre-line;
}
.admet-table tr:nth-child(even) {
    background-color: #fafafa;
}
.admet-table tr:hover {
    background-color: #f0f7ff;
}
</style>
<div class="admet-table">
""" + html_table + "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Export
    csv = work.to_csv(index=False).encode("utf-8")
    st.download_button("CSV Export (ADMET)", csv, file_name="admet_results.csv", mime="text/csv")

    st.caption("Note: the scores/proxies are heuristic to support decision making. They are not for regulatory filing.")





# Allow running as a script (for dev/testing)
if __name__ == "__main__":
    import sys
    print("ADMET/Tox ready. Use it via run() in Streamlit.")

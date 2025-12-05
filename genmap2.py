import streamlit as st
import json
from html import escape
import numpy as np
import certifi
import ssl
ssl_context = ssl.create_default_context(cafile=certifi.where())
import urllib.request
from Genmap_modules.status_manager import (
    add_warning,
    get_session_status,
    get_warnings,
    reset_warnings,
    set_session_status,
)
from auth import login
from modules.feedback import render_feedback_box
from modules.feedback_admin import render_admin_panel
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)


# ----- Fonctions utilitaires -----
def safe_str(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)

def sanitize_filename(s):
    return str(s).translate(str.maketrans({':': '_', ' ': '_', '/': '_', '\\': '_', '*': '_', '?': '_', '"': '_', '<': '_', '>': '_', '|': '_'}))

def get_smiles_from_name(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
        req = urllib.request.Request(url, verify=False)
        with urllib.request.urlopen(req, context=ssl_context, timeout=4, verify=False) as response:
            data = response.read()
            result = json.loads(data)
            return result["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except Exception as e:
        add_warning(f"Error during Pubchem search: {e}")
        return None

def clear_session():
    """Reset current research session state while preserving user login and preferences."""
    protected = {"auth_ok", "username", "roles", "last_activity", "dark_mode"}
    for k in list(st.session_state.keys()):
        if k in protected:
            continue
        try:
            del st.session_state[k]
        except Exception:
            pass
    reset_warnings()
    set_session_status("Idle")
    st.success("Session reinitialised (user login preserved).")


def apply_theme():
    """Inject a dark theme when the user enables it."""
    if st.session_state.get("dark_mode"):
        st.markdown(
            """
            <style>
            body, .stApp {background-color: #1e1e1e; color: #fafafa;}
            .stButton>button {background-color: #333333; color: #fafafa;}
            </style>
            """,
            unsafe_allow_html=True,
        )

STATUS_PANEL_STYLE = """
<style>
.sidebar-status-panel {
    font-size: 0.85rem;
    line-height: 1.25;
}
.sidebar-status-panel .status-title {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.sidebar-status-panel .status-line {
    margin-bottom: 0.15rem;
}
.sidebar-status-panel .status-subtitle {
    margin-top: 0.35rem;
    font-weight: 600;
}
.sidebar-status-panel ul {
    margin: 0.2rem 0 0;
    padding-left: 1rem;
}
.sidebar-status-panel li {
    margin-bottom: 0.15rem;
}
.sidebar-status-panel .status-muted {
    color: #6c6c6c;
    font-size: 0.78rem;
    margin-top: 0.3rem;
}
</style>
"""

# ----- UI Générale -----
st.set_page_config(page_title="Genmap©", layout="wide")
st.title("🧬 Genmap© - Genetic to Molecular Activity Predictor")
st.sidebar.title("Navigation")
if not login():
    st.stop()
menu = st.sidebar.radio("Go to step:", [
    "1. 1 Genetic entry",
    "2. 2 ChemBL target",
    "3. 3 Molecules",
    "4. 4 Activity Prediction",
    "6. 3b Molecules in-silico breeding",
    "7. 4b ADMET/Tox",
    "8. 4c Similarity Network",
    "5. 5 Report / Report"
])
set_session_status(menu)
st.sidebar.button("🔄 Reinitialize the session", on_click=clear_session)
dark_mode = st.sidebar.toggle("🌙 Dark mode", key="dark_mode")
status_panel = st.sidebar.container()
apply_theme()

# Step 4: IC50 activity threshold slider (placed above feedback panel)
if menu.startswith("4"):
    default_ic50 = st.session_state.get("ic50_threshold", 1000)
    try:
        default_ic50 = int(default_ic50)
    except (TypeError, ValueError):
        default_ic50 = 1000
    seuil = st.sidebar.slider(
        "Activity threshold IC50 (nM)",
        min_value=10,
        max_value=10000,
        value=default_ic50,
        step=10,
    )
    st.session_state["ic50_threshold"] = seuil

render_feedback_box(sidebar=True)


# ----- Initialisation des variables session -----
def init_session():
    st.session_state.setdefault("last_selected_desc", ["Poids moléculaire", "LogP", "TPSA"])
    st.session_state.setdefault("clf", None)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("to_predict", [])
    st.session_state.setdefault("session_entries", [])
    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("warning_messages", [])
    st.session_state.setdefault("session_status", "Idle")
    st.session_state.setdefault("selected_pathology", None)
    st.session_state.setdefault("selected_target_name", None)
    st.session_state.setdefault("selected_target_chembl_id", None)
    st.session_state.setdefault("gene_symbol", None)
    st.session_state.setdefault("gene_name", None)
    # Thresholds used in Step 4 (activity / probability)
    st.session_state.setdefault("ic50_threshold", None)
    st.session_state.setdefault("step4_admet_prob_threshold", None)

init_session()
reset_warnings()

# ----- Routing vers les Genmap_modules -----
if menu.startswith("1"):
    from Genmap_modules.step1_gene_input import run as run_step1
    run_step1()
elif menu.startswith("2"):
    from Genmap_modules.step2_target_selection import run as run_step2
    run_step2()
elif menu.startswith("3"):
    from Genmap_modules.step3_molecule_selection import run as run_step3
    run_step3()
elif menu.startswith("7"):
    from Genmap_modules.step4_5_admet_fast import run as run_step4_5
    run_step4_5()
elif menu.startswith("4"):
    from Genmap_modules.step4_prediction import run as run_step4
    run_step4()
elif menu.startswith("8"):
    from Genmap_modules.step4c_similarity_network import run as run_step4c
    run_step4c()
elif menu.startswith("6"):
    from Genmap_modules.step3b_insilico_design import run as run_step3b
    run_step3b()
elif menu.startswith("5"):
    from Genmap_modules.step5_export import run as run_step5
    run_step5()

status_label = get_session_status()
pathology_label = st.session_state.get("selected_pathology") or "Not selected"
gene_info = st.session_state.get("selected_gene") or {}
gene_symbol = gene_info.get("symbol") or st.session_state.get("gene_symbol")
gene_name = gene_info.get("name") or st.session_state.get("gene_name")
if gene_symbol and gene_name:
    gene_display = f"{gene_symbol} ({gene_name})"
elif gene_symbol:
    gene_display = gene_symbol
elif gene_name:
    gene_display = gene_name
else:
    gene_display = "Not selected"
target_name = st.session_state.get("selected_target_name")
target_id = st.session_state.get("selected_target_chembl_id")
if target_name and target_id:
    target_display = f"{target_name} ({target_id})"
elif target_id:
    target_display = target_id
elif target_name:
    target_display = target_name
else:
    target_display = "Not selected"

ic50_thr = st.session_state.get("ic50_threshold")
prob_thr = st.session_state.get("step4_admet_prob_threshold")
ic50_display = f"{ic50_thr} nM" if ic50_thr is not None else "Not set"
prob_display = f"{prob_thr:.2f}" if isinstance(prob_thr, (int, float)) else "Not set"
warnings = get_warnings()
with status_panel:
    st.markdown("<hr style='margin:0.4rem 0;' />", unsafe_allow_html=True)
    st.markdown(STATUS_PANEL_STYLE, unsafe_allow_html=True)
    warning_items = "".join(f"<li>{escape(message)}</li>" for message in warnings)
    if warnings:
        warnings_html = f"<div class='status-subtitle'>Warnings</div><ul>{warning_items}</ul>"
    else:
        warnings_html = "<p class='status-muted'>No warnings.</p>"
    panel_html = f"""<div class="sidebar-status-panel">
    <div class="status-title">Session status</div>
    <p class="status-line"><strong>Step:</strong> {status_label}</p>
    <p class="status-line"><strong>Pathology:</strong> {pathology_label}</p>
    <p class="status-line"><strong>Gene:</strong> {gene_display}</p>
    <p class="status-line"><strong>Target:</strong> {target_display}</p>
    <p class="status-line"><strong>IC50 threshold:</strong> {ic50_display}</p>
    <p class="status-line"><strong>Activity prob. thr.:</strong> {prob_display}</p>
    {warnings_html}
    </div>"""
    st.markdown(panel_html, unsafe_allow_html=True)
render_admin_panel()

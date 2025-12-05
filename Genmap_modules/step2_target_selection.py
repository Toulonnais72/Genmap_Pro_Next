import streamlit as st
import requests
from chembl_webresource_client.new_client import new_client
from stmol import showmol
import py3Dmol
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_targets_from_uniprot(uniprot_id):
    try:
        res = new_client.target.filter(target_components__accession=uniprot_id)
        return list(res)
    except Exception:
        return []

def get_targets_from_ensembl(ensembl_id):
    # Mapping Ensembl → UniProt via UniProt API, puis cible ChEMBL
    url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/stream?from=Ensembl_PRO&to=UniProtKB_AC-ID&ids={ensembl_id}"
    accs = []
    try:
        r = requests.get(url, timeout=8, verify=False)
        if r.ok:
            for line in r.text.splitlines():
                if line and not line.startswith("From"):
                    accs.append(line.strip().split("\t")[-1])
    except Exception:
        pass
    # Cherche les cibles pour tous les accs trouvés
    targets = []
    for acc in accs:
        targets += get_targets_from_uniprot(acc)
    return targets, accs

def get_uniprot_from_gene_symbol(gene_symbol):
    """Recherche UniProt(s) associés à un symbole de gène humain."""
    url = "https://mygene.info/v3/query"
    try:
        r = requests.get(url, params={
            "q": gene_symbol,
            "species": "human",
            "fields": "uniprot",
            "size": 1
        }, timeout=8, verify=False)
        accs = []
        if r.ok:
            hits = r.json().get("hits", [])
            for h in hits:
                up = h.get("uniprot")
                if isinstance(up, dict):
                    for v in up.values():
                        if isinstance(v, list):
                            accs += v
                        else:
                            accs.append(v)
                elif isinstance(up, str):
                    accs.append(up)
        return accs
    except Exception:
        return []

def run():
    st.header("2. Sélection de la cible ChEMBL / Visualisation 3D")

    # Récupération entrée de l'étape 1
    gene = st.session_state.get("selected_gene", {})
    uniprot_id = st.session_state.get("uniprot_id")
    ensembl_id = st.session_state.get("ensembl_id")

    if uniprot_id:
        st.info(f"Recherche directe pour UniProt : {uniprot_id}")
        targets = get_targets_from_uniprot(uniprot_id)
        accs = [uniprot_id]
    elif ensembl_id:
        st.info(f"Recherche pour Ensembl : {ensembl_id}")
        targets, accs = get_targets_from_ensembl(ensembl_id)
    elif gene.get("symbol"):
        st.info(f"Recherche pour symbole de gène : {gene['symbol']}")
        accs = get_uniprot_from_gene_symbol(gene["symbol"])
        targets = []
        for acc in accs:
            targets += get_targets_from_uniprot(acc)
    else:
        st.warning("Veuillez d'abord saisir un gène, un ID UniProt ou un ID Ensembl à l'étape 1.")
        return

    if not targets:
        st.error("Aucune cible ChEMBL trouvée pour l'entrée fournie.")
        return

    # Préparation des choix pour l'utilisateur
    choices = [
        f"{t.get('pref_name','[Sans nom]')} | {t['target_chembl_id']} | {', '.join([c.get('accession','') for c in t.get('target_components', []) if c.get('accession')])}"
        for t in targets
    ]
    idx = st.selectbox("Choisissez une cible ChEMBL", range(len(choices)), format_func=lambda i: choices[i])
    selected_target = targets[idx]
    st.session_state["selected_target_chembl_id"] = selected_target["target_chembl_id"]
    st.success(f"Cible sélectionnée : {selected_target['pref_name']} ({selected_target['target_chembl_id']})")

    # Affichage 3D si possible
    accessions = [c.get('accession') for c in selected_target.get('target_components', []) if c.get('accession')]
    if accessions:
        acc = accessions[0]
        url_pdb = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
        r_pdb = requests.get(url_pdb, timeout=10, verify=False)
        pdb_ids = []
        if r_pdb.ok:
            data = r_pdb.json()
            for ref in data.get("uniProtKBCrossReferences", []):
                if ref.get("database") == "PDB":
                    pdb_ids.append(ref.get("id"))
        if pdb_ids:
            pdb_id = st.selectbox("Structure 3D PDB disponible :", pdb_ids)
            view = py3Dmol.view(query=f"pdb:{pdb_id}")
            view.setStyle({"cartoon": {"color": "spectrum"}})
            view.zoomTo()
            showmol(view, height=500, width=700)
            st.info(f"Affichage de la structure 3D PDB : {pdb_id} pour {acc}")
        else:
            st.info("Aucune structure PDB disponible pour cette cible.")
    else:
        st.info("Aucune accession UniProt détectée pour cette cible.")


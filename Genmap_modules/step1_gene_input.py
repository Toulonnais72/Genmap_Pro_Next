import streamlit as st
import requests
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Dictionnaire de gènes par catégories pathologiques (personnalise à volonté)
PATHO_GENES = {
    "Oncologie": [
        {"symbol": "TP53", "name": "Tumor protein p53"},
        {"symbol": "BRCA1", "name": "Breast cancer type 1 susceptibility protein"},
        {"symbol": "BRCA2", "name": "Breast cancer type 2 susceptibility protein"},
        {"symbol": "KRAS", "name": "GTPase KRas"},
        {"symbol": "NRAS", "name": "GTPase NRas"},
        {"symbol": "HRAS", "name": "GTPase HRas"},
        {"symbol": "BRAF", "name": "Serine/threonine-protein kinase B-raf"},
        {"symbol": "EGFR", "name": "Epidermal growth factor receptor"},
        {"symbol": "PIK3CA", "name": "Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit alpha"},
        {"symbol": "ALK", "name": "Anaplastic lymphoma kinase"},
        {"symbol": "MET", "name": "Hepatocyte growth factor receptor"},
        {"symbol": "ERBB2", "name": "Receptor tyrosine-protein kinase erbB-2 (HER2)"},
        {"symbol": "PTEN", "name": "Phosphatase and tensin homolog"},
        {"symbol": "CDKN2A", "name": "Cyclin-dependent kinase inhibitor 2A"},
        {"symbol": "APC", "name": "Adenomatous polyposis coli protein"},
        {"symbol": "MLH1", "name": "MutL homolog 1"},
        {"symbol": "MSH2", "name": "MutS homolog 2"},
        {"symbol": "RB1", "name": "Retinoblastoma-associated protein"},
        {"symbol": "VHL", "name": "Von Hippel-Lindau tumor suppressor"},
        {"symbol": "STK11", "name": "Serine/threonine kinase 11 (LKB1)"},
    ],
    "Maladies rares": [
        {"symbol": "CFTR", "name": "Cystic fibrosis transmembrane conductance regulator"},
        {"symbol": "SMN1", "name": "Survival of motor neuron protein 1"},
        {"symbol": "FBN1", "name": "Fibrillin-1"},
        {"symbol": "GAA", "name": "Acid alpha-glucosidase"},
        {"symbol": "DMD", "name": "Dystrophin"},
        {"symbol": "PAH", "name": "Phenylalanine-4-hydroxylase"},
        {"symbol": "IDUA", "name": "Alpha-L-iduronidase"},
        {"symbol": "GBA", "name": "Glucocerebrosidase"},
        {"symbol": "COL7A1", "name": "Collagen alpha-1(VII) chain"},
        {"symbol": "PHEX", "name": "Phosphate-regulating neutral endopeptidase"},
        {"symbol": "MECP2", "name": "Methyl CpG binding protein 2"},
        {"symbol": "GLA", "name": "Alpha-galactosidase A"},
        {"symbol": "LAMP2", "name": "Lysosome-associated membrane glycoprotein 2"},
        {"symbol": "SLC6A8", "name": "Creatine transporter 1"},
    ],
    "Cardiologie": [
        {"symbol": "MYH7", "name": "Myosin-7"},
        {"symbol": "SCN5A", "name": "Sodium channel protein type 5 subunit alpha"},
        {"symbol": "TNNT2", "name": "Troponin T, cardiac muscle"},
        {"symbol": "LMNA", "name": "Prelamin-A/C"},
        {"symbol": "RYR2", "name": "Ryanodine receptor 2"},
        {"symbol": "KCNQ1", "name": "Potassium voltage-gated channel subfamily Q member 1"},
        {"symbol": "MYBPC3", "name": "Myosin-binding protein C, cardiac-type"},
        {"symbol": "PCSK9", "name": "Proprotein convertase subtilisin/kexin type 9"},
        {"symbol": "LDLR", "name": "Low-density lipoprotein receptor"},
    ],
    "Neurologie": [
        {"symbol": "APP", "name": "Amyloid-beta A4 protein precursor"},
        {"symbol": "PSEN1", "name": "Presenilin-1"},
        {"symbol": "MAPT", "name": "Microtubule-associated protein tau"},
        {"symbol": "HTT", "name": "Huntingtin"},
        {"symbol": "SOD1", "name": "Superoxide dismutase [Cu-Zn]"},
        {"symbol": "LRRK2", "name": "Leucine-rich repeat serine/threonine-protein kinase 2"},
        {"symbol": "SNCA", "name": "Alpha-synuclein"},
        {"symbol": "GRN", "name": "Progranulin"},
        {"symbol": "TARDBP", "name": "TAR DNA-binding protein 43"},
    ],
    "Immunologie": [
        {"symbol": "HLA-A", "name": "Major histocompatibility complex, class I, A"},
        {"symbol": "IL2RA", "name": "Interleukin-2 receptor subunit alpha"},
        {"symbol": "CTLA4", "name": "Cytotoxic T-lymphocyte protein 4"},
        {"symbol": "FOXP3", "name": "Forkhead box protein P3"},
        {"symbol": "TNF", "name": "Tumor necrosis factor"},
        {"symbol": "IFNG", "name": "Interferon gamma"},
        {"symbol": "IL6", "name": "Interleukin-6"},
        {"symbol": "STAT3", "name": "Signal transducer and activator of transcription 3"},
        {"symbol": "JAK3", "name": "Janus kinase 3"},
    ],
    "Métabolisme/Endocrinologie": [
        {"symbol": "INS", "name": "Insulin"},
        {"symbol": "GCK", "name": "Glucokinase"},
        {"symbol": "GHR", "name": "Growth hormone receptor"},
        {"symbol": "LEP", "name": "Leptin"},
        {"symbol": "MC4R", "name": "Melanocortin receptor 4"},
        {"symbol": "TSHR", "name": "Thyroid-stimulating hormone receptor"},
        {"symbol": "POMC", "name": "Pro-opiomelanocortin"},
        {"symbol": "CYP21A2", "name": "Steroid 21-hydroxylase"},
        {"symbol": "HNF1A", "name": "Hepatocyte nuclear factor 1-alpha"},
    ],
    "Maladies infectieuses": [
        {"symbol": "CCR5", "name": "C-C chemokine receptor type 5 (HIV coreceptor)"},
        {"symbol": "TLR4", "name": "Toll-like receptor 4"},
        {"symbol": "IL28B", "name": "Interferon lambda-3 (HCV response)"},
        {"symbol": "CD4", "name": "T-cell surface glycoprotein CD4"},
        {"symbol": "IFNL3", "name": "Interferon lambda 3"},
        {"symbol": "DARC", "name": "Duffy antigen receptor for chemokines (malaria)"},
        {"symbol": "HLA-B", "name": "Major histocompatibility complex, class I, B"},
        {"symbol": "FUT2", "name": "Fucosyltransferase 2 (Norovirus resistance)"},
        {"symbol": "MX1", "name": "Interferon-induced GTP-binding protein Mx1"},
        {"symbol": "OAS1", "name": "2'-5'-oligoadenylate synthase 1"},
    ],
    "Hématologie": [
        {"symbol": "HBB", "name": "Hemoglobin subunit beta (sickle cell, thalassemia)"},
        {"symbol": "F8", "name": "Coagulation factor VIII"},
        {"symbol": "F9", "name": "Coagulation factor IX"},
        {"symbol": "VWF", "name": "Von Willebrand factor"},
        {"symbol": "G6PD", "name": "Glucose-6-phosphate dehydrogenase"},
        {"symbol": "HBA1", "name": "Hemoglobin subunit alpha 1"},
        {"symbol": "HBA2", "name": "Hemoglobin subunit alpha 2"},
        {"symbol": "ITGA2B", "name": "Integrin alpha-IIb"},
        {"symbol": "ITGB3", "name": "Integrin beta-3"},
        {"symbol": "ELANE", "name": "Neutrophil elastase"},
    ],
}

# === AJOUT : extension maladies rares ===

EXTRA_RARE_PATHO_GENES = {
    "Maladies rares — Neuromusculaires": [
        {"symbol": "DMD", "name": "Dystrophin"},
        {"symbol": "SMN1", "name": "Survival motor neuron 1"},
        {"symbol": "RYR1", "name": "Ryanodine receptor 1"},
        {"symbol": "LMNA", "name": "Prelamin A/C"},
        {"symbol": "COL6A1", "name": "Collagen VI alpha-1"},
        {"symbol": "COL6A2", "name": "Collagen VI alpha-2"},
        {"symbol": "COL6A3", "name": "Collagen VI alpha-3"},
        {"symbol": "FKRP", "name": "Fukutin-related protein"},
        {"symbol": "CAPN3", "name": "Calpain-3"},
        {"symbol": "SGCA", "name": "Sarcoglycan alpha"},
        {"symbol": "SGCB", "name": "Sarcoglycan beta"},
        {"symbol": "SGCD", "name": "Sarcoglycan delta"},
        {"symbol": "SGCG", "name": "Sarcoglycan gamma"},
        {"symbol": "LAMA2", "name": "Laminin subunit alpha-2"},
    ],

    "Maladies rares — Lysosomales": [
        {"symbol": "GBA", "name": "Glucocerebrosidase (Gaucher)"},
        {"symbol": "GLA", "name": "Alpha-galactosidase A (Fabry)"},
        {"symbol": "IDUA", "name": "Alpha-L-iduronidase (MPS I)"},
        {"symbol": "IDS", "name": "Iduronate-2-sulfatase (MPS II)"},
        {"symbol": "GUSB", "name": "Beta-glucuronidase (MPS VII)"},
        {"symbol": "SGSH", "name": "Heparan-N-sulfatase (MPS IIIA)"},
        {"symbol": "NAGLU", "name": "Alpha-N-acetylglucosaminidase (MPS IIIB)"},
        {"symbol": "HGSNAT", "name": "Heparan-alpha-glucosaminide N-acetyltransferase (MPS IIIC)"},
        {"symbol": "ARSA", "name": "Arylsulfatase A (MLD)"},
        {"symbol": "HEXA", "name": "Hexosaminidase A (Tay–Sachs)"},
        {"symbol": "SMPD1", "name": "Sphingomyelin phosphodiesterase 1 (Niemann–Pick A/B)"},
        {"symbol": "GALC", "name": "Galactocerebrosidase (Krabbe)"},
        {"symbol": "NPC1", "name": "Niemann–Pick C1"},
        {"symbol": "PPT1", "name": "Palmitoyl-protein thioesterase 1 (CLN1)"},
        {"symbol": "TPP1", "name": "Tripeptidyl-peptidase 1 (CLN2)"},
        {"symbol": "CLN3", "name": "Batten disease (CLN3)"},
    ],

    "Maladies rares — Mitochondriales": [
        {"symbol": "MT-ND1", "name": "Mitochondrial Complex I subunit ND1"},
        {"symbol": "MT-ND4", "name": "Mitochondrial Complex I subunit ND4"},
        {"symbol": "MT-ATP6", "name": "ATP synthase subunit a (ATP6)"},
        {"symbol": "POLG", "name": "DNA polymerase gamma"},
        {"symbol": "TWNK", "name": "Twinkle helicase"},
        {"symbol": "SURF1", "name": "Surfeit locus protein 1 (Leigh)"},
        {"symbol": "OPA1", "name": "Dynamin-like GTPase OPA1"},
        {"symbol": "OPA3", "name": "Optic atrophy 3"},
        {"symbol": "PDHA1", "name": "Pyruvate dehydrogenase E1 alpha"},
        {"symbol": "SLC52A2", "name": "Riboflavin transporter 2"},
        {"symbol": "SLC52A3", "name": "Riboflavin transporter 3"},
    ],

    "Maladies rares — Peroxysomales": [
        {"symbol": "ABCD1", "name": "Adrenoleukodystrophy protein (X-ALD)"},
        {"symbol": "PEX1", "name": "Peroxin-1"},
        {"symbol": "PEX6", "name": "Peroxin-6"},
        {"symbol": "PHYH", "name": "Phytanoyl-CoA hydroxylase (Refsum)"},
        {"symbol": "HSD17B4", "name": "Peroxisomal multifunctional enzyme 2"},
    ],

    "Maladies rares — Ciliopathies": [
        {"symbol": "CEP290", "name": "Centrosomal protein 290 (Joubert/Leber)"},
        {"symbol": "AHI1", "name": "Jouberin (Joubert)"},
        {"symbol": "BBS1", "name": "Bardet–Biedl syndrome 1"},
        {"symbol": "BBS10", "name": "Bardet–Biedl syndrome 10"},
        {"symbol": "NPHP1", "name": "Nephrocystin-1 (Nephronophthisis)"},
        {"symbol": "IFT140", "name": "Intraflagellar transport 140"},
        {"symbol": "OFD1", "name": "Oral-facial-digital syndrome 1"},
        {"symbol": "PKHD1", "name": "Fibrocystin (ARPKD)"},
    ],

    "Maladies rares — Immunodéficiences primitives": [
        {"symbol": "BTK", "name": "Bruton tyrosine kinase (Agammaglobulinemia)"},
        {"symbol": "ADA", "name": "Adenosine deaminase (SCID)"},
        {"symbol": "IL2RG", "name": "Common gamma chain (X-SCID)"},
        {"symbol": "RAG1", "name": "Recombination activating 1"},
        {"symbol": "RAG2", "name": "Recombination activating 2"},
        {"symbol": "CYBB", "name": "gp91phox (CGD)"},
        {"symbol": "WAS", "name": "Wiskott–Aldrich syndrome"},
        {"symbol": "STAT3", "name": "Hyper-IgE (dominant)"},
        {"symbol": "DOCK8", "name": "Combined immunodeficiency"},
        {"symbol": "AIRE", "name": "APECED"},
    ],

    "Maladies rares — Coagulopathies / Hématologie": [
        {"symbol": "F7", "name": "Coagulation factor VII deficiency"},
        {"symbol": "F13A1", "name": "Coagulation factor XIII A1"},
        {"symbol": "HBB", "name": "Hemoglobin beta (SCD/thalassemia)"},
        {"symbol": "HBA1", "name": "Hemoglobin alpha 1"},
        {"symbol": "HBA2", "name": "Hemoglobin alpha 2"},
        {"symbol": "ELANE", "name": "Neutrophil elastase (Kostmann)"},
        {"symbol": "GP1BA", "name": "Bernard–Soulier syndrome"},
        {"symbol": "ITGA2B", "name": "Glanzmann thrombasthenia (alphaIIb)"},
        {"symbol": "ITGB3", "name": "Glanzmann thrombasthenia (beta3)"},
    ],

    "Maladies rares — Tissu conjonctif / Aorte": [
        {"symbol": "FBN1", "name": "Fibrillin-1 (Marfan)"},
        {"symbol": "COL3A1", "name": "Collagen III alpha-1 (vEDS)"},
        {"symbol": "COL5A1", "name": "Collagen V alpha-1 (cEDS)"},
        {"symbol": "COL5A2", "name": "Collagen V alpha-2 (cEDS)"},
        {"symbol": "TGFBR2", "name": "TGF-beta receptor 2 (Loeys–Dietz)"},
        {"symbol": "TGFBR1", "name": "TGF-beta receptor 1 (Loeys–Dietz)"},
        {"symbol": "SMAD3", "name": "SMAD family member 3"},
        {"symbol": "FLNA", "name": "Filamin A (periventricular nodular heterotopia)"},
    ],

    "Maladies rares — Endocrino-métaboliques": [
        {"symbol": "PAH", "name": "Phenylalanine hydroxylase (PKU)"},
        {"symbol": "GAA", "name": "Acid alpha-glucosidase (Pompe)"},
        {"symbol": "GALT", "name": "Galactose-1-phosphate uridylyltransferase"},
        {"symbol": "HGD", "name": "Homogentisate 1,2-dioxygenase (Alkaptonuria)"},
        {"symbol": "OTC", "name": "Ornithine transcarbamylase (Urea cycle)"},
        {"symbol": "ASS1", "name": "Argininosuccinate synthase 1"},
        {"symbol": "ASL", "name": "Argininosuccinate lyase"},
        {"symbol": "CPS1", "name": "Carbamoyl-phosphate synthase 1"},
        {"symbol": "SLC22A5", "name": "Carnitine transporter (Primary carnitine deficiency)"},
        {"symbol": "ABCC8", "name": "KATP channel (Congenital hyperinsulinism)"},
        {"symbol": "KCNJ11", "name": "KATP channel (Congenital hyperinsulinism)"},
    ],

    "Maladies rares — Rénales / Oreille interne": [
        {"symbol": "COL4A5", "name": "Type IV collagen alpha-5 (Alport)"},
        {"symbol": "COL4A3", "name": "Type IV collagen alpha-3"},
        {"symbol": "COL4A4", "name": "Type IV collagen alpha-4"},
        {"symbol": "UMOD", "name": "Uromodulin (ADTKD)"},
        {"symbol": "PKHD1", "name": "AR-polycystic kidney disease"},
        {"symbol": "SLC26A4", "name": "Pendrin (Pendred syndrome)"},
        {"symbol": "TECTA", "name": "Alpha-tectorin (DFNA8/12)"},
        {"symbol": "GJB2", "name": "Connexin 26 (DFNB1)"},
    ],

    "Maladies rares — Dermatologie / Epiderme": [
        {"symbol": "COL7A1", "name": "Collagen VII (DEB)"},
        {"symbol": "KRT14", "name": "Keratin 14 (EBS)"},
        {"symbol": "KRT5", "name": "Keratin 5 (EBS)"},
        {"symbol": "DSP", "name": "Desmoplakin (Carvajal/Naxos)"},
        {"symbol": "DSPP", "name": "Dentin sialophosphoprotein (Dentinogenesis imperfecta)"},
        {"symbol": "FERMT1", "name": "Kindlin-1 (Kindler)"},
    ],

    "Maladies rares — Syndromiques": [
        {"symbol": "PTPN11", "name": "Noonan syndrome (SHP2)"},
        {"symbol": "SOS1", "name": "Noonan syndrome"},
        {"symbol": "RAF1", "name": "Noonan / RASopathy"},
        {"symbol": "HRAS", "name": "Costello syndrome"},
        {"symbol": "MEK1", "name": "RASopathy (MAP2K1)"},
        {"symbol": "MEK2", "name": "RASopathy (MAP2K2)"},
        {"symbol": "NF1", "name": "Neurofibromatosis type 1"},
        {"symbol": "TSC1", "name": "Tuberous sclerosis 1"},
        {"symbol": "TSC2", "name": "Tuberous sclerosis 2"},
        {"symbol": "PTCH1", "name": "Gorlin syndrome"},
    ],
}

# Fusion non destructive: on ajoute sans dupliquer les symboles déjà présents
for cat, genes in EXTRA_RARE_PATHO_GENES.items():
    PATHO_GENES.setdefault(cat, [])
    existing = {g["symbol"] for g in PATHO_GENES[cat]}
    PATHO_GENES[cat].extend([g for g in genes if g["symbol"] not in existing])

def detect_id_type(query):
    q = query.strip().upper()
    if re.fullmatch(r"P\d{5}(\.\d+)?", q) or re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9](\.\d+)?", q):
        return "uniprot"
    elif re.fullmatch(r"ENSG\d{11}", q):
        return "ensembl"
    else:
        return "gene"

def get_gene_suggestions(query, size=12):
    url = "https://mygene.info/v3/query"
    params = {
        "q": query,
        "species": "human",
        "fields": "symbol,name,entrezgene,ensembl.gene,alias,uniprot",
        "size": size
    }
    try:
        r = requests.get(url, params=params, timeout=7, verify=False)
        r.raise_for_status()
        hits = r.json().get("hits", [])
    except Exception as e:
        st.warning(f"Erreur lors de la requête vers mygene.info : {e}")
        return []
    suggestions = []
    for g in hits:
        label = f"{g.get('symbol','?')} — {g.get('name','?')}"
        alias = g.get("alias", [])
        if alias:
            if isinstance(alias, str):
                alias = [alias]
            label += f" (Alias: {', '.join(alias[:2])})"
        eid = g.get('entrezgene')
        ensembl = g.get("ensembl", None)
        if isinstance(ensembl, list) and ensembl:
            ensembl_id = ensembl[0].get("gene")
        elif isinstance(ensembl, dict):
            ensembl_id = ensembl.get("gene")
        else:
            ensembl_id = None
        eid = eid or ensembl_id
        label += f" (Entrez: {eid})" if eid else ""
        suggestions.append({
            "label": label,
            "symbol": g.get("symbol"),
            "name": g.get("name", ""),
            "entrezgene": g.get("entrezgene"),
            "ensembl": ensembl_id,
            "aliases": alias,
            "uniprot": g.get("uniprot", None),
            "full": g
        })
    return suggestions

def run():
    st.header("1. Sélection du gène — par pathologie ou recherche libre")
    st.markdown("**Sélectionnez une pathologie puis un gène, ou effectuez une recherche libre ci-dessous.**")

    #with st.expander("🔍 Recherche libre d'un gène (symbole, nom, UniProt, Ensembl)", expanded=True):
    #    user_query = st.text_input("Recherche", key="gene_search_2")
        # --> Place your autocomplete suggestion logic here
        # (or use your existing code)

    st.markdown("### 🧬 Gènes par pathologie")

    # Expanders for each pathology (dropdown-style menu)
    for cat, genes in PATHO_GENES.items():
        with st.expander(f"{cat} ({len(genes)})", expanded=False):
            # Dynamically set columns (2 to 4 based on number of genes)
            ncols = 2 if len(genes) <= 8 else 3 if len(genes) < 16 else 4
            cols = st.columns(ncols)
            for idx, gene in enumerate(genes):
                with cols[idx % ncols]:
                    if st.button(f"{gene['symbol']} ({gene['name']})", key=f"button_{cat}_{gene['symbol']}"):
                        st.session_state["user_query"] = gene['symbol']
                        st.success(f"Gène sélectionné : {gene['symbol']} — {gene['name']}")

    # Recherche libre
    user_query = st.text_input(
        "Recherche gène / UniProt / Ensembl (nom, symbole ou alias accepté)",
        value=st.session_state.get("user_query", ""),
        key="gene_search"
    )

    if user_query and len(user_query.strip()) > 2:
        id_type = detect_id_type(user_query)
        st.caption(f"Type d'entrée détecté : `{id_type}`")
        if id_type == "uniprot":
            st.success(f"ID UniProt reconnu : {user_query.strip().upper()}")
            st.session_state["uniprot_id"] = user_query.strip().upper()
            st.session_state["selected_gene"] = {"uniprot_id": user_query.strip().upper()}
        elif id_type == "ensembl":
            st.success(f"ID Ensembl reconnu : {user_query.strip().upper()}")
            st.session_state["ensembl_id"] = user_query.strip().upper()
            st.session_state["selected_gene"] = {"ensembl": user_query.strip().upper()}
        else:
            suggestions = get_gene_suggestions(user_query.strip())
            if suggestions:
                st.markdown("#### Sélectionnez un gène dans la liste ci-dessous :")
                selected = st.selectbox(
                    "Suggestions de gènes trouvés :",
                    suggestions,
                    format_func=lambda s: s["label"],
                    key="gene_suggestion"
                )
                st.success(f"Gène sélectionné : **{selected['label']}**")
                with st.expander("Infos détaillées sur le gène sélectionné"):
                    st.write(f"**Symbole :** {selected['symbol']}")
                    st.write(f"**Nom complet :** {selected['name']}")
                    st.write(f"**Aliases :** {', '.join(selected['aliases']) if selected['aliases'] else 'N/A'}")
                    st.write(f"**Entrez :** {selected['entrezgene']}")
                    st.write(f"**Ensembl :** {selected['ensembl']}")
                    st.write(f"**UniProt :** {selected['uniprot']}")
                    st.json(selected["full"], expanded=False)
                st.session_state["selected_gene"] = selected
                st.session_state["gene_symbol"] = selected["symbol"]
                st.session_state["gene_entrez"] = selected["entrezgene"]
                st.session_state["gene_ensembl"] = selected["ensembl"]
                st.session_state["gene_uniprot"] = selected["uniprot"]
            else:
                st.warning("Aucun gène correspondant trouvé.")
                st.session_state["selected_gene"] = None
    else:
        st.info("Commencez à taper pour rechercher un gène (ou cliquez sur un gène pathologique ci-dessus).")

    # Affichage de confirmation
    if st.session_state.get("selected_gene"):
        info = st.session_state["selected_gene"]
        st.success(
            f"Sélection enregistrée pour la suite : "
            f"**Gène : {info.get('symbol', '')}** "
            f"**Nom : {info.get('name', '')}** "
            f"**UniProt : {info.get('uniprot_id', info.get('uniprot', ''))}** "
            f"**Ensembl : {info.get('ensembl', '')}**"
        )
    else:
        st.info("Aucune sélection pour le moment.")


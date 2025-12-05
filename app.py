"""Minimal Streamlit application with a visualisation tab."""

import numpy as np
import streamlit as st

from modules.viz import mol_to_image, plot_umap
from modules.lims_connectors import export_sdf, export_json, export_pdf

st.set_page_config(page_title="GenMap Visualisation", layout="wide")

st.title("GenMap Visualisation")
tab1, tab2 = st.tabs(["UMAP", "Molecule"])

with tab1:
    st.write("Project random data using UMAP and display with Plotly.")
    if st.button("Generate example"):
        data = np.random.rand(20, 10)
        fig = plot_umap(data)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    smiles = st.text_input("SMILES", value="CCO")
    if smiles:
        img = mol_to_image(smiles)
        st.image(img)
    st.write("Batch export")
    smiles_list = st.text_area("SMILES list", value="CCO\nO")
    molecules = [{"smiles": s} for s in smiles_list.splitlines() if s.strip()]
    if molecules:
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "Export SDF", export_sdf(molecules), file_name="molecules.sdf"
        )
        col2.download_button(
            "Export JSON", export_json(molecules), file_name="molecules.json"
        )
        col3.download_button(
            "Export PDF", export_pdf(molecules), file_name="molecules.pdf"
        )

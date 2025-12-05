import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
from export_helpers import wrap_text_cell, draw_table_row

# --- Utilities for fonts & text safety ---

def _assets_dir():
    # Try to resolve an Assets/ folder next to this module; fallback to CWD/Assets
    here = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    cand = os.path.join(here, 'Assets')
    return cand if os.path.isdir(cand) else os.path.join(os.getcwd(), 'Assets')

def _setup_fonts(pdf: FPDF):
    """Attempt to register DejaVu (regular/bold/italic). Returns a dict with availability flags.
    If DejaVu is missing, we will fallback to built-in Helvetica (no full Unicode)."""
    avail = {"dejavu": False, "dejavu_b": False, "dejavu_i": False}
    assets = _assets_dir()
    ttf_regular = os.path.join(assets, "DejaVuSans.ttf")
    ttf_bold    = os.path.join(assets, "DejaVuSans-Bold.ttf")
    ttf_italic  = os.path.join(assets, "DejaVuSans-Oblique.ttf")  # optional

    try:
        if os.path.isfile(ttf_regular):
            pdf.add_font("DejaVu", "", ttf_regular, uni=True)
            avail["dejavu"] = True
        if os.path.isfile(ttf_bold):
            pdf.add_font("DejaVu", "B", ttf_bold, uni=True)
            avail["dejavu_b"] = True
        if os.path.isfile(ttf_italic):
            pdf.add_font("DejaVu", "I", ttf_italic, uni=True)
            avail["dejavu_i"] = True
    except Exception:
        # If any font load fails, we keep the flags as set; built-ins will be used as needed
        pass

    # Final safety: strip any TTF fonts that point to a non‑existent file.
    # This avoids crashes if an older configuration registered fonts with
    # absolute paths that are no longer valid on this machine.
    try:
        bad_keys = []
        for fname, meta in getattr(pdf, "fonts", {}).items():
            ttf_path = meta.get("ttffile")
            if ttf_path and not os.path.isfile(ttf_path):
                bad_keys.append(fname)
        for fname in bad_keys:
            del pdf.fonts[fname]
    except Exception:
        # On any error, fall back silently; built-in fonts will still work.
        pass

    # Ensure availability flags are consistent with what is actually registered
    fonts = getattr(pdf, "fonts", {})
    if "DejaVu" not in fonts:
        # No valid DejaVu font currently registered: force fallback to Helvetica
        avail["dejavu"] = False
        avail["dejavu_b"] = False
        avail["dejavu_i"] = False

    return avail

def _safe_text(s, unicode_ok: bool):
    """If we have Unicode fonts, return as-is. Otherwise, coerce to latin-1-safe text."""
    if s is None:
        return ""
    s = str(s)
    if unicode_ok:
        return s
    try:
        return s.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return s


def _fmt_num(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

# --- Streamlit Step 5 ---

def run():
    st.header("5. Report / Export")
    df = st.session_state.get("prediction_results", None)

    if df is None or df.empty:
        st.info("No predictions available.")
        return

    gene = st.session_state.get("selected_gene", {})
    target = st.session_state.get("selected_target_chembl_id", "")
    ic_th = st.session_state.get("ic50_threshold", None)  # seuil IC50 (nM) choisi à l'étape 4

    st.markdown("### 🧬 Gene selected")
    if gene:
        st.markdown(
            f"**Gene:** `{gene.get('symbol', '')}`  \n"
            f"**Name:** {gene.get('name', '')}  \n"
            f"**Ensembl:** {gene.get('ensembl', '')}  \n"
            f"**UniProt:** {gene.get('uniprot', '')}"
        )
    else:
        st.info("No gene selected.")

    st.markdown("### 🎯 Selected target")
    if target:
        st.markdown(f"**ChEMBL Target :** [{target}](https://www.ebi.ac.uk/chembl/target_report_card/{target})")
    else:
        st.info("No target selected.")

    # Inclure le seuil dans l'export CSV
    df_csv = df.copy()
    if ic_th is not None and "Seuil IC50 (nM)" not in df_csv.columns:
        df_csv["Seuil IC50 (nM)"] = ic_th

    st.dataframe(df_csv)
    csv = df_csv.to_csv(index=False).encode("utf-8")
    st.download_button("�Y'� Export CSV ", csv, "prediction_results.csv", "text/csv")

    if st.button("�Y"" Generate PDF"):
        pdf = FPDF(orientation="L", unit="mm", format="A4")

        # 1) Register fonts (DejaVu from Assets, if available).
        #    This must happen *before* adding any pages so FPDF knows the family.
        avail = _setup_fonts(pdf)
        has_unicode = avail["dejavu"]  # True only if DejaVuSans.ttf was found in Assets
        font_name = "DejaVu" if has_unicode else "Helvetica"

        # 2) Set a valid current font so add_page() does not hit an undefined family.
        pdf.set_font(font_name, "", 12)

        # 3) Now we can add pages.
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf = FPDF(orientation="L", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Register fonts (DejaVu if available; else fallback to Helvetica)
        avail = _setup_fonts(pdf)
        has_unicode = avail["dejavu"]  # bold/italic optional
        font_name = "DejaVu" if has_unicode else "Helvetica"

        # Title
        pdf.set_font(font_name, "", 14)
        pdf.cell(0, 12, txt=_safe_text("Molecular Predictons Reports", has_unicode), ln=True, align="C")

        # Gene / target sections
        pdf.set_font(font_name, "", 11)
        pdf.ln(2)
        pdf.cell(0, 8, _safe_text("Informations on the gene selected:", has_unicode), ln=True)

        def _kv(label, value):
            pdf.set_font(font_name, "B" if (avail["dejavu_b"] if has_unicode else True) else "", 11)
            pdf.cell(35, 8, _safe_text(label, has_unicode), 0, 0)
            pdf.set_font(font_name, "", 11)
            pdf.cell(0, 8, _safe_text(value, has_unicode), ln=1)

        if gene:
            _kv("Gene:", gene.get("symbol", ""))
            _kv("Complete Name:", gene.get("name", ""))
            _kv("Ensembl:", gene.get("ensembl", ""))
            _kv("UniProt:", gene.get("uniprot", ""))
        else:
            pdf.cell(0, 8, _safe_text("No gene selected.", has_unicode), ln=True)

        pdf.ln(2)
        pdf.cell(0, 8, _safe_text("Informations on the molecular target:", has_unicode), ln=True)
        if target:
            _kv("ChEMBL ID :", target)
        else:
            pdf.cell(0, 8, _safe_text("No target selected", has_unicode), ln=True)

        # seuil IC50 choisi (si dispo)
        if ic_th is not None:
            _kv("IC50 (nM) threshold: ", str(ic_th))

        # Prepare dataframe copy for PDF (avoid emojis if no unicode)
        df_for_pdf = df.copy()
        if ic_th is not None and "Seuil IC50 (nM)" not in df_for_pdf.columns:
            df_for_pdf["Seuil IC50 (nM)"] = ic_th
        if "Prédiction" in df_for_pdf.columns:
            df_for_pdf["Prédiction"] = df_for_pdf["Prédiction"].replace({
                "🟩 Actif": "Actif", "🟥 Inactif": "Inactif"
            })

        pdf.ln(4)
        pdf.set_font(font_name, "B" if (avail["dejavu_b"] if has_unicode else True) else "", 12)
        pdf.cell(0, 8, _safe_text("Summary Table of Predictions:", has_unicode), ln=True)

        # ---------------- Compact table with SMILES clipped to cell width ----------------
        pdf.set_font(font_name, "", 9)

        # helper: fit text to given cell width (in mm) using the current font
        def _fit_text_to_cell(text: str, width_mm: float, right_padding: float = 2.0) -> str:
            t = str(text or "")
            max_w = max(0.0, width_mm - right_padding)
            if pdf.get_string_width(t) <= max_w:
                return t
            ell = "…"
            if pdf.get_string_width(ell) > max_w:
                ell = "..."
            lo, hi = 0, len(t)
            best = ell
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = t[:mid] + ell
                if pdf.get_string_width(cand) <= max_w:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best

        # Select and order columns
        cols = list(df_for_pdf.columns)
        preferred = [
            "Nom de la molécule", "Name", "SMILES", "Prédiction", "Probabilité",
            "Seuil IC50 (nM)",  # affiché si la place le permet
            "Poids moléculaire", "MolWt", "LogP", "MolLogP", "TPSA"
        ]
        cols_shown = [c for c in preferred if c in cols]
        for c in cols:
            if len(cols_shown) >= 6:
                break
            if c not in cols_shown:
                cols_shown.append(c)
        cols_shown = cols_shown[:6]

        # Column widths (mm) tuned for landscape A4 usable width ~277
        # Name | SMILES | Pred | Proba | C5 | C6
        widths_map = [40, 95, 28, 28, 40, 40]
        col_widths = widths_map[:len(cols_shown)]

        # Header
        for w, col in zip(col_widths, cols_shown):
            pdf.cell(w, 8, _safe_text(str(col), has_unicode), border=1, align="C")
        pdf.ln()

        # Render rows with exact clipping for SMILES
        def _render_cell(val, colname, width):
            text = str(val)
            if colname.lower() == "smiles":
                text = _fit_text_to_cell(text, width)
            if colname in ("Poids moléculaire","MolWt","TPSA","Probabilité","LogP","MolLogP","IC50(nM)"):
                text = _fmt_num(val, 3)
            pdf.cell(width, 8, _safe_text(text, has_unicode), border=1)

        max_rows = 18
        for _, row in df_for_pdf.head(max_rows).iterrows():
            for w, col in zip(col_widths, cols_shown):
                _render_cell(row[col], col, w)
            pdf.ln()

        pdf.ln(4)
        pdf.set_font(font_name, "", 9)
        pdf.multi_cell(0, 6, _safe_text(
            "NB: SMILES abbreviated to column width for readability. The CSV contains all columns. An appendix with the full SMILES follows.",
            has_unicode
        ))

        # ---------------- Appendix with full SMILES ----------------
        pdf.add_page()
        pdf.set_font(font_name, "", 13)
        pdf.cell(0, 10, _safe_text("Appendix - Complete SMILES", has_unicode), ln=True)

        pdf.set_font(font_name, "", 8)

        w_name, w_smi = 70, 200
        pdf.cell(w_name, 7, _safe_text("Nom de la molécule", has_unicode), border=1)
        pdf.cell(w_smi, 7, _safe_text("SMILES (complet)", has_unicode), border=1)
        pdf.ln()

        # Source rows: use prediction_results if available to match user selection
        src = st.session_state.get("prediction_results", df_for_pdf)
        for _, r in src.iterrows():
            name = r.get("Nom de la molécule") or r.get("Name") or ""
            smi  = r.get("SMILES", "")
            y_before = pdf.get_y()
            x_before = pdf.get_x()

            pdf.multi_cell(w_name, 5, _safe_text(str(name), has_unicode), border=1)
            y_after = pdf.get_y()
            pdf.set_xy(x_before + w_name, y_before)
            pdf.multi_cell(w_smi, 5, _safe_text(str(smi), has_unicode), border=1)
            pdf.set_y(max(y_after, pdf.get_y()))

        # Save and offer download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as ftmp:
            pdf.output(ftmp.name)
            with open(ftmp.name, "rb") as f:
                st.download_button(
                    "Download PDF",
                    data=f.read(),
                    file_name="rapport_prediction.pdf",
                    mime="application/pdf",
                )
            try:
                os.remove(ftmp.name)
            except PermissionError:
                pass

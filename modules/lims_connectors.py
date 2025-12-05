import json
from io import BytesIO

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - optional dependency
    letter = (595.27, 841.89)
    canvas = None


def export_sdf(molecules):
    """Return SDF representation of molecules.

    Parameters
    ----------
    molecules : list of dict
        Each dict must contain a ``smiles`` key along with any
        additional properties which will be exported as SDF fields.

    Returns
    -------
    bytes
        The SDF file contents.
    """
    blocks = []
    for mol in molecules:
        if Chem is not None:
            rd_mol = Chem.MolFromSmiles(mol["smiles"])
            block = Chem.MolToMolBlock(rd_mol)
        else:  # fallback to a minimal block if RDKit is unavailable
            block = mol["smiles"] + "\n"
        props = []
        for key, value in mol.items():
            if key == "smiles":
                continue
            props.append(f"> <{key}>\n{value}\n")
        block += "\n".join(props) + "\n$$$$\n"
        blocks.append(block)
    return "".join(blocks).encode("utf-8")


def export_json(molecules, schema="genmap-1.0"):
    """Return JSON representation of molecules with schema information."""
    data = {"schema": schema, "molecules": molecules}
    return json.dumps(data, indent=2).encode("utf-8")


def _wrap_smiles(smiles):
    """Insert zero-width spaces to allow line wrapping."""
    return "\u200b".join(smiles)


def export_pdf(molecules):
    """Return a PDF listing the molecules' SMILES strings.

    When ReportLab is available, a proper PDF is generated. Otherwise a
    lightweight fallback is used which separates pages with form feed
    characters. Long SMILES are wrapped using zero-width spaces so PDF
    viewers can break lines when needed.
    """
    if canvas is None:
        # Fallback: plain text with form feed page breaks
        lines_per_page = 40
        lines = [f"{idx}. {_wrap_smiles(m['smiles'])}" for idx, m in enumerate(molecules, 1)]
        pages = ["\n".join(lines[i:i+lines_per_page]) for i in range(0, len(lines), lines_per_page)]
        return "\f".join(pages).encode("utf-8")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    for idx, mol in enumerate(molecules, 1):
        smiles_wrapped = _wrap_smiles(mol["smiles"])
        c.drawString(40, y, f"{idx}. {smiles_wrapped}")
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

import json

from modules.lims_connectors import export_sdf, export_json, export_pdf

from PyPDF2 import PdfReader


def test_export_sdf_contains_properties():
    molecules = [{"smiles": "CCO", "id": 1, "name": "ethanol"}]
    sdf_bytes = export_sdf(molecules)
    text = sdf_bytes.decode("utf-8")
    assert "> <id>" in text
    assert "ethanol" in text


def test_export_json_schema_and_properties():
    molecules = [{"smiles": "CCO", "id": 1, "name": "ethanol"}]
    json_bytes = export_json(molecules)
    data = json.loads(json_bytes.decode("utf-8"))
    assert data["schema"] == "genmap-1.0"
    assert data["molecules"][0]["name"] == "ethanol"


def test_export_pdf_paginates(tmp_path):
    molecules = [{"smiles": "C" * 100} for _ in range(60)]
    pdf_bytes = export_pdf(molecules)
    pdf_path = tmp_path / "mols.pdf"
    pdf_path.write_bytes(pdf_bytes)
    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) > 1

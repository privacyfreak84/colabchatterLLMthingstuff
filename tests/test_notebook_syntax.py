"""Static syntax checks for executable notebook cells."""
import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "03_chatterbox_custom_text_gradio.ipynb"


def test_custom_text_gradio_notebook_is_valid_json_and_python():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    code_cells = [
        (index, cell)
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]
    assert code_cells

    for index, cell in code_cells:
        source = "".join(cell.get("source", []))
        assert source.strip(), f"Code cell {index} is empty"
        ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{index}")

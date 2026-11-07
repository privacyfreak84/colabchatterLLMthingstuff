"""Static syntax and safety checks for executable notebook cells."""
import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "03_chatterbox_custom_text_gradio.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_sources(notebook):
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_custom_text_gradio_notebook_is_valid_json_and_python():
    notebook = load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    sources = code_sources(notebook)
    assert sources

    for index, source in enumerate(sources):
        assert source.strip(), f"Code cell {index} is empty"
        ast.parse(source, filename=f"{NOTEBOOK.name}:code-cell-{index}")


def test_google_drive_is_opt_in_and_not_required_for_generation():
    notebook = load_notebook()
    sources = code_sources(notebook)
    all_code = "\n\n".join(sources)
    setup = sources[0]
    output_setup = sources[3]
    generation = sources[4]

    assert "USE_GOOGLE_DRIVE = False" in setup
    assert 'if USE_GOOGLE_DRIVE:\n    drive.mount("/content/drive")' in setup
    assert 'if USE_GOOGLE_DRIVE else None' in all_code
    assert "if USE_GOOGLE_DRIVE and DRIVE_OUTPUT_DIR is not None:" in generation
    assert "temporary Colab storage" in generation
    assert "drive.mount(\"/content/drive\")" not in output_setup

def test_setup_selects_numpy_for_the_active_python_version_and_shows_progress():
    setup = code_sources(load_notebook())[0]

    assert '"numpy>=2.0.0" if sys.version_info >= (3, 13) else "numpy<2.0.0"' in setup
    assert "Python 3.13+ requires NumPy 2.x" in setup
    assert "Python 3.12 and older require NumPy 1.x" in setup
    assert '"--progress-bar", "on"' in setup
    assert "PROJECT_DIR" not in setup  # The notebook never used the cloned repository.
    assert '"NumPy 2.x is incompatible with this pinned setup"' not in setup


def test_reference_conditioning_profiles_keep_standard_and_offer_extended_windows():
    notebook = load_notebook()
    sources = code_sources(notebook)
    reference_helpers = sources[3]
    generation = sources[4]
    ui = sources[5]

    assert '"t3_prompt_seconds": 6' in reference_helpers
    assert '"s3gen_prompt_seconds": 10' in reference_helpers
    assert '"voice_encoder_max_seconds": 10' in reference_helpers
    assert '"t3_prompt_seconds": 10' in reference_helpers
    assert '"s3gen_prompt_seconds": 15' in reference_helpers
    assert '"voice_encoder_max_seconds": 30' in reference_helpers
    assert "MAX_REFERENCE_AUDIO_SECONDS = 30" in reference_helpers

    # The extended settings are applied only while reference conditionals are prepared.
    assert "model.ENC_COND_LEN = int(" in generation
    assert "model.DEC_COND_LEN = int(" in generation
    assert "model.t3.hp.speech_cond_prompt_len = int(" in generation
    assert "original_conditioning_limits" in generation
    assert ") = original_conditioning_limits" in generation
    assert '"reference_conditioning"' in generation

    assert "conditioning_profile_input = gr.Dropdown(" in ui
    assert "conditioning_profile_input," in ui

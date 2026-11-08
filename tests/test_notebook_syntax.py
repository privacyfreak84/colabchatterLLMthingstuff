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
    ui = sources[6]

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
    assert 'conditioning_settings["voice_encoder_max_seconds"]' in generation
    assert 'conditioning_settings["speaker_encoder_max_seconds"]' not in generation

    assert "conditioning_profile_input = gr.Dropdown(" in ui
    assert "conditioning_profile_input," in ui


def test_generation_audits_unknown_text_tokens_without_rewriting_input():
    notebook = load_notebook()
    generation = code_sources(notebook)[4]
    usage = next("".join(cell.get("source", [])) for cell in notebook["cells"] if "Usage notes and troubleshooting" in "".join(cell.get("source", [])))

    assert "def audit_text_tokenization(" in generation
    assert 'backend.token_to_id("[UNK]")' in generation
    assert "model.tokenizer.text_to_tokens(chunk, language_id=language_id)" in generation
    assert '"text_token_audit": text_token_audit' in generation
    assert "unknown [UNK] text token(s)" in generation
    assert "Mixed scripts / repeated phrases" in usage
    # This is diagnostic only: it must not silently replace or drop manuscript text.
    assert "text = text.replace" not in generation
    assert "chunk = chunk.replace" not in generation


def test_longform_batch_workflow_has_resume_logs_and_audiobook_exports():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch = sources[5]
    ui = sources[6]
    all_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "AUDIOBOOK_HELPER_COMMIT" in batch
    assert "def run_audiobook_job(" in batch
    assert "def preview_audiobook_plan(" in batch
    assert "create_or_resume_job(" in batch
    assert "verify_chunk_wav(" in batch
    assert "assemble_job_masters(" in batch
    assert "create_job_bundle(" in batch
    assert "force_rerender_chunks(" in batch
    assert "build_audio_qa_report(" in batch
    assert "append_app_log(" in batch
    assert "load_activity_logs(" in batch
    assert "list_recent_audiobook_jobs(" in batch
    assert "audit_text_tokenization(" in batch
    assert 'with gr.Tab("Audiobook Jobs")' in ui
    assert 'with gr.Tab("Logs and Recovery")' in ui
    assert "Preview section and chunk plan" in ui
    assert "Generate / resume audiobook job" in ui
    assert "approval_checkbox" in ui
    assert "batch_rerender_ids_input" in ui
    assert "batch_reroll_seed_input" in ui
    assert "batch_qa_output" in ui
    assert "List saved audiobook jobs" in ui
    assert "inspected_qa_output" in ui
    assert "allowed_paths=[str(LOCAL_OUTPUT_DIR)] + ([str(DRIVE_OUTPUT_DIR)]" in ui
    assert "full-book approval gate" in all_text.lower()
    quick_helpers = next(source for source in sources if "def split_text_into_chunks" in source)
    assert "longform.normalize_manuscript_text(text)" in quick_helpers


def test_audiobook_generator_yields_match_all_nine_gradio_outputs():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch_source = sources[5]
    ui_source = sources[6]

    batch_tree = ast.parse(batch_source)
    generator = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_audiobook_job"
    )
    yielded_tuple_sizes = []
    for node in ast.walk(generator):
        if isinstance(node, ast.Yield) and isinstance(node.value, ast.Tuple):
            yielded_tuple_sizes.append(len(node.value.elts))
    assert len(yielded_tuple_sizes) == 5
    assert set(yielded_tuple_sizes) == {9}

    ui_tree = ast.parse(ui_source)
    batch_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_batch_button"
    )
    outputs = next(keyword.value for keyword in batch_click.keywords if keyword.arg == "outputs")
    inputs = next(keyword.value for keyword in batch_click.keywords if keyword.arg == "inputs")
    assert isinstance(outputs, ast.List)
    assert len(outputs.elts) == 9
    assert isinstance(inputs, ast.List)
    assert len(inputs.elts) == len(generator.args.args) - 1  # progress is injected by Gradio

    preview_function = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "preview_audiobook_plan"
    )
    preview_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "preview_button"
    )
    preview_inputs = next(keyword.value for keyword in preview_click.keywords if keyword.arg == "inputs")
    assert isinstance(preview_inputs, ast.List)
    assert len(preview_inputs.elts) == len(preview_function.args.args)
    preview_outputs = next(keyword.value for keyword in preview_click.keywords if keyword.arg == "outputs")
    assert isinstance(preview_outputs, ast.List)
    assert len(preview_outputs.elts) == 3


def test_pdf_ingestion_and_wrapped_line_reflow_are_wired_into_audiobook_ui():
    notebook = load_notebook()
    sources = code_sources(notebook)
    setup = sources[0]
    batch = sources[5]
    ui = sources[6]
    all_text = "\\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert '"pypdf"' in setup
    assert "def read_audiobook_source(" in batch
    assert "longform.extract_pdf_manuscript(" in batch
    assert "longform.normalize_manuscript_text(" in batch
    assert "pdf_page_start" in batch and "pdf_page_end" in batch
    assert "text_paragraph_start" in batch and "text_paragraph_end" in batch
    assert "longform.select_text_paragraph_range(" in batch
    assert "source_pages=source_pages" in batch
    assert "source_pdf_bytes=source_pdf_bytes" in batch
    assert "pdf_extraction_report=ingestion_report" in batch
    assert "cleaned_text_preview_output" in ui
    assert "pdf_page_start_input" in ui and "pdf_page_end_input" in ui
    assert 'file_types=[".pdf", ".txt", ".md"]' in ui
    assert "punctuation is retained" in all_text.lower()
    assert "scanned/image-only" in all_text.lower()


def test_preview_callback_returns_plan_and_cleaned_text_preview():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch_source = sources[5]
    ui_source = sources[6]
    batch_tree = ast.parse(batch_source)
    preview_function = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "preview_audiobook_plan"
    )
    assert len(preview_function.args.args) == 12

    ui_tree = ast.parse(ui_source)
    preview_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "preview_button"
    )
    outputs = next(keyword.value for keyword in preview_click.keywords if keyword.arg == "outputs")
    assert isinstance(outputs, ast.List)
    assert len(outputs.elts) == 3


def test_gradio_ui_has_no_login_authentication():
    notebook = load_notebook()
    sources = code_sources(notebook)
    setup = sources[0]
    ui = sources[6]
    all_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "getpass" not in setup
    assert "AUTH_USERNAME" not in ui
    assert "AUTH_PASSWORD" not in ui
    assert "auth=(" not in ui
    assert "authentication is disabled" in all_text


def test_spoiler_free_accent_analysis_is_wired_into_notebook_ui():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch = sources[5]
    ui = sources[6]
    all_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "ACCENT_ANALYSIS_HELPER_COMMIT" in batch
    assert "load_accentcl_model(" in batch
    assert "weights_only=True" not in batch  # safe loading lives in the pinned helper
    assert "def run_accent_analysis(" in batch
    assert "ChatterboxMultilingualTTS.from_local(" in batch
    assert "def list_saved_accent_audio_choices(" in batch
    assert 'with gr.Tab("Accent Check")' in ui
    assert 'file_types=[".wav", ".mp3"]' in ui
    assert "Analyze accent without transcription" in ui
    assert "accent_segments_output" in ui
    assert "accent_json_output" in ui and "accent_csv_output" in ui
    assert 'max_file_size="2gb"' in ui
    assert "without transcribing it" in all_text.lower()
    assert "sentence text" in all_text.lower()


def test_accent_analysis_ui_event_input_and_output_arities_match_callback():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch_tree = ast.parse(sources[5])
    ui_tree = ast.parse(sources[6])
    callback = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_accent_analysis"
    )
    analyse_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "accent_analyse_button"
    )
    inputs = next(keyword.value for keyword in analyse_click.keywords if keyword.arg == "inputs")
    outputs = next(keyword.value for keyword in analyse_click.keywords if keyword.arg == "outputs")
    assert isinstance(inputs, ast.List) and len(inputs.elts) == len(callback.args.args) - 1
    assert isinstance(outputs, ast.List) and len(outputs.elts) == 4


def test_section_selection_and_atomic_rerender_controls_are_wired_into_batch_ui():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch = sources[5]
    ui = sources[6]
    all_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "selected_section_ids" in batch
    assert "longform.filter_plan_sections(" in batch
    assert "preserve_previous_renders" in batch
    assert "preserve_previous=bool(preserve_previous_renders)" in batch
    assert "chunk_ids_by_section" in batch
    assert "Unknown chunk/section ID" in batch
    assert "pending_replacement = chunk.pop(" in batch
    assert 'section_selection_input = gr.Dropdown(' in ui
    assert "multiselect=True" in ui
    assert 'label="Sections to speak (preview the source first; empty = all sections)"' in ui
    assert "batch_keep_previous_renders_input" in ui
    assert "selected_sections_master.wav" in all_text
    assert "atomically replaces the old active file" in all_text.lower()


def test_section_selection_generator_and_preview_ui_arities_match():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch_tree = ast.parse(sources[5])
    ui_tree = ast.parse(sources[6])

    generator = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_audiobook_job"
    )
    batch_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_batch_button"
    )
    batch_inputs = next(k.value for k in batch_click.keywords if k.arg == "inputs")
    assert isinstance(batch_inputs, ast.List)
    assert len(batch_inputs.elts) == len(generator.args.args) - 1

    preview = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "preview_audiobook_plan"
    )
    preview_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "preview_button"
    )
    preview_inputs = next(k.value for k in preview_click.keywords if k.arg == "inputs")
    preview_outputs = next(k.value for k in preview_click.keywords if k.arg == "outputs")
    assert isinstance(preview_inputs, ast.List)
    assert isinstance(preview_outputs, ast.List)
    assert len(preview_inputs.elts) == len(preview.args.args)
    assert len(preview_outputs.elts) == 3


def test_custom_output_name_is_wired_through_audiobook_generation_and_accent_scan():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch = sources[5]
    ui = sources[6]
    assert "custom_output_name: str" in batch
    assert "output_name=(custom_output_name or None)" in batch
    assert "batch_output_name_input = gr.Textbox(" in ui
    assert "batch_output_name_input," in ui
    assert "*/*_whole_book_master.wav" in batch
    assert "*/*_selected_sections_master.wav" in batch
    assert "output_name" in batch


def test_accent_factorization_is_wired_into_quick_tts_too():
    notebook = load_notebook()
    sources = code_sources(notebook)
    quick = sources[4]
    ui = sources[6]
    quick_tree = ast.parse(quick)
    ui_tree = ast.parse(ui)

    callback = next(
        node for node in quick_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_speech"
    )
    click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "generate_button"
    )
    inputs = next(keyword.value for keyword in click.keywords if keyword.arg == "inputs")
    assert isinstance(inputs, ast.List)
    assert len(inputs.elts) == len(callback.args.args) - 1
    assert "uploaded_accent_reference: str | None" in quick
    assert "accent_prompt_path=" in quick
    assert "accent_reference" in quick
    assert "accent_reference_audio_input = gr.Audio(" in ui


def test_accent_factorization_patch_is_pinned_and_wired_into_audiobook_jobs():
    notebook = load_notebook()
    sources = code_sources(notebook)
    setup = sources[0]
    batch = sources[5]
    ui = sources[6]

    assert "ACCENT_FACTORIZATION_PATCH_COMMIT = \"dd97c3241b603b142b86a92bfcfdfd45f4ecf4c8\"" in setup
    assert "install_chatterbox_accent_factorization(" in setup and "    ChatterboxMultilingualTTS" in setup
    assert "accent_prompt_path=" in batch
    assert "uploaded_accent_reference: str | None" in batch
    assert "accent_prompt_start_seconds" in batch
    assert "batch_accent_prompt_start_seconds_input" in ui
    assert "accent_reference_source_sha256" in batch
    assert "accent_reference_mode" in batch
    assert "batch_accent_reference_audio_input = gr.Audio(" in ui
    assert "batch_accent_reference_audio_input," in ui
    assert "The narrator reference supplies speaker identity" in ui

    # The patched model must factor the two input paths before decoding, not score
    # or rewrite generated audio afterwards.
    patch_source = (REPO_ROOT / "tools" / "chatterbox_accent_factorization.py").read_text(encoding="utf-8")
    assert "self.ve.embeds_from_wavs([identity_16k]" in patch_source
    assert "self.s3gen.embed_ref(" in patch_source
    assert "style_prompt_24k" in patch_source
    assert 's3gen_ref_dict["embedding"] = self.s3gen.speaker_encoder.inference(identity_for_s3)' in patch_source
    assert "accent_prompt_path" in patch_source


def test_non_pdf_inputs_expose_paragraph_range_controls():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch = sources[5]
    ui = sources[6]
    assert "text_paragraph_start_input" in ui
    assert "text_paragraph_end_input" in ui
    assert "select_text_paragraph_range(" in batch
    assert "selected_paragraph_start" in batch
    assert "selected_paragraph_end" in batch
    assert "TXT/Markdown/pasted text has no inherent page numbers" in batch

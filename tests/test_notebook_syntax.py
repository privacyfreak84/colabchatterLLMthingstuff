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
    assert "assemble_selected_chunk_wavs(" in batch
    assert "def assemble_audiobook_wavs(" in batch
    assert "def play_saved_audiobook_audio(" in batch
    assert "create_job_bundle(" in batch
    assert "force_rerender_chunks(" in batch
    assert "build_audio_qa_report(" in batch
    assert "append_app_log(" in batch
    assert "load_activity_logs(" in batch
    assert "list_recent_audiobook_jobs(" in batch
    assert "audit_text_tokenization(" in batch
    assert 'with gr.Tab("Audiobook Jobs")' in ui
    assert 'with gr.Tab("Logs and Recovery")' in ui
    assert "Preview manuscript and chunk plan" in ui
    assert "Generate / resume audiobook" in ui
    assert 'with gr.Tab("Create / Resume")' in ui
    assert 'with gr.Tab("Listen & Assemble")' in ui
    assert "approval_checkbox" in ui
    assert "batch_rerender_ids_input" in ui
    assert "batch_reroll_seed_input" in ui
    assert "batch_qa_output" in ui
    assert "Refresh saved jobs" in ui
    assert "Latest spoken chunk" in ui
    assert "Previously generated WAV" in ui
    assert "Use all completed chunks in this job" in ui
    assert "Nothing is combined into a master automatically" in ui
    assert "batch_master_output" not in ui
    assert "batch_section_masters_output" not in ui
    assert "inspected_qa_output" in ui
    assert "allowed_paths=[str(LOCAL_OUTPUT_DIR)] + ([str(DRIVE_OUTPUT_DIR)]" in ui
    assert "full-book approval gate" in all_text.lower()
    quick_helpers = next(source for source in sources if "def split_text_into_chunks" in source)
    assert "longform.normalize_manuscript_text(text)" in quick_helpers


def test_audiobook_generator_yields_match_all_eight_gradio_outputs():
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
    assert set(yielded_tuple_sizes) == {8}

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
    assert len(outputs.elts) == 8
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
    assert '"*/chunks/*.wav"' in batch
    assert '"*/assemblies/*.wav"' in batch
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
    assert 'label="Sections to speak (empty selection = all)"' in ui
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
    assert [item.id for item in preview_inputs.elts] == [
        "batch_file_input",
        "batch_text_input",
        "pdf_page_start_input",
        "pdf_page_end_input",
        "text_paragraph_start_input",
        "text_paragraph_end_input",
        "batch_chunk_chars_input",
        "speak_headings_input",
        "batch_chunk_pause_input",
        "batch_paragraph_pause_input",
        "batch_section_pause_input",
        "section_selection_input",
    ]
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

    assert "ACCENT_FACTORIZATION_PATCH_COMMIT = \"9ea38a1d6708bf632f3de185eda1b51617843b0a\"" in setup
    assert "install_chatterbox_accent_factorization(" in setup and "    ChatterboxMultilingualTTS" in setup
    assert "accent_prompt_path=" in batch
    assert "uploaded_accent_reference: str | None" in batch
    assert "accent_prompt_start_seconds" in batch
    assert "batch_accent_prompt_start_seconds_input" in ui
    assert "accent_reference_source_sha256" in batch
    assert "accent_reference_mode" in batch
    assert "batch_accent_reference_audio_input = gr.Audio(" in ui
    assert "batch_accent_reference_audio_input," in ui
    assert "The narrator recording supplies speaker identity" in ui

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



def test_pitch_preserving_speech_rate_control_is_wired_to_quick_and_audiobook_generation():
    notebook = load_notebook()
    sources = code_sources(notebook)
    quick = next(source for source in sources if "def generate_speech(" in source)
    batch = next(source for source in sources if "def run_audiobook_job(" in source)
    ui = next(source for source in sources if 'with gr.Blocks(' in source)
    helper_loader = next(source for source in sources if "SPEECH_RATE_HELPER_COMMIT" in source)

    assert 'SPEECH_RATE_HELPER_COMMIT = "cbbbe10084b5a2af6c5297a9b335e44f20d0f3c8"' in helper_loader
    assert "speech_rate.adjust_speech_rate(" in quick
    assert "speech_rate.adjust_speech_rate(" in batch
    assert '"speech_rate": speech_rate_value' in quick
    assert '"speech_rate": speech_rate_value' in batch
    assert '"speech_rate_processor": speech_rate.PROCESSOR_ID' in batch
    assert "model_duration_seconds" in quick and "model_duration_seconds" in batch
    assert "0.85x = slower" in ui
    assert "does not lower the voice pitch or add reverb/echo" in ui
    assert "A different narration rate requires a new Job ID" in ui

    quick_tree = ast.parse(quick)
    quick_callback = next(
        node for node in quick_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_speech"
    )
    quick_arg_names = [arg.arg for arg in quick_callback.args.args]
    assert quick_arg_names[quick_arg_names.index("conditioning_profile") + 1] == "speech_rate_value"

    batch_tree = ast.parse(batch)
    batch_callback = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_audiobook_job"
    )
    batch_arg_names = [arg.arg for arg in batch_callback.args.args]
    assert batch_arg_names[batch_arg_names.index("conditioning_profile") + 1] == "speech_rate_value"

    ui_tree = ast.parse(ui)
    quick_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "generate_button"
    )
    quick_inputs = next(keyword.value for keyword in quick_click.keywords if keyword.arg == "inputs")
    quick_names = [item.id for item in quick_inputs.elts]
    assert quick_names[quick_names.index("conditioning_profile_input") + 1] == "speech_rate_input"

    batch_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_batch_button"
    )
    batch_inputs = next(keyword.value for keyword in batch_click.keywords if keyword.arg == "inputs")
    batch_names = [item.id for item in batch_inputs.elts]
    assert batch_names[batch_names.index("batch_conditioning_profile_input") + 1] == "batch_speech_rate_input"



def test_audiobook_saved_playback_and_manual_assembly_ui_callbacks_match():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch = sources[5]
    ui = sources[6]
    batch_tree = ast.parse(batch)
    ui_tree = ast.parse(ui)

    assembly = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "assemble_audiobook_wavs"
    )
    assert len(assembly.args.args) == 5  # progress is injected by Gradio
    refresh = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "refresh_audiobook_job_choices"
    )
    assert len(refresh.args.args) == 0
    assert "list_audiobook_audio_choices(" in batch
    assert "longform._safe_relative_path(" in batch
    assert "def _audiobook_job_sample_rate(" in batch
    assert 'settings.get("sample_rate", 24000)' in batch
    assert "sample_rate=int(model.sr)" not in batch[
        batch.index("def list_audiobook_job_choices("):
        batch.index("def load_activity_logs(")
    ]

    assemble_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "assembly_button"
    )
    inputs = next(k.value for k in assemble_click.keywords if k.arg == "inputs")
    outputs = next(k.value for k in assemble_click.keywords if k.arg == "outputs")
    assert isinstance(inputs, ast.List) and len(inputs.elts) == len(assembly.args.args) - 1
    assert [item.id for item in inputs.elts] == [
        "saved_audiobook_job_input",
        "assembly_chunk_selection_input",
        "assembly_all_chunks_input",
        "assembly_output_name_input",
    ]
    assert isinstance(outputs, ast.List) and len(outputs.elts) == 5

    saved_audio_change = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "change"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "saved_audiobook_audio_input"
    )
    audio_inputs = next(k.value for k in saved_audio_change.keywords if k.arg == "inputs")
    audio_outputs = next(k.value for k in saved_audio_change.keywords if k.arg == "outputs")
    assert isinstance(audio_inputs, ast.List) and len(audio_inputs.elts) == 2
    assert isinstance(audio_outputs, ast.List) and len(audio_outputs.elts) == 2

    # A completed generation no longer calls automatic whole-book/section assembly.
    generator = next(
        node for node in batch_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_audiobook_job"
    )
    called_names = {
        node.func.attr
        for node in ast.walk(generator)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "assemble_job_masters" not in called_names
    assert "assemble_selected_chunk_wavs" not in called_names
    assert "create_job_bundle" in called_names


def test_latest_audio_player_path_is_initialized_before_first_streamed_update():
    notebook = load_notebook()
    batch = code_sources(notebook)[5]
    tree = ast.parse(batch)
    generator = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_audiobook_job"
    )

    latest_audio_assignments = [
        node.lineno
        for node in ast.walk(generator)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "latest_audio_path"
            for target in node.targets
        )
    ]
    yield_nodes = [
        node for node in ast.walk(generator)
        if isinstance(node, ast.Yield)
    ]

    assert latest_audio_assignments, "latest_audio_path must be initialized"
    assert yield_nodes, "the generator must stream at least one update"
    assert min(latest_audio_assignments) < min(node.lineno for node in yield_nodes), (
        "latest_audio_path must be initialized before the first yield; "
        "otherwise a new job fails before the first chunk is generated"
    )

def test_audiobook_and_accent_helpers_are_pinned_to_current_validated_implementations():
    notebook = load_notebook()
    batch = next(
        source for source in code_sources(notebook)
        if "AUDIOBOOK_HELPER_COMMIT" in source
        and "ACCENT_ANALYSIS_HELPER_COMMIT" in source
    )

    expected_audiobook_commit = "90c6070f4520cceee7fe475e16ad3485aee13989"
    expected_accent_commit = "7325ce585a17f2c7184b7dbce4f86a40f6acac77"
    assert f'AUDIOBOOK_HELPER_COMMIT = "{expected_audiobook_commit}"' in batch
    assert f'ACCENT_ANALYSIS_HELPER_COMMIT = "{expected_accent_commit}"' in batch

    # The local source-of-truth helpers must contain the APIs and verification
    # used by those immutable notebook pins.
    longform_source = (
        REPO_ROOT / "tools" / "longform_audiobook.py"
    ).read_text(encoding="utf-8")
    accent_source = (
        REPO_ROOT / "tools" / "accent_analysis.py"
    ).read_text(encoding="utf-8")

    assert "def assemble_selected_chunk_wavs(" in longform_source
    assert "ACCENTCL_CHECKPOINT_REVISION" in accent_source
    assert "ACCENTCL_CHECKPOINT_SIZE_BYTES" in accent_source
    assert "ACCENTCL_CHECKPOINT_SHA256" in accent_source
    assert "def load_verified_accent_checkpoint(" in accent_source

def test_saved_chunk_rerender_picker_exposes_exact_chunk_choices_and_keeps_selection_only():
    notebook = load_notebook()
    sources = code_sources(notebook)
    batch = sources[5]
    ui = sources[6]

    assert "def refresh_rerender_chunk_choices(" in batch
    assert "longform.verify_chunk_wav(" in batch
    assert "chunk.get(\"text\", \"\")" in batch
    assert "complete, verified" in batch
    assert "complete, WAV needs repair" in batch
    assert "whole section" in batch
    assert "rerender_values = list(rerender_chunk_ids)" in batch

    # The UI presents IDs as a real multiselect rather than requiring manual typing.
    assert "batch_rerender_ids_input = gr.Dropdown(" in ui
    assert "multiselect=True" in ui
    assert "batch_rerender_refresh_button = gr.Button(" in ui
    assert "batch_rerender_refresh_button.click(" in ui
    assert "inputs=[batch_job_id_input]" in ui
    assert "Only the selected chunks are regenerated" in ui

    ui_tree = ast.parse(ui)
    refresh_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "batch_rerender_refresh_button"
    )
    outputs = next(k.value for k in refresh_click.keywords if k.arg == "outputs")
    assert isinstance(outputs, ast.List) and len(outputs.elts) == 2

    run_click = next(
        node for node in ast.walk(ui_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "click"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_batch_button"
    )
    inputs = next(k.value for k in run_click.keywords if k.arg == "inputs")
    assert isinstance(inputs, ast.List)
    assert "batch_rerender_ids_input" in [item.id for item in inputs.elts]


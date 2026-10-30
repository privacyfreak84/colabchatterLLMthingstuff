import pytest

from tools.chatterbox_voice_bridge import build_bridge_task_specs


TERMS = {
    "Avalokiteshvara": "अवलोकितेश्वर",
    "Maitreya": "मैत्रेय",
    "Manjushri": "मञ्जुश्री",
    "Vijnanavada": "विज्ञानवाद",
}
BASE = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "min_p": 0.05,
    "top_p": 0.95,
}


def test_bridge_builder_creates_six_deterministic_tasks_per_term():
    tasks = build_bridge_task_specs(TERMS, BASE)

    assert len(tasks) == 24
    assert len({task["task_id"] for task in tasks}) == 24
    assert [task["term"] for task in tasks[:6]] == ["Avalokiteshvara"] * 6
    assert tasks[0]["mode"] == "hindi_source"
    assert tasks[2]["mode"] == "hybrid"
    assert tasks[4]["kind"] == "voice_conversion"
    assert tasks[4]["source_task_id"] == "B001_HI_SOURCE_DEFAULT"
    assert tasks[5]["source_task_id"] == "B001_HI_SOURCE_PACE"


def test_bridge_builder_uses_same_language_and_english_decoder_modes():
    tasks = build_bridge_task_specs(TERMS, BASE)
    by_id = {task["task_id"]: task for task in tasks}

    assert by_id["B002_HI_SOURCE_DEFAULT"]["language_id"] == "hi"
    assert by_id["B002_HYBRID_DEFAULT"]["language_id"] == "hi"
    assert by_id["B002_HI_SOURCE_DEFAULT"]["text"] == "इस शब्द का उच्चारण मैत्रेय है।"
    assert by_id["B002_HYBRID_DEFAULT"]["text"] == by_id["B002_HI_SOURCE_DEFAULT"]["text"]


def test_pace_variant_changes_only_the_documented_pacing_controls():
    tasks = build_bridge_task_specs(TERMS, BASE)
    by_id = {task["task_id"]: task for task in tasks}

    baseline = by_id["B001_HI_SOURCE_DEFAULT"]["generation_parameters"]
    pace = by_id["B001_HI_SOURCE_PACE"]["generation_parameters"]

    assert baseline == BASE
    assert pace == {**BASE, "exaggeration": 0.7, "cfg_weight": 0.3}
    assert by_id["B001_HYBRID_PACE"]["generation_parameters"] == pace


def test_builder_does_not_mutate_callers_parameters():
    source = dict(BASE)
    build_bridge_task_specs(TERMS, source)
    assert source == BASE


def test_builder_rejects_wrong_term_count():
    with pytest.raises(ValueError, match="exactly four"):
        build_bridge_task_specs({"one": "एक"}, BASE)


@pytest.mark.parametrize(
    "forms",
    [
        {"one": "एक", "two": "दो", "three": "तीन", "four": ""},
        {"one": "एक", "two": "दो", "three": "तीन", "": "चार"},
    ],
)
def test_builder_rejects_blank_terms_or_forms(forms):
    with pytest.raises(ValueError, match="non-empty"):
        build_bridge_task_specs(forms, BASE)

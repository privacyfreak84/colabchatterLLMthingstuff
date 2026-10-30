from pronunciation_audition import build_audition_tasks, preflight_tasks
import pytest

TERMS = [
    ("P001", "Avalokiteshvara", "Then Avalokiteshvara spoke to the Buddha."),
    ("P002", "Avatamsaka", "The Avatamsaka Sutra is discussed here."),
    ("P003", "Buddha", "The Buddha spoke."),
    ("P004", "Gautama", "Gautama Buddha taught this."),
    ("P005", "Maitreya", "Maitreya spoke to the Buddha."),
    ("P006", "Manjushri", "Manjushri asked a question."),
    ("P007", "nirvana", "They discussed nirvana."),
    ("P008", "samsara", "The term samsara appears here."),
    ("P009", "Sandhinirmochana-sutra", "The Sandhinirmochana-sutra is a source."),
    ("P010", "Subhuti", "Subhuti answered."),
    ("P011", "sutra", "The sutra is translated."),
    ("P012", "Vijnanavada", "Vijnanavada is a tradition."),
    ("P013", "Yogachara", "Yogachara is the practice."),
]


def rows():
    return [
        {"audition_id": i, "source_term": term, "occurrences": "1", "pages": "2",
         "context_prompt": context}
        for i, term, context in TERMS
    ]


class FakeTokenizerInner:
    def get_vocab(self):
        return {"[UNK]": 0, "a": 1}


class FakeTokenizer:
    def __init__(self, unknown_language=None):
        self.tokenizer = FakeTokenizerInner()
        self.unknown_language = unknown_language

    def text_to_tokens(self, text, language_id):
        return [[0, 1]] if language_id == self.unknown_language else [[1, 1]]


class FakeModel:
    def __init__(self, unknown_language=None):
        self.tokenizer = FakeTokenizer(unknown_language)


def test_expected_task_count_and_variants():
    tasks = build_audition_tasks(rows())
    assert len(tasks) == 36
    assert sum(x["variant"] == "source_english" for x in tasks) == 13
    assert sum(x["variant"] == "iast_english" for x in tasks) == 10
    assert sum(x["variant"] == "devanagari_hindi" for x in tasks) == 13


def test_iast_variant_replaces_only_the_target_term():
    tasks = build_audition_tasks(rows())
    task = next(x for x in tasks if x["task_id"] == "P001_EN_IAST")
    assert task["candidate_form"] == "Avalokiteśvara"
    assert task["text"] == "Then Avalokiteśvara spoke to the Buddha."
    assert task["language_id"] == "en"


def test_identical_iast_forms_do_not_create_duplicate_audio():
    tasks = build_audition_tasks(rows())
    buddha = [x for x in tasks if x["audition_id"] == "P003"]
    assert {x["variant"] for x in buddha} == {"source_english", "devanagari_hindi"}


def test_devanagari_variant_uses_hindi_language_id():
    tasks = build_audition_tasks(rows())
    task = next(x for x in tasks if x["task_id"] == "P006_HI_DEVANAGARI")
    assert task["language_id"] == "hi"
    assert "मञ्जुश्री" in task["text"]
    assert task["reference_language"] == "hi"


def test_hyphenated_book_term_replacement_is_exact():
    tasks = build_audition_tasks(rows())
    task = next(x for x in tasks if x["task_id"] == "P009_EN_IAST")
    assert task["text"] == "The Saṃdhinirmocana-sūtra is a source."


def test_missing_target_term_in_context_is_rejected():
    data = rows()
    data[0]["context_prompt"] = "The source passage omits the target name."
    with pytest.raises(ValueError, match="is absent"):
        build_audition_tasks(data)


def test_tokenizer_preflight_marks_unknown_tasks_without_generating():
    tasks = build_audition_tasks(rows())
    checked = preflight_tasks(FakeModel(unknown_language="hi"), tasks, lambda x: x)
    assert all(x["status"] == "ready" for x in checked if x["language_id"] == "en")
    assert all(x["status"] == "skipped_unknown_tokens" for x in checked if x["language_id"] == "hi")
    assert all(x["unknown_token_count"] == 1 for x in checked if x["language_id"] == "hi")


def test_duplicate_audition_ids_are_rejected():
    data = rows()
    data[1]["audition_id"] = data[0]["audition_id"]
    with pytest.raises(ValueError, match="Duplicate audition_id"):
        build_audition_tasks(data)

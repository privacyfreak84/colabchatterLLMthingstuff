from types import SimpleNamespace

import pytest

from tools.manuscript_heading_reflow import (
    install_chapter_heading_reflow,
    reflow_wrapped_chapter_headings,
)
from tools.longform_audiobook import normalize_manuscript_text


def test_joins_one_line_wrapped_chapter_title_without_flattening_body():
    wrapped = (
        "Chapter 2. Characteristics of Ultimate\n"
        "Truth\n"
        "\n"
        "At that time Logical Questioner, in the presence of the Buddha, asked..."
    )
    repaired = reflow_wrapped_chapter_headings(wrapped)
    assert repaired == (
        "Chapter 2. Characteristics of Ultimate Truth\n"
        "\n"
        "At that time Logical Questioner, in the presence of the Buddha, asked..."
    )
    assert normalize_manuscript_text(repaired) == (
        "Chapter 2. Characteristics of Ultimate Truth\n\n"
        "At that time Logical Questioner, in the presence of the Buddha, asked..."
    )


def test_joins_multi_line_chapter_title():
    wrapped = (
        "Chapter 2\n"
        "Characteristics of Ultimate\n"
        "Truth\n"
        "\n"
        "At that time..."
    )
    repaired = reflow_wrapped_chapter_headings(wrapped)
    assert repaired.startswith("Chapter 2 Characteristics of Ultimate Truth\n\n")


def test_does_not_join_first_prose_line_after_chapter_heading():
    text = (
        "Chapter 2. Characteristics of Ultimate\n"
        "At that time Logical Questioner, in the presence of the Buddha, asked..."
    )
    assert reflow_wrapped_chapter_headings(text) == text
    assert normalize_manuscript_text(text) == (
        "Chapter 2. Characteristics of Ultimate\n\n"
        "At that time Logical Questioner, in the presence of the Buddha, asked..."
    )


def test_does_not_change_unrelated_paragraphs_or_non_chapter_headings():
    text = "Preface\nA short opening thought\n\nFirst paragraph."
    assert reflow_wrapped_chapter_headings(text) == text


def test_does_not_join_sentence_like_line():
    text = "Chapter 3: The Path\nThe Buddha spoke."
    assert reflow_wrapped_chapter_headings(text) == text


def test_installs_on_longform_helper_and_is_idempotent(monkeypatch):
    helper = SimpleNamespace(normalize_manuscript_text=normalize_manuscript_text)
    original = helper.normalize_manuscript_text
    assert install_chapter_heading_reflow(helper) == "installed"
    patched = helper.normalize_manuscript_text
    assert patched is not original
    assert install_chapter_heading_reflow(helper) == "already installed"
    assert helper.normalize_manuscript_text is patched
    assert helper.normalize_manuscript_text(
        "Chapter 2. Characteristics of Ultimate\nTruth\n\nAt that time..."
    ) == "Chapter 2. Characteristics of Ultimate Truth\n\nAt that time..."

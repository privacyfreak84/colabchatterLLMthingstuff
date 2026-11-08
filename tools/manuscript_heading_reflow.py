"""Conservative repair for chapter headings split across PDF visual lines.

This module wraps the existing manuscript normalizer instead of changing the pinned
long-form helper, preserving existing audiobook job compatibility.
"""
from __future__ import annotations

import re
from typing import Any


_CHAPTER_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:chapter|part|book)\s+"
    r"(?:[0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"""[.!?][\"'”’)}\]]*\s*$""")
_TITLE_CONNECTORS = frozenset({
    "a", "an", "the", "of", "and", "or", "nor", "to", "for", "in",
    "on", "at", "by", "from", "with", "as", "into", "over", "under",
    "but", "upon", "through", "between", "without", "within",
})


def _is_title_fragment(line: str) -> bool:
    """Return True for a short, unpunctuated line that looks like title case."""
    candidate = re.sub(r"[ \t]+", " ", line).strip()
    if not candidate or len(candidate) > 100 or len(candidate.split()) > 10:
        return False
    if _SENTENCE_END.search(candidate):
        return False

    words = re.findall(r"[^\W\d_][\w’'-]*", candidate, flags=re.UNICODE)
    if not words:
        return False

    significant_words = [word for word in words if word.casefold() not in _TITLE_CONNECTORS]
    return bool(significant_words) and all(word[0].isupper() for word in significant_words)


def reflow_wrapped_chapter_headings(text: str) -> str:
    """Join obvious wrapped Chapter/Part/Book title fragments before normalization.

    The repair is deliberately narrow: it only runs after an explicit numbered
    Chapter/Part/Book heading and only joins short, unpunctuated, title-case lines.
    Ordinary prose such as "At that time..." is not treated as a title continuation.
    """
    if not isinstance(text, str):
        raise TypeError("Manuscript text must be a string.")

    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    lines = normalized_newlines.split("\n")
    repaired: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not _CHAPTER_HEADING.match(line.strip()):
            repaired.append(line)
            index += 1
            continue

        fragments: list[str] = []
        lookahead = index + 1
        while lookahead < len(lines):
            candidate = lines[lookahead].strip()
            if not _is_title_fragment(candidate):
                break
            fragments.append(candidate)
            lookahead += 1

        if fragments:
            repaired.append(line.rstrip() + " " + " ".join(fragments))
            index = lookahead
        else:
            repaired.append(line)
            index += 1

    return "\n".join(repaired)


def install_chapter_heading_reflow(longform_module: Any) -> str:
    """Install the cleanup wrapper on a loaded longform helper, safely and once."""
    normalizer = getattr(longform_module, "normalize_manuscript_text", None)
    if not callable(normalizer):
        raise TypeError("The supplied long-form helper has no callable manuscript normalizer.")
    if getattr(normalizer, "_chapter_heading_reflow_installed", False):
        return "already installed"

    original_normalizer = normalizer

    def patched_normalizer(text: str) -> str:
        return original_normalizer(reflow_wrapped_chapter_headings(text))

    patched_normalizer._chapter_heading_reflow_installed = True
    patched_normalizer._original_normalizer = original_normalizer
    longform_module.normalize_manuscript_text = patched_normalizer
    return "installed"

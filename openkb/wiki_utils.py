"""Shared Markdown section-manipulation utilities for wiki pages.

Used by both the compiler pipeline and the entity writer to manipulate
H2 sections in wiki pages (insert entries, ensure sections exist, etc.).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def iter_h2_headings(lines: list[str]) -> list[tuple[int, str]]:
    """Return ``[(line_index, normalized_heading), ...]`` for every ATX H2.

    A line counts as H2 when it starts with ``"## "`` (two hashes + space).
    ``normalized_heading`` is the line with trailing whitespace stripped, so
    ``"## Documents "`` normalizes to ``"## Documents"`` — letting callers
    use exact-string comparison without tripping on stray whitespace.
    """
    return [
        (i, line.rstrip())
        for i, line in enumerate(lines)
        if line.startswith("## ")
    ]


def get_section_bounds(lines: list[str], heading: str) -> tuple[int, int] | None:
    """Return the [start, end) bounds for a Markdown H2 section.

    Uses :func:`iter_h2_headings` so the same H2 detection that finds the
    target heading also determines the section's end (the next H2). A
    drifted ``"## Documents "`` matches ``"## Documents"`` because both
    sides are normalized.
    """
    headings = iter_h2_headings(lines)
    for k, (idx, normalized) in enumerate(headings):
        if normalized == heading:
            start = idx + 1
            end = headings[k + 1][0] if k + 1 < len(headings) else len(lines)
            return start, end
    return None


def ensure_h2_section(lines: list[str], heading: str) -> None:
    """Ensure an H2 section ``heading`` exists in ``lines``; append if missing.

    Recovers from hand-edited or drifted wiki files where the expected
    section was removed or renamed — without this, downstream inserts would
    silently no-op and entries would be dropped.
    """
    if get_section_bounds(lines, heading) is not None:
        return
    logger.warning(
        "Wiki page is missing %r section; appending it. "
        "Check whether the file was hand-edited away from the canonical layout.",
        heading,
    )
    while lines and lines[-1] == "":
        lines.pop()
    if lines:
        lines.append("")
    lines.append(heading)
    lines.append("")


def section_contains_link(lines: list[str], heading: str, link: str) -> bool:
    """Check whether an index entry already exists inside the named section."""
    bounds = get_section_bounds(lines, heading)
    if bounds is None:
        return False

    start, end = bounds
    entry_prefix = f"- {link}"
    return any(line.startswith(entry_prefix) for line in lines[start:end])


def replace_section_entry(lines: list[str], heading: str, link: str, entry: str) -> bool:
    """Replace the first matching entry within a specific section."""
    bounds = get_section_bounds(lines, heading)
    if bounds is None:
        return False

    start, end = bounds
    entry_prefix = f"- {link}"
    for i in range(start, end):
        if lines[i].startswith(entry_prefix):
            lines[i] = entry
            return True
    return False


def insert_section_entry(lines: list[str], heading: str, entry: str) -> bool:
    """Insert a new entry at the top of a specific section.

    Returns True if the entry was inserted, False if the section was not found.
    """
    bounds = get_section_bounds(lines, heading)
    if bounds is None:
        return False

    start, _ = bounds
    lines.insert(start, entry)
    return True

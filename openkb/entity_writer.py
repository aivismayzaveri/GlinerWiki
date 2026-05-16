"""Entity wiki page writer for OpenKB.

Creates and updates entity pages in wiki/entities/, maintains the entity
index grouped by type, and adds bidirectional backlinks between entities
and summaries/concepts.
"""
from __future__ import annotations

import logging
from pathlib import Path

from openkb.entity_extractor import (
    ENTITY_TYPES,
    MergedEntity,
    _sanitize_entity_slug,
    _normalize_name,
)
from openkb.wiki_utils import ensure_h2_section, insert_section_entry


def _sanitize_concept_slug(name: str) -> str:
    """Convert a concept name to a safe filename slug."""
    import unicodedata
    import re
    s = unicodedata.normalize("NFKC", name)
    s = re.sub(r'[^\w\-]', '-', s).strip("-")
    return s or "unnamed-concept"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity page I/O
# ---------------------------------------------------------------------------

def _read_entity_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from an entity page. Returns (metadata, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].strip()
    body = text[end + 3:].lstrip("\n")
    meta = {}
    for line in fm_block.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "aliases":
                # Parse YAML list: [a, b, c]
                if val.startswith("[") and val.endswith("]"):
                    meta[key] = [s.strip() for s in val[1:-1].split(",") if s.strip()]
                else:
                    meta[key] = [val] if val else []
            elif key == "sources":
                if val.startswith("[") and val.endswith("]"):
                    meta[key] = [s.strip() for s in val[1:-1].split(",") if s.strip()]
                else:
                    meta[key] = [val] if val else []
            else:
                meta[key] = val
    return meta, body


def _build_entity_page(
    entity: MergedEntity,
    existing_text: str = "",
) -> str:
    """Build or update an entity page with merged metadata.

    If existing_text is provided, merges aliases and appends new sources
    rather than overwriting.
    """
    slug = _sanitize_entity_slug(entity.canonical_name)
    source_link = f"summaries/{entity.sources[0]}" if entity.sources else ""

    if existing_text:
        existing_meta, existing_body = _read_entity_frontmatter(existing_text)
        # Merge aliases
        old_aliases = set(existing_meta.get("aliases", []))
        new_aliases = set(entity.aliases)
        all_aliases = sorted(old_aliases | new_aliases)

        # Merge sources
        old_sources = set(existing_meta.get("sources", []))
        new_sources = {f"summaries/{s}" for s in entity.sources}
        all_sources = sorted(old_sources | new_sources)

        # Keep existing description if LLM didn't provide one
        description = entity.description or existing_meta.get("brief", "")

        # Rebuild frontmatter
        fm_lines = [f"type: {entity.entity_type}"]
        if all_aliases:
            fm_lines.append(f"aliases: [{', '.join(all_aliases)}]")
        if all_sources:
            fm_lines.append(f"sources: [{', '.join(all_sources)}]")
        if description:
            fm_lines.append(f"brief: {description}")
        frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n"

        # If body already has content, keep it but add new mention if needed
        if source_link and f"[[{source_link}]]" not in existing_body:
            # Append to Mentions section if it exists, otherwise keep body as-is
            if "## Mentions" in existing_body:
                existing_body = existing_body.rstrip() + f"\n- [[{source_link}]]"
            body = existing_body
        else:
            body = existing_body

        # Add LLM-inferred related concepts
        if entity.related_concepts:
            lines = body.split("\n")
            ensure_h2_section(lines, "## Related Concepts")
            for concept_name in entity.related_concepts:
                concept_slug = _sanitize_concept_slug(concept_name)
                entry = f"- [[concepts/{concept_slug}]]"
                if entry not in body:
                    insert_section_entry(lines, "## Related Concepts", entry)
            body = "\n".join(lines)

        return frontmatter + "\n" + body

    # New entity page
    fm_lines = [f"type: {entity.entity_type}"]
    if entity.aliases:
        fm_lines.append(f"aliases: [{', '.join(entity.aliases)}]")
    if entity.sources:
        source_links = [f"summaries/{s}" for s in entity.sources]
        fm_lines.append(f"sources: [{', '.join(source_links)}]")
    if entity.description:
        fm_lines.append(f"brief: {entity.description}")
    frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n"

    # Build body
    heading = entity.canonical_name
    body = f"\n# {heading}\n"
    if entity.description:
        body += f"\n{entity.description}\n"

    body += "\n## Mentions\n"
    if source_link:
        body += f"- [[{source_link}]]\n"

    body += "\n## Related Entities\n"

    # LLM-inferred related concepts
    body += "\n## Related Concepts\n"
    for concept_name in entity.related_concepts:
        concept_slug = _sanitize_concept_slug(concept_name)
        body += f"- [[concepts/{concept_slug}]]\n"

    return frontmatter + body


def write_entity_pages(
    wiki_dir: Path,
    entities: list[MergedEntity],
    doc_name: str,
) -> list[str]:
    """Create or update entity pages in wiki/entities/.

    Args:
        wiki_dir: Path to the wiki directory.
        entities: List of merged entities to write.
        doc_name: Source document name for backlinks.

    Returns:
        List of entity slugs that were written.
    """
    entities_dir = wiki_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)

    written_slugs: list[str] = []

    for entity in entities:
        slug = _sanitize_entity_slug(entity.canonical_name)
        path = entities_dir / f"{slug}.md"

        existing_text = ""
        if path.exists():
            existing_text = path.read_text(encoding="utf-8")

        page_text = _build_entity_page(entity, existing_text)
        path.write_text(page_text, encoding="utf-8")
        written_slugs.append(slug)

    return written_slugs


# ---------------------------------------------------------------------------
# Entity index
# ---------------------------------------------------------------------------

def update_entity_index(wiki_dir: Path, all_entities: list[MergedEntity]) -> None:
    """Maintain wiki/entities/index.md grouped by entity type.

    Reads existing index, merges new entities, and rewrites.
    """
    entities_dir = wiki_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    index_path = entities_dir / "index.md"

    # Parse existing index to preserve entries
    existing_entries: dict[str, list[str]] = {}  # type -> [entry lines]
    if index_path.exists():
        current_type = ""
        for line in index_path.read_text(encoding="utf-8").split("\n"):
            if line.startswith("## "):
                current_type = line[3:].strip()
                existing_entries[current_type] = []
            elif line.startswith("- [[entities/") and current_type:
                existing_entries[current_type].append(line)

    # Add new entities
    for entity in all_entities:
        slug = _sanitize_entity_slug(entity.canonical_name)
        entry = f"- [[entities/{slug}]]"
        if entity.description:
            entry += f" — {entity.description}"

        etype = entity.entity_type
        if etype not in existing_entries:
            existing_entries[etype] = []

        # Check for duplicate entry
        already_exists = any(
            f"[[entities/{slug}]]" in line
            for line in existing_entries[etype]
        )
        if not already_exists:
            existing_entries[etype].append(entry)

    # Write index grouped by type (only types that have entries)
    lines = ["# Entity Index\n"]
    for etype in ENTITY_TYPES:
        entries = existing_entries.get(etype, [])
        if entries:
            lines.append(f"\n## {etype}")
            lines.extend(sorted(entries))

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Backlinks
# ---------------------------------------------------------------------------


def add_entity_backlinks(
    wiki_dir: Path,
    doc_name: str,
    entity_slugs: list[str],
) -> None:
    """Add [[entities/X]] backlinks to the summary page.

    Creates or updates a '## Related Entities' section in the summary.
    """
    if not entity_slugs:
        return

    summary_path = wiki_dir / "summaries" / f"{doc_name}.md"
    if not summary_path.exists():
        return

    text = summary_path.read_text(encoding="utf-8")
    missing = [s for s in entity_slugs if f"[[entities/{s}]]" not in text]
    if not missing:
        return

    lines = text.split("\n")
    ensure_h2_section(lines, "## Related Entities")
    for slug in reversed(missing):
        insert_section_entry(lines, "## Related Entities", f"- [[entities/{slug}]]")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def add_concept_backlinks(
    wiki_dir: Path,
    doc_name: str,
    concept_slugs: list[str],
) -> None:
    """Add [[concepts/X]] backlinks to the summary page.

    Creates or updates a '## Related Concepts' section in the summary.
    """
    if not concept_slugs:
        return

    summary_path = wiki_dir / "summaries" / f"{doc_name}.md"
    if not summary_path.exists():
        return

    text = summary_path.read_text(encoding="utf-8")
    missing = [s for s in concept_slugs if f"[[concepts/{s}]]" not in text]
    if not missing:
        return

    lines = text.split("\n")
    ensure_h2_section(lines, "## Related Concepts")
    for slug in reversed(missing):
        insert_section_entry(lines, "## Related Concepts", f"- [[concepts/{slug}]]")
    summary_path.write_text("\n".join(lines), encoding="utf-8")




# ---------------------------------------------------------------------------
# Concept page writer
# ---------------------------------------------------------------------------

def _build_concept_page(
    concept: MergedEntity,
    doc_name: str,
    existing_text: str = "",
) -> str:
    """Build or update a concept wiki page from a MergedEntity.

    Concept pages live in wiki/concepts/ and link to related entities
    and summaries.
    """
    source_link = f"summaries/{doc_name}"

    if existing_text:
        existing_meta, existing_body = _read_entity_frontmatter(existing_text)
        # Merge sources
        old_sources = set(existing_meta.get("sources", []))
        new_sources = {f"summaries/{s}" for s in concept.sources}
        all_sources = sorted(old_sources | new_sources)

        description = concept.description or existing_meta.get("brief", "")

        # Rebuild frontmatter
        fm_lines = ["type: CONCEPT"]
        if all_sources:
            fm_lines.append(f"sources: [{', '.join(all_sources)}]")
        if description:
            fm_lines.append(f"brief: {description}")
        frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n"

        # Add mention if missing
        body = existing_body
        if source_link and f"[[{source_link}]]" not in body:
            if "## Mentions" in body:
                body = body.rstrip() + f"\n- [[{source_link}]]"

        # Add LLM-inferred related entities
        if concept.related_entities:
            lines = body.split("\n")
            ensure_h2_section(lines, "## Related Entities")
            for entity_name in concept.related_entities:
                entity_slug = _sanitize_entity_slug(entity_name)
                entry = f"- [[entities/{entity_slug}]]"
                if entry not in body:
                    insert_section_entry(lines, "## Related Entities", entry)
            body = "\n".join(lines)

        return frontmatter + "\n" + body

    # New concept page
    fm_lines = ["type: CONCEPT"]
    if concept.sources:
        source_links = [f"summaries/{s}" for s in concept.sources]
        fm_lines.append(f"sources: [{', '.join(source_links)}]")
    if concept.description:
        fm_lines.append(f"brief: {concept.description}")
    frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n"

    heading = concept.canonical_name
    body = f"\n# {heading}\n"
    if concept.description:
        body += f"\n{concept.description}\n"

    body += "\n## Mentions\n"
    body += f"- [[{source_link}]]\n"

    # LLM-inferred related entities
    body += "\n## Related Entities\n"
    for entity_name in concept.related_entities:
        entity_slug = _sanitize_entity_slug(entity_name)
        body += f"- [[entities/{entity_slug}]]\n"

    body += "\n## Related Concepts\n"

    return frontmatter + body


def write_concept_pages(
    wiki_dir: Path,
    concepts: list[MergedEntity],
    doc_name: str,
) -> list[str]:
    """Create or update concept pages in wiki/concepts/.

    Args:
        wiki_dir: Path to the wiki directory.
        concepts: List of merged concepts (category == "concept") to write.
        doc_name: Source document name for backlinks.

    Returns:
        List of concept slugs that were written.
    """
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    written_slugs: list[str] = []

    for concept in concepts:
        slug = _sanitize_concept_slug(concept.canonical_name)
        path = concepts_dir / f"{slug}.md"

        existing_text = ""
        if path.exists():
            existing_text = path.read_text(encoding="utf-8")

        page_text = _build_concept_page(concept, doc_name, existing_text)
        path.write_text(page_text, encoding="utf-8")
        written_slugs.append(slug)

    return written_slugs


# ---------------------------------------------------------------------------
# Temporal metadata embedding
# ---------------------------------------------------------------------------

def embed_temporal_metadata(
    wiki_dir: Path,
    doc_name: str,
    temporal: list[MergedEntity],
) -> None:
    """Embed DATE/TIME entities as date_mentioned frontmatter in the summary page.

    Does not create wiki pages for temporal entities — just adds metadata.
    """
    if not temporal:
        return

    summary_path = wiki_dir / "summaries" / f"{doc_name}.md"
    if not summary_path.exists():
        return

    text = summary_path.read_text(encoding="utf-8")
    dates = sorted(set(t.canonical_name for t in temporal))

    if text.startswith("---"):
        fm_end = text.find("---", 3)
        if fm_end != -1:
            fm_block = text[:fm_end]
            body = text[fm_end:]
            if "date_mentioned:" in fm_block:
                # Already has date_mentioned — merge
                import re
                existing = re.search(r'date_mentioned:\s*\[([^\]]*)\]', fm_block)
                if existing:
                    old_dates = [d.strip().strip('"') for d in existing.group(1).split(",") if d.strip()]
                    all_dates = sorted(set(old_dates + dates))
                    new_line = f'date_mentioned: [{", ".join(all_dates)}]'
                    fm_block = fm_block[:existing.start()] + new_line + fm_block[existing.end():]
            else:
                fm_block = fm_block.rstrip() + f"\ndate_mentioned: [{', '.join(dates)}]"
            text = fm_block + body

    summary_path.write_text(text, encoding="utf-8")


def add_entity_links_to_concept_pages(
    wiki_dir: Path,
    concepts: list[MergedEntity],
) -> None:
    """Add [[entities/X]] links to existing concept pages based on LLM-inferred relationships.

    For each concept's related_entities, ensure the concept page has a
    ## Related Entities section with the entity links.
    """
    concepts_dir = wiki_dir / "concepts"
    if not concepts_dir.exists():
        return

    for concept in concepts:
        if not concept.related_entities:
            continue
        slug = _sanitize_concept_slug(concept.canonical_name)
        path = concepts_dir / f"{slug}.md"
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        missing = []
        for entity_name in concept.related_entities:
            entity_slug = _sanitize_entity_slug(entity_name)
            if f"[[entities/{entity_slug}]]" not in text:
                missing.append(entity_slug)
        if not missing:
            continue

        lines = text.split("\n")
        ensure_h2_section(lines, "## Related Entities")
        for entity_slug in reversed(missing):
            insert_section_entry(lines, "## Related Entities", f"- [[entities/{entity_slug}]]")
        path.write_text("\n".join(lines), encoding="utf-8")

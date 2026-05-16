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
    body += "\n## Related Concepts\n"

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


def add_entity_links_to_concepts(
    wiki_dir: Path,
    entity_slugs: list[str],
) -> None:
    """Ensure entity pages link back to related concepts (if concepts exist).

    This is a lightweight pass — it checks if any concept pages mention
    the entity name and adds a Related Concepts link on the entity page.
    """
    if not entity_slugs:
        return

    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    if not entities_dir.exists() or not concepts_dir.exists():
        return

    # Build a simple name → concept slug index
    concept_files = list(concepts_dir.glob("*.md"))
    concept_names: dict[str, str] = {}  # normalized_name → slug
    for cf in concept_files:
        concept_names[_normalize_name(cf.stem)] = cf.stem

    for slug in entity_slugs:
        entity_path = entities_dir / f"{slug}.md"
        if not entity_path.exists():
            continue

        text = entity_path.read_text(encoding="utf-8")
        meta, body = _read_entity_frontmatter(text)

        # Check if any concept page name matches the entity name or aliases
        entity_norm = _normalize_name(slug)
        related_concepts: list[str] = []
        for concept_norm, concept_slug in concept_names.items():
            if concept_norm in entity_norm or entity_norm in concept_norm:
                if f"[[concepts/{concept_slug}]]" not in body:
                    related_concepts.append(concept_slug)

        if related_concepts:
            lines = body.split("\n")
            ensure_h2_section(lines, "## Related Concepts")
            for cs in related_concepts:
                entry = f"- [[concepts/{cs}]]"
                if entry not in body:
                    insert_section_entry(lines, "## Related Concepts", entry)

            # Rebuild full page
            fm_lines = [f"type: {meta.get('type', 'CONCEPT')}"]
            if meta.get("aliases"):
                fm_lines.append(f"aliases: [{', '.join(meta['aliases'])}]")
            if meta.get("sources"):
                fm_lines.append(f"sources: [{', '.join(meta['sources'])}]")
            if meta.get("brief"):
                fm_lines.append(f"brief: {meta['brief']}")
            frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n"
            entity_path.write_text(frontmatter + "\n".join(lines), encoding="utf-8")

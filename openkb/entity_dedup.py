"""LLM-powered entity deduplication for OpenKB.

Uses an LLM to find semantically duplicate entities across the wiki,
then merges them by updating the canonical page and re-linking all
wikilinks across the wiki.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import litellm

logger = logging.getLogger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def collect_entity_graph(wiki_dir: Path) -> dict:
    """Build a graph of all entities with their links.

    Returns a dict: {slug: {name, type, aliases, linked_to: [slugs], linked_from: [slugs]}}
    """
    entities_dir = wiki_dir / "entities"
    if not entities_dir.is_dir():
        return {}

    slug_to_name: dict[str, str] = {}

    # First pass: read all entity pages
    nodes: dict[str, dict] = {}
    for path in sorted(entities_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        slug = path.stem
        text = path.read_text(encoding="utf-8")

        name = slug.replace("-", " ")
        etype = "UNKNOWN"
        aliases: list[str] = []
        linked_to: list[str] = []
        for line in text.split("\n"):
            if line.startswith("# ") and not line.startswith("## "):
                name = line[2:].strip()
            if line.startswith("type:"):
                etype = line[5:].strip()
            if line.startswith("aliases:"):
                raw = line[8:].strip().strip("[]")
                if raw:
                    aliases = [a.strip() for a in raw.split(",")]
            for match in _WIKILINK_RE.finditer(line):
                target = match.group(1).split("|")[0].strip()
                if target.startswith("entities/"):
                    target_slug = target[9:]
                    if target_slug != slug and target_slug:
                        linked_to.append(target_slug)

        slug_to_name[slug] = name
        nodes[slug] = {
            "name": name,
            "type": etype,
            "aliases": aliases,
            "linked_to": linked_to,
            "linked_from": [],  # filled in second pass
        }

    # Second pass: build incoming links
    for slug, node in nodes.items():
        for target_slug in node["linked_to"]:
            if target_slug in nodes:
                nodes[target_slug].setdefault("linked_from", []).append(slug)

    return nodes


class MergeGroup:
    """A proposed merge of multiple entity slugs into one canonical entity."""
    def __init__(
        self,
        canonical_name: str,
        canonical_slug: str,
        duplicate_slugs: list[str],
        reason: str,
        aliases_to_add: list[str],
    ):
        self.canonical_name = canonical_name
        self.canonical_slug = canonical_slug
        self.duplicate_slugs = duplicate_slugs
        self.reason = reason
        self.aliases_to_add = aliases_to_add


def _slug_from_name(name: str) -> str:
    """Convert entity name to slug (same logic as entity_extractor._sanitize_entity_slug)."""
    import unicodedata
    s = unicodedata.normalize("NFKC", name)
    s = s.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s or "unnamed-entity"


class EntityDedupAgent:
    """LLM-powered entity deduplication."""

    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = base_url

    def find_merges(self, entity_graph: dict) -> list[MergeGroup]:
        """Ask LLM to find duplicate entities in the graph.

        Returns a list of MergeGroup proposals.
        """
        if not entity_graph:
            return []

        entity_list = []
        for slug, node in entity_graph.items():
            # Build neighbourhood description for context
            neighbours = list(set(node["linked_to"]) | set(node["linked_from"]))
            neighbour_names = [
                entity_graph.get(n, {}).get("name", n)
                for n in neighbours
                if n in entity_graph
            ]
            entity_list.append({
                "slug": slug,
                "name": node["name"],
                "type": node["type"],
                "aliases": node.get("aliases", []),
                "linked_to_names": neighbour_names[:10],  # cap for prompt size
            })

        prompt = """\
You are an entity deduplication agent. Find entities in this knowledge base \
that refer to the same real-world thing. Be conservative: only merge when you are confident.

Entities (JSON):
{entities_json}

Return a JSON array of merge groups. Each group must have:
- canonical_name: the best canonical name for the merged entity
- canonical_slug: the wiki slug for the canonical entity (e.g. "google" from "Google")
- slugs: all wiki slugs to merge (including canonical_slug)
- reason: brief explanation of why these are the same entity
- aliases_to_add: additional aliases discovered for the canonical entity

Rules:
- "Google" and "Google LLC" and "Google Inc." → merge (corporate suffix variation)
- "U.S." and "USA" and "United States" → merge (abbreviation expansion)
- "Tim Cook" and "Timothy Cook" and "T. Cook" → merge (name variation)
- "Claude 3.5 Sonnet" and "Claude" in AI context → merge if linked to same neighbours
- "Apple" (company) and "Apple Inc." → merge
- "OpenAI" and "Open AI" → merge
- "GPT-4" and "GPT4" and "Generative Pre-trained Transformer 4" → merge
- "BERT" and "Bidirectional Encoder Representations from Transformers" → merge
- "Python" (language) and "Python" (snake) → DO NOT merge unless context differs
- "Google" and "DeepMind" → DO NOT merge unless they have many shared neighbours
- Two entities that both link to many of the same other entities → likely same

Return ONLY valid JSON (a JSON array), no fences, no explanation.
""".format(entities_json=json.dumps(entity_list, indent=2))

        try:
            kwargs: dict = {"max_tokens": 4096}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("LLM entity dedup failed: %s", exc)
            return []

        # Strip fences
        if raw.startswith("```"):
            first_nl = raw.find("\n")
            raw = raw[first_nl + 1:] if first_nl != -1 else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            from json_repair import repair_json
            parsed = json.loads(repair_json(raw.strip()))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse LLM dedup output: %s", exc)
            return []

        if not isinstance(parsed, list):
            logger.warning("LLM dedup returned non-array: %s", type(parsed).__name__)
            return []

        groups: list[MergeGroup] = []
        used_slugs: set[str] = set()

        for item in parsed:
            if not isinstance(item, dict):
                continue
            canonical_slug = item.get("canonical_slug", "").strip()
            duplicate_slugs = item.get("slugs", [])
            if not canonical_slug or not duplicate_slugs:
                continue
            # Skip if any slug was already assigned to a previous canonical
            slugs_in_group = set(duplicate_slugs)
            if slugs_in_group & used_slugs:
                continue
            used_slugs |= slugs_in_group
            groups.append(MergeGroup(
                canonical_name=item.get("canonical_name", canonical_slug),
                canonical_slug=canonical_slug,
                duplicate_slugs=[s for s in duplicate_slugs if s != canonical_slug],
                reason=item.get("reason", ""),
                aliases_to_add=item.get("aliases_to_add", []) or [],
            ))

        logger.info("LLM dedup: proposed %d merge groups", len(groups))
        return groups


def merge_entities_wiki(
    wiki_dir: Path,
    groups: list[MergeGroup],
) -> dict:
    """Apply merge groups: update canonical pages, re-link all wikilinks, rebuild index.

    Returns {"merged": N, "links_updated": M, "descriptions": [...]}.
    """
    if not groups:
        return {"merged": 0, "links_updated": 0, "descriptions": []}

    entities_dir = wiki_dir / "entities"
    slug_map: dict[str, str] = {}  # duplicate_slug -> canonical_slug
    for g in groups:
        for dup in g.duplicate_slugs:
            slug_map[dup] = g.canonical_slug

    links_updated = 0

    # Step 1: Update all wikilinks across wiki (before deleting duplicates)
    for md_path in wiki_dir.rglob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        original = text
        for dup_slug, canon_slug in slug_map.items():
            # Replace [[entities/dup-slug]] with [[entities/canonical-slug]]
            # Handle both [[entities/foo]] and [[entities/foo|display]]
            text = re.sub(
                rf"\[\[entities/{re.escape(dup_slug)}(\|[^\]]+)?\]\]",
                rf"[[entities/{canon_slug}\1]]",
                text,
            )
        if text != original:
            md_path.write_text(text, encoding="utf-8")
            links_updated += 1

    # Step 2: Update canonical pages with merged aliases, delete duplicates
    from openkb.entity_writer import _read_entity_frontmatter

    descriptions: list[str] = []
    for group in groups:
        if not group.duplicate_slugs:
            continue

        canon_path = entities_dir / f"{group.canonical_slug}.md"
        if not canon_path.exists():
            continue

        # Read canonical + all duplicates
        all_aliases: set[str] = {group.canonical_name}
        all_sources: set[str] = set()
        existing_brief = ""

        for slug in [group.canonical_slug] + group.duplicate_slugs:
            path = entities_dir / f"{slug}.md"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            meta, body = _read_entity_frontmatter(text)
            all_aliases.update(meta.get("aliases", []))
            all_sources.update(meta.get("sources", []))
            if not existing_brief:
                existing_brief = meta.get("brief", "")
            if slug != group.canonical_slug:
                # Merge related entities/concepts from duplicate into canonical body
                _merge_related_into_canonical(canon_path, text, group.canonical_slug)

        all_aliases.discard(group.canonical_name)
        all_aliases.update(group.aliases_to_add)
        all_aliases.discard("")

        # Update canonical frontmatter
        _update_canonical_frontmatter(
            canon_path, group, sorted(all_aliases), sorted(all_sources), existing_brief
        )

        # Delete duplicates
        for slug in group.duplicate_slugs:
            dup_path = entities_dir / f"{slug}.md"
            if dup_path.exists():
                dup_path.unlink()

        descriptions.append(
            f"{', '.join(group.duplicate_slugs)} → {group.canonical_slug}"
        )

    # Step 3: Rebuild entity index
    _rebuild_entity_index(wiki_dir)

    logger.info("Entity dedup: %d merged, %d links updated", len(groups), links_updated)
    return {
        "merged": len(groups),
        "links_updated": links_updated,
        "descriptions": descriptions,
    }


def _merge_related_into_canonical(canon_path: Path, dup_text: str, canon_slug: str) -> None:
    """Merge ## Related Entities and ## Related Concepts from duplicate into canonical."""
    if not canon_path.exists():
        return
    from openkb.wiki_utils import ensure_h2_section, insert_section_entry, section_contains_entry

    canon_text = canon_path.read_text(encoding="utf-8")
    lines = canon_text.split("\n")

    # Parse related entities/concepts from duplicate
    in_section = None
    entries_to_add: list[tuple[str, str]] = []  # (section, entry)
    for line in dup_text.split("\n"):
        stripped = line.strip()
        if stripped == "## Related Entities":
            in_section = "## Related Entities"
            continue
        if stripped == "## Related Concepts":
            in_section = "## Related Concepts"
            continue
        if stripped.startswith("## "):
            in_section = None
            continue
        if in_section and stripped.startswith("- [[entities/") or stripped.startswith("- [[concepts/"):
            entries_to_add.append((in_section, line.strip()))

    for section, entry in entries_to_add:
        if not section_contains_entry(lines, section, entry):
            insert_section_entry(lines, section, entry)

    canon_path.write_text("\n".join(lines), encoding="utf-8")


def _update_canonical_frontmatter(
    path: Path,
    group: MergeGroup,
    aliases: list[str],
    sources: list[str],
    brief: str,
) -> None:
    """Update canonical entity page frontmatter with merged data."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return
    end = text.find("---", 3)
    if end == -1:
        return
    fm_block = text[:end]
    body = text[end + 3:].lstrip("\n")

    # Get existing type
    etype = "UNKNOWN"
    for line in fm_block.split("\n"):
        if line.startswith("type:"):
            etype = line[5:].strip()
            break

    fm_lines = [f"type: {etype}"]
    if aliases:
        fm_lines.append(f"aliases: [{', '.join(aliases)}]")
    if sources:
        fm_lines.append(f"sources: [{', '.join(sources)}]")
    if brief:
        fm_lines.append(f"brief: {brief}")
    new_fm = "---\n" + "\n".join(fm_lines) + "\n---"
    new_text = new_fm + "\n" + body
    path.write_text(new_text, encoding="utf-8")


def _rebuild_entity_index(wiki_dir: Path) -> None:
    """Rebuild wiki/entities/index.md from entity pages on disk."""
    from openkb.entity_extractor import ENTITY_TYPES

    entities_dir = wiki_dir / "entities"
    if not entities_dir.is_dir():
        return

    entries: dict[str, list[str]] = {}
    for path in sorted(entities_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text(encoding="utf-8")
        meta, _ = _read_entity_frontmatter(text)
        etype = meta.get("type", "UNKNOWN")
        desc = meta.get("brief", "")
        entry = f"- [[entities/{path.stem}]]"
        if desc:
            entry += f" — {desc}"
        entries.setdefault(etype, []).append(entry)

    lines = ["# Entity Index\n"]
    for etype in ENTITY_TYPES:
        type_entries = entries.get(etype, [])
        if type_entries:
            lines.append(f"\n## {etype}")
            lines.extend(sorted(type_entries))

    (entities_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Import for _read_entity_frontmatter
from openkb.entity_writer import _read_entity_frontmatter as _read_entity_frontmatter
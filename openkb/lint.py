"""Structural lint checks for the OpenKB wiki.

Checks for:
- Broken [[wikilinks]] — link targets that don't exist
- Orphaned pages — pages with no incoming or outgoing links
- Missing wiki entries — raw files without corresponding sources/summaries
- Index sync — index.md links vs actual files on disk
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# Matches [[wikilink]] or [[subdir/link]]
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Files to exclude from lint scanning (schema, logs, etc.)
_EXCLUDED_FILES = {"AGENTS.md", "SCHEMA.md", "log.md"}


def _normalize_target(target: str) -> str:
    """Normalize a wikilink target for fuzzy comparison.

    Applies, in order:
    - NFKC unicode normalization (e.g. full-width '）' → ASCII ')')
    - Lowercase
    - Underscore → hyphen
    - Collapse repeated hyphens
    - Strip leading/trailing hyphens (per segment when path-like)

    Path separators are preserved so ``concepts/Gist_Memory`` normalizes to
    ``concepts/gist-memory``.
    """
    s = unicodedata.normalize("NFKC", target)
    s = s.lower().replace("_", "-")
    # Normalize each path segment independently to avoid collapsing the '/'
    parts = [re.sub(r"-+", "-", p).strip("-") for p in s.split("/")]
    return "/".join(parts)


def build_norm_index(known_targets: set[str]) -> dict[str, str]:
    """Build the normalized-form → canonical-target index used by
    :func:`strip_ghost_wikilinks`.

    Useful when calling ``strip_ghost_wikilinks`` repeatedly with the same
    ``known_targets`` (e.g. ``fix_broken_links`` scanning N wiki files, or
    ``_save_transcript`` stripping N assistant turns) — build the index
    once and pass it via the ``norm_index`` parameter to avoid O(N·M)
    redundant rebuilds.
    """
    return {_normalize_target(t): t for t in known_targets}


def strip_ghost_wikilinks(
    content: str,
    known_targets: set[str],
    *,
    norm_index: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Strip [[wikilinks]] whose targets do not exist in ``known_targets``.

    For each ``[[X]]`` or ``[[X|alias]]`` in ``content``:

    - If ``X`` is in ``known_targets`` exactly, the link is kept as-is.
    - Otherwise, ``X`` is normalized (see :func:`_normalize_target`) and
      matched against the normalized form of each known target. On a hit,
      the link is rewritten to the canonical target form.
    - Otherwise, the brackets are removed and the link becomes plain text
      (the alias if provided, otherwise the slug rendered as words).

    Args:
        content: Markdown text containing zero or more ``[[wikilinks]]``.
        known_targets: Valid link targets, e.g.
            ``{"concepts/attention", "summaries/paper"}``.
        norm_index: Optional pre-built normalized index from
            :func:`build_norm_index`. Pass this when calling in a loop
            with the same ``known_targets`` to skip redundant rebuilds.

    Returns:
        Tuple of ``(rewritten_content, ghost_targets)`` where
        ``ghost_targets`` is the list of unresolved targets that were
        stripped (one entry per occurrence, in document order).
    """
    if norm_index is None:
        norm_index = build_norm_index(known_targets)

    ghosts: list[str] = []

    def _repl(m: re.Match) -> str:
        raw = m.group(1)
        if "|" in raw:
            target, alias = raw.split("|", 1)
            target = target.strip()
            alias = alias.strip()
        else:
            target = raw.strip()
            alias = None

        # Direct hit
        if target in known_targets:
            return m.group(0)

        # Fuzzy normalized hit → rewrite to canonical
        canonical = norm_index.get(_normalize_target(target))
        if canonical is not None:
            if alias:
                return f"[[{canonical}|{alias}]]"
            return f"[[{canonical}]]"

        # Ghost — strip brackets, leave readable display
        ghosts.append(target)
        if alias:
            return alias
        stem = target.rsplit("/", 1)[-1]
        return stem.replace("-", " ").replace("_", " ")

    cleaned = _WIKILINK_RE.sub(_repl, content)
    return cleaned, ghosts


def _read_md(path: Path) -> str:
    """Read a Markdown file safely, returning empty string on error."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _all_wiki_pages(wiki: Path) -> dict[str, Path]:
    """Return a mapping of stem/relative-path → absolute Path for all .md files.

    Keys are normalized: 'concepts/attention', 'summaries/paper', 'index', etc.
    """
    pages: dict[str, Path] = {}
    for md in wiki.rglob("*.md"):
        rel = md.relative_to(wiki)
        # Store both the full relative path without extension and the stem
        key = str(rel.with_suffix("")).replace("\\", "/")
        pages[key] = md
        # Also index by stem alone for convenience
        pages[md.stem] = md
    return pages


def _extract_wikilinks(text: str) -> list[str]:
    """Return all wikilink targets found in *text*.

    Handles ``[[target|display text]]`` alias syntax — only the target is returned.
    """
    raw = _WIKILINK_RE.findall(text)
    return [link.split("|")[0].strip() for link in raw]


def list_existing_wiki_targets(wiki_dir: Path) -> set[str]:
    """Return the set of currently-existing wikilink targets on disk.

    Includes every ``concepts/{stem}``, ``summaries/{stem}``, and
    ``entities/{stem}`` for .md files actually present in the wiki,
    plus ``index`` when ``index.md`` exists. Used to seed the whitelist
    passed to :func:`strip_ghost_wikilinks` from both the compile pipeline
    and any other code path that writes LLM-generated content to the wiki
    (e.g. ``openkb query --save``).
    """
    targets: set[str] = set()
    concepts_dir = wiki_dir / "concepts"
    summaries_dir = wiki_dir / "summaries"
    entities_dir = wiki_dir / "entities"
    if concepts_dir.is_dir():
        targets.update(f"concepts/{p.stem}" for p in concepts_dir.glob("*.md"))
    if summaries_dir.is_dir():
        targets.update(f"summaries/{p.stem}" for p in summaries_dir.glob("*.md"))
    if entities_dir.is_dir():
        targets.update(f"entities/{p.stem}" for p in entities_dir.glob("*.md"))
    if (wiki_dir / "index.md").exists():
        targets.add("index")
    return targets


def fix_broken_links(wiki: Path) -> tuple[int, int]:
    """Rewrite or strip broken [[wikilinks]] across the wiki in place.

    For each Markdown page under ``wiki`` (excluding ``reports/`` and
    ``sources/`` and excluded files), runs :func:`strip_ghost_wikilinks`
    against the set of valid targets currently on disk. Targets that match
    fuzzily (case, ``_`` vs ``-``, NFKC) are rewritten to canonical form;
    targets that have no match are demoted to plain text.

    Args:
        wiki: Path to the wiki root directory.

    Returns:
        Tuple of ``(files_changed, ghosts_stripped)``.
    """
    pages = _all_wiki_pages(wiki)
    # The same fuzzy normalization _all_wiki_pages stores both the full
    # relative path (e.g. ``concepts/attention``) and the bare stem
    # (``attention``). Use the full-path keys so that links like
    # ``[[concepts/foo]]`` resolve against ``concepts/`` files only.
    known_targets: set[str] = {
        key for key in pages if "/" in key or key == "index"
    }
    # Build the normalized index once and reuse across every file —
    # otherwise strip_ghost_wikilinks would rebuild it per file (O(F·M)).
    norm_index = build_norm_index(known_targets)

    files_changed = 0
    ghosts_stripped = 0
    for md in wiki.rglob("*.md"):
        if md.name in _EXCLUDED_FILES:
            continue
        rel_parts = md.relative_to(wiki).parts
        if rel_parts and rel_parts[0] in ("reports", "sources"):
            continue
        text = _read_md(md)
        cleaned, ghosts = strip_ghost_wikilinks(
            text, known_targets, norm_index=norm_index,
        )
        if cleaned != text:
            md.write_text(cleaned, encoding="utf-8")
            files_changed += 1
            ghosts_stripped += len(ghosts)
    return files_changed, ghosts_stripped


def find_broken_links(wiki: Path) -> list[str]:
    """Scan all wiki pages for [[wikilinks]] pointing to non-existent targets.

    Args:
        wiki: Path to the wiki root directory.

    Returns:
        List of error strings describing each broken link.
    """
    pages = _all_wiki_pages(wiki)
    errors: list[str] = []

    for md in wiki.rglob("*.md"):
        if md.name in _EXCLUDED_FILES:
            continue
        # Skip reports/ and sources/ — auto-generated, not wiki content
        rel_parts = md.relative_to(wiki).parts
        if rel_parts and rel_parts[0] in ("reports", "sources"):
            continue
        text = _read_md(md)
        for target in _extract_wikilinks(text):
            # Normalise target: strip leading/trailing whitespace and slashes
            target_norm = target.strip().strip("/")
            # Check if target resolves as a key in our page map
            if target_norm not in pages:
                rel = md.relative_to(wiki)
                errors.append(f"Broken link [[{target}]] in {rel}")

    return sorted(errors)


def find_orphans(wiki: Path) -> list[str]:
    """Find pages that have no links to or from other pages.

    A page is orphaned if:
    - No other page links to it (no incoming links), AND
    - It has no outgoing wikilinks itself.

    index.md is excluded from orphan detection.

    Args:
        wiki: Path to the wiki root directory.

    Returns:
        List of relative page paths that are orphaned.
    """
    # Exclude index, schema, log, and sources/ (sources are auto-generated, not expected to be linked)
    all_mds = [
        p for p in wiki.rglob("*.md")
        if p.name not in {"index.md", *_EXCLUDED_FILES}
        and "sources" not in p.relative_to(wiki).parts
    ]
    if not all_mds:
        return []

    # Build outgoing links per page
    outgoing: dict[str, set[str]] = {}
    for md in all_mds:
        rel = str(md.relative_to(wiki).with_suffix("")).replace("\\", "/")
        text = _read_md(md)
        outgoing[rel] = set(_extract_wikilinks(text))

    # Build incoming link set (which pages are linked to)
    incoming: set[str] = set()
    for links in outgoing.values():
        for lnk in links:
            incoming.add(lnk.strip().strip("/"))
        # Also add stems
        for lnk in links:
            incoming.add(Path(lnk.strip()).stem)

    orphans: list[str] = []
    for rel, links in outgoing.items():
        stem = Path(rel).stem
        has_incoming = rel in incoming or stem in incoming
        has_outgoing = bool(links)
        if not has_incoming and not has_outgoing:
            orphans.append(rel)

    return sorted(orphans)


def find_missing_entries(raw: Path, wiki: Path) -> list[str]:
    """Find files in raw/ that have no corresponding wiki entries.

    A file is considered "present" if it has either a sources/ or summaries/
    page with the same stem.

    Args:
        raw: Path to the raw documents directory.
        wiki: Path to the wiki root directory.

    Returns:
        List of filenames in raw/ with no wiki entry.
    """
    sources_dir = wiki / "sources"
    summaries_dir = wiki / "summaries"

    sources_stems = {p.stem for p in sources_dir.glob("*.md")} if sources_dir.exists() else set()
    summary_stems = {p.stem for p in summaries_dir.glob("*.md")} if summaries_dir.exists() else set()
    known_stems = sources_stems | summary_stems

    missing: list[str] = []
    if raw.exists():
        for f in raw.iterdir():
            if f.is_file() and f.stem not in known_stems:
                missing.append(f.name)

    return sorted(missing)


def check_index_sync(wiki: Path) -> list[str]:
    """Compare index.md wikilinks against actual files on disk.

    Returns issues for:
    - Links in index.md pointing to non-existent pages
    - Pages in summaries/ or concepts/ not mentioned in index.md

    Args:
        wiki: Path to the wiki root directory.

    Returns:
        List of sync issue strings.
    """
    index_path = wiki / "index.md"
    issues: list[str] = []

    if not index_path.exists():
        return ["index.md does not exist"]

    index_text = _read_md(index_path)
    index_links = set(_extract_wikilinks(index_text))
    pages = _all_wiki_pages(wiki)

    # Check that all index links resolve
    for lnk in index_links:
        lnk_norm = lnk.strip().strip("/")
        if lnk_norm not in pages:
            issues.append(f"index.md links to missing page: [[{lnk}]]")

    # Check that summaries and concepts pages are mentioned in index
    index_stems = {Path(lnk.strip()).stem for lnk in index_links}
    index_text_lower = index_text.lower()

    for subdir in ("summaries", "concepts"):
        subdir_path = wiki / subdir
        if not subdir_path.exists():
            continue
        for md in sorted(subdir_path.glob("*.md")):
            stem = md.stem
            if stem not in index_stems and stem.lower() not in index_text_lower:
                issues.append(f"{subdir}/{stem}.md not mentioned in index.md")

    return sorted(issues)


# ---------------------------------------------------------------------------
# Entity lint: dedup and source validation
# ---------------------------------------------------------------------------

def _read_entity_meta(text: str) -> dict:
    """Parse YAML-like frontmatter from an entity page.

    Returns a dict with keys: type, aliases (list), sources (list), brief.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm_block = text[3:end].strip()
    meta: dict = {}
    for line in fm_block.split("\n"):
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key in ("aliases", "sources"):
            if val.startswith("[") and val.endswith("]"):
                meta[key] = [s.strip() for s in val[1:-1].split(",") if s.strip()]
            else:
                meta[key] = [val] if val else []
        else:
            meta[key] = val
    return meta


def _build_entity_index(entities_dir: Path) -> dict[str, dict]:
    """Build an index of all entity pages on disk.

    Returns a dict mapping slug → {"path": Path, "meta": dict, "text": str}.
    """
    index: dict[str, dict] = {}
    if not entities_dir.is_dir():
        return index
    for path in sorted(entities_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        text = _read_md(path)
        meta = _read_entity_meta(text)
        index[path.stem] = {"path": path, "meta": meta, "text": text}
    return index


def _entity_name_tokens(name: str) -> set[str]:
    """Normalize an entity name into a set of word tokens for fuzzy comparison.

    Splits camelCase/PascalCase, strips common corporate suffixes and
    punctuation, then splits on whitespace.
    ``"OpenAI Inc."`` → ``{"open", "ai"}``,  ``"Tim Cook"`` →
    ``{"tim", "cook"}``.
    """
    s = unicodedata.normalize("NFKC", name)
    # Split camelCase/PascalCase BEFORE lowercasing: "OpenAI" → "Open AI"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    _STOP = {"inc", "corp", "ltd", "llc", "llp", "co",
             "the", "a", "an", "and", "or", "of", "for"}
    tokens = {t for t in s.split() if len(t) > 1 and t not in _STOP}
    return tokens


def _entity_names_similar(name_a: str, name_b: str) -> bool:
    """Check if two entity names likely refer to the same thing.

    Uses token-based Jaccard similarity with prefix matching to handle:
    - ``"OpenAI Inc"`` vs ``"Open AI"`` (compound word splitting)
    - ``"Tim Cook"`` vs ``"Timothy Cook"`` (prefix matching)
    - ``"Apple"`` vs ``"Apple Inc"`` (subset matching)
    """
    tok_a = _entity_name_tokens(name_a)
    tok_b = _entity_name_tokens(name_b)
    if not tok_a or not tok_b:
        return False

    overlap = len(tok_a & tok_b)
    union = len(tok_a | tok_b)

    # Direct token overlap
    if union > 0 and overlap / union >= 0.5:
        return True

    # Subset: one name's tokens are entirely contained in the other
    # Catches "Apple" vs "Apple Inc", "Tim Cook" vs "Tim Cook Jr"
    if tok_a <= tok_b or tok_b <= tok_a:
        return True

    # Prefix matching: "tim" matches "timothy", "ai" matches "artificial"
    prefix_matches = 0
    for ta in tok_a:
        for tb in tok_b:
            if ta.startswith(tb) or tb.startswith(ta):
                prefix_matches += 1
                break
    effective_overlap = overlap + prefix_matches
    if union > 0 and effective_overlap / union >= 0.5:
        return True

    return False


def find_entity_duplicates(entities_dir: Path) -> list[list[str]]:
    """Find entity pages that are duplicates or near-duplicates.

    Detection:
    1. Exact match: two slugs whose canonical names normalize identically
       (via ``_normalize_target``).
    2. Fuzzy match: two slugs whose name tokens have Jaccard similarity ≥ 0.6
       (catches ``"OpenAI Inc"`` vs ``"Open AI"``).

    Returns a list of groups, where each group is a list of ≥2 slugs that
    should be merged.  Groups are sorted largest-first.
    """
    index = _build_entity_index(entities_dir)
    if len(index) < 2:
        return []

    # --- Pass 1: exact normalized-name grouping ---
    norm_to_slugs: dict[str, list[str]] = {}
    for slug, entry in index.items():
        canonical = _extract_entity_name(entry["text"], slug)
        norm = _normalize_target(canonical)
        norm_to_slugs.setdefault(norm, []).append(slug)

    exact_groups = [slugs for slugs in norm_to_slugs.values() if len(slugs) > 1]

    # --- Pass 2: fuzzy token-overlap grouping ---
    slugs = list(index.keys())
    name_cache: dict[str, str] = {}
    for slug in slugs:
        name_cache[slug] = _extract_entity_name(index[slug]["text"], slug)

    fuzzy_groups: list[list[str]] = []
    used: set[str] = set()
    for i, a in enumerate(slugs):
        if a in used:
            continue
        group = [a]
        for b in slugs[i + 1:]:
            if b in used:
                continue
            if _entity_names_similar(name_cache[a], name_cache[b]):
                group.append(b)
                used.add(b)
        if len(group) > 1:
            used.add(a)
            fuzzy_groups.append(group)

    # Merge overlapping groups from both passes
    all_groups = exact_groups + fuzzy_groups
    merged: list[list[str]] = []
    assigned: set[str] = set()
    for group in sorted(all_groups, key=len, reverse=True):
        group_set = set(group)
        if group_set & assigned:
            continue
        merged.append(sorted(group_set))
        assigned |= group_set

    return sorted(merged, key=len, reverse=True)


def _extract_entity_name(text: str, slug: str) -> str:
    """Extract the canonical entity name from a page.

    Tries, in order:
    1. The first H1 heading (``# Entity Name``)
    2. The slug with hyphens replaced by spaces
    """
    for line in text.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return slug.replace("-", " ")


def _build_merged_entity_page(
    canonical_slug: str,
    entity_type: str,
    aliases: list[str],
    sources: list[str],
    description: str,
    existing_text: str = "",
) -> str:
    """Build a merged entity page, combining metadata from duplicates."""
    if existing_text:
        existing_meta = _read_entity_meta(existing_text)
        old_aliases = set(existing_meta.get("aliases", []))
        old_sources = set(existing_meta.get("sources", []))
        all_aliases = sorted(old_aliases | set(aliases))
        all_sources = sorted(old_sources | set(sources))
        desc = description or existing_meta.get("brief", "")

        fm_lines = [f"type: {entity_type}"]
        if all_aliases:
            fm_lines.append(f"aliases: [{', '.join(all_aliases)}]")
        if all_sources:
            fm_lines.append(f"sources: [{', '.join(all_sources)}]")
        if desc:
            fm_lines.append(f"brief: {desc}")
        frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n"

        # Preserve existing body, add missing mentions
        body_end = existing_text.find("---", existing_text.find("---", 3) + 3) if existing_text.count("---") >= 4 else -1
        if body_end != -1:
            body = existing_text[body_end + 3:]
        else:
            _, body = existing_text.split("---", 2)[-1], ""
        if "---" in existing_text:
            parts = existing_text.split("---")
            body = "---".join(parts[2:]) if len(parts) > 2 else ""
        else:
            body = existing_text

        # Add missing source mentions
        for src in sources:
            link = f"[[summaries/{src}]]"
            if link not in body and "## Mentions" in body:
                body = body.rstrip() + f"\n- {link}"

        return frontmatter + body

    # New page
    fm_lines = [f"type: {entity_type}"]
    if aliases:
        fm_lines.append(f"aliases: [{', '.join(aliases)}]")
    if sources:
        source_links = [f"summaries/{s}" for s in sources]
        fm_lines.append(f"sources: [{', '.join(source_links)}]")
    if description:
        fm_lines.append(f"brief: {description}")
    frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n"

    heading = canonical_slug.replace("-", " ").title()
    body = f"\n# {heading}\n"
    if description:
        body += f"\n{description}\n"
    body += "\n## Mentions\n"
    for src in sources:
        body += f"- [[summaries/{src}]]\n"
    body += "\n## Related Entities\n"
    body += "\n## Related Concepts\n"

    return frontmatter + body


def _rebuild_entity_index(wiki: Path) -> None:
    """Rebuild wiki/entities/index.md from entity pages on disk."""
    # Canonical type order (mirrors entity_extractor.ENTITY_TYPES keys)
    _ENTITY_TYPE_ORDER = [
        "PERSON", "ORGANIZATION", "LOCATION", "FACILITY", "EVENT",
        "DATE", "TIME", "MONEY", "QUANTITY", "PRODUCT", "WORK_OF_ART",
        "CONCEPT", "TECHNOLOGY", "JOB_TITLE", "LAW", "LANGUAGE",
        "NATIONALITY", "IDENTIFIER", "FILE", "MATERIAL",
    ]

    entities_dir = wiki / "entities"
    if not entities_dir.is_dir():
        return

    # Parse all entity pages
    entries: dict[str, list[str]] = {}
    for path in sorted(entities_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        meta = _read_entity_meta(_read_md(path))
        etype = meta.get("type", "CONCEPT")
        desc = meta.get("brief", "")
        entry = f"- [[entities/{path.stem}]]"
        if desc:
            entry += f" — {desc}"
        entries.setdefault(etype, []).append(entry)

    lines = ["# Entity Index\n"]
    for etype in _ENTITY_TYPE_ORDER:
        type_entries = entries.get(etype, [])
        if type_entries:
            lines.append(f"\n## {etype}")
            lines.extend(sorted(type_entries))

    (entities_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fix_entity_duplicates(wiki: Path) -> tuple[int, list[str]]:
    """Merge duplicate entity pages in place.

    For each group of duplicates, the page with a description (or longest
    name) is kept as canonical.  Aliases and sources from all pages in the
    group are merged into the canonical page.  Duplicate files are deleted
    and the entity index is rebuilt.

    Self-contained — does not import from entity_writer or entity_extractor
    to avoid pulling in heavy dependencies (litellm, gliner2).

    Args:
        wiki: Path to the wiki root directory.

    Returns:
        Tuple of ``(duplicates_removed, merge_descriptions)``.
    """
    entities_dir = wiki / "entities"
    if not entities_dir.is_dir():
        return 0, []

    groups = find_entity_duplicates(entities_dir)
    if not groups:
        return 0, []

    index = _build_entity_index(entities_dir)
    total_removed = 0
    descriptions: list[str] = []

    for group in groups:
        # Pick canonical: prefer page with a description, then longest name
        def _score(slug: str) -> tuple[int, int]:
            meta = index[slug]["meta"]
            has_desc = 1 if meta.get("brief") else 0
            name_len = len(slug)
            return (has_desc, name_len)

        group_sorted = sorted(group, key=_score, reverse=True)
        canonical_slug = group_sorted[0]
        duplicate_slugs = group_sorted[1:]

        # Collect all aliases and sources from the group
        all_aliases: set[str] = set()
        all_sources: set[str] = set()
        description = ""
        entity_type = "CONCEPT"

        for slug in group:
            meta = index[slug]["meta"]
            all_aliases.update(meta.get("aliases", []))
            all_sources.update(meta.get("sources", []))
            if not description and meta.get("brief"):
                description = meta["brief"]
            if meta.get("type") and slug == canonical_slug:
                entity_type = meta["type"]

        # Remove canonical slug from aliases
        all_aliases.discard(canonical_slug.replace("-", " "))
        all_aliases.discard(canonical_slug)

        # Build merged page and write
        existing = index[canonical_slug]["text"]
        merged_text = _build_merged_entity_page(
            canonical_slug, entity_type,
            sorted(all_aliases), sorted(all_sources),
            description, existing_text=existing,
        )
        (entities_dir / f"{canonical_slug}.md").write_text(
            merged_text, encoding="utf-8",
        )

        # Delete duplicate pages
        for slug in duplicate_slugs:
            dup_path = entities_dir / f"{slug}.md"
            if dup_path.exists():
                dup_path.unlink()
                total_removed += 1

        kept = f"entities/{canonical_slug}"
        removed = ", ".join(f"entities/{s}" for s in duplicate_slugs)
        descriptions.append(f"{removed} → {kept}")

    # Rebuild entity index
    _rebuild_entity_index(wiki)

    return total_removed, descriptions


def check_entity_sources(wiki: Path) -> list[str]:
    """Validate that all entity source references point to existing summary files.

    Returns a list of warning strings for each missing source.
    """
    entities_dir = wiki / "entities"
    if not entities_dir.is_dir():
        return []

    warnings: list[str] = []
    summaries_dir = wiki / "summaries"

    for path in sorted(entities_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        meta = _read_entity_meta(_read_md(path))
        for source in meta.get("sources", []):
            # source is like "summaries/paper.md"
            source_path = wiki / source
            if not source_path.exists():
                warnings.append(
                    f"entities/{path.stem} references missing source: {source}"
                )
    return warnings


def run_entity_lint(wiki: Path) -> str:
    """Run entity-specific lint checks and return a formatted report section.

    Checks:
    - Duplicate entities (exact and fuzzy name matches)
    - Missing source references
    """
    entities_dir = wiki / "entities"
    if not entities_dir.is_dir():
        return "### Entity Checks\n\nNo entities directory found.\n"

    groups = find_entity_duplicates(entities_dir)
    source_issues = check_entity_sources(wiki)

    lines = ["### Entity Checks"]

    # Duplicates
    dup_count = sum(len(g) - 1 for g in groups)
    lines.append(f"\nDuplicate entities ({dup_count} to merge in {len(groups)} group(s)):")
    if groups:
        for group in groups:
            names = ", ".join(f"entities/{s}" for s in group)
            lines.append(f"- {names}")
    else:
        lines.append("No duplicates found.")
    lines.append("")

    # Source validation
    lines.append(f"Missing entity sources ({len(source_issues)}):")
    if source_issues:
        for issue in source_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("All entity sources are valid.")

    return "\n".join(lines)


def run_structural_lint(kb_dir: Path) -> str:
    """Run all structural lint checks and return a formatted Markdown report.

    Args:
        kb_dir: Root of the knowledge base (contains wiki/ and raw/).

    Returns:
        Formatted Markdown string with lint results.
    """
    wiki = kb_dir / "wiki"
    raw = kb_dir / "raw"

    broken = find_broken_links(wiki)
    orphans = find_orphans(wiki)
    missing = find_missing_entries(raw, wiki)
    sync_issues = check_index_sync(wiki)

    lines = ["## Structural Lint Report\n"]

    # Broken links
    lines.append(f"### Broken Links ({len(broken)})")
    if broken:
        for issue in broken:
            lines.append(f"- {issue}")
    else:
        lines.append("No broken links found.")
    lines.append("")

    # Orphans
    lines.append(f"### Orphaned Pages ({len(orphans)})")
    if orphans:
        for page in orphans:
            lines.append(f"- {page}")
    else:
        lines.append("No orphaned pages found.")
    lines.append("")

    # Missing entries
    lines.append(f"### Raw Files Without Wiki Entry ({len(missing)})")
    if missing:
        for name in missing:
            lines.append(f"- {name}")
    else:
        lines.append("All raw files have wiki entries.")
    lines.append("")

    # Index sync
    lines.append(f"### Index Sync Issues ({len(sync_issues)})")
    if sync_issues:
        for issue in sync_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("Index is in sync.")
    lines.append("")

    # Entity checks
    lines.append(run_entity_lint(wiki))

    return "\n".join(lines)

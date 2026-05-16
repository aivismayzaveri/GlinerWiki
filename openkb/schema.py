from __future__ import annotations

from pathlib import Path

AGENTS_MD = """\
# Wiki Schema

## Directory Structure
- sources/ — Document content. Short docs as .md, long docs as .json (per-page). Do not modify directly.
- sources/images/ — Extracted images from documents, referenced by sources.
- summaries/ — One per source document. Summary of key content.
- concepts/ — Abstract ideas, theories, methodologies extracted by GLiNER2 and reviewed by LLM.
- entities/ — Named entities extracted from documents (people, orgs, technologies, etc.).
- explorations/ — Saved query results, analyses, and comparisons worth keeping.
- reports/ — Lint health check reports. Auto-generated.

## Extraction Categories
GLiNER2 extracts all items, then LLM reviews and routes into categories:
- **Entities** (wiki/entities/): Named things — PERSON, ORGANIZATION, LOCATION, FACILITY, EVENT, PRODUCT, WORK_OF_ART, TECHNOLOGY, LANGUAGE, MATERIAL, PROJECT, AWARD, METRIC, SPECIES, DRUG
- **Concepts** (wiki/concepts/): Abstract ideas — CONCEPT type only
- **Temporal** (metadata only): DATE, TIME — embedded as `validity[]` blocks in entity pages; no own wiki pages
- **Discard**: MONEY, QUANTITY, IDENTIFIER, JOB_TITLE, NATIONALITY — filtered out (attributes, not named entities)

## Entity Validity (Bi-temporal)
Each entity page may have a `validity[]` block in its frontmatter representing when facts about the entity were true:

```
validity:
  - fact: "joined Google"
    valid_from: "2015-03"
    valid_to: "2022-12"
    recorded_at: "2026-01-15"
    source: summaries/attention-paper
  - fact: "CEO of Apple"
    valid_from: "2011-08"
    valid_to: "open"
    recorded_at: "2026-01-15"
    source: summaries/apple-report
```

- `fact`: Natural-language description of the fact (what happened)
- `valid_from`: When the fact became true (partial dates OK: "2015", "Q3 2025", "early 2020s")
- `valid_to`: When the fact stopped being true ("open" = still true)
- `recorded_at`: When this was first recorded in the wiki (jj transaction_time)
- `source`: Which summary page this fact came from

This enables temporal queries: "what changed since 2025", "what happened in 2025", "when did X happen".

## Special Files
- index.md — Content catalog: every page with link, one-line summary, organized by category.
- entities/index.md — Entity catalog grouped by type (PERSON, ORGANIZATION, etc.). No CONCEPT/DATE/TIME pages.
- log.md — Chronological append-only record of operations (ingests, queries, lints).

## Page Types
- **Summary Page** (summaries/): Key content of a single source document.
- **Concept Page** (concepts/): Abstract idea extracted by GLiNER2, reviewed by LLM. Links to related entities.
- **Entity Page** (entities/): Named entity with type, aliases, validity timeline, mentions, and cross-references to concepts.
- **Exploration Page** (explorations/): Saved query results — analyses, comparisons, syntheses.
- **Index Page** (index.md): One-liner summary of every page in the wiki. Auto-maintained.

## Index Page Format
index.md lists all documents, concepts, and explorations with metadata:
- Documents: name, one-liner description, type (short|pageindex), detail access path
- Concepts: name, one-liner description
- Explorations: name, one-liner description

## Log Format
Each log entry: `## [YYYY-MM-DD HH:MM:SS] operation | description`
Operations: ingest, query, lint

## Format
- Use [[wikilink]] to link other wiki pages (e.g., [[concepts/attention]], [[entities/tim-cook]])
- Entity pages link to related entities via ## Related Entities
- Concept pages link to related entities via ## Related Entities
- Standard Markdown heading hierarchy
- Keep each page focused on a single topic
- Do not include YAML frontmatter (---) in generated content; it is managed by code

## Entity Deduplication
`openkb lint --entity-dedup` runs an LLM agent that:
1. Reads all entity pages and builds a link-graph (who links to whom)
2. Asks LLM to find semantic duplicates (same real-world entity, different page)
3. Auto-merges: canonical page absorbs aliases/sources, all wikilinks are re-pointed
4. Deletes duplicate pages and rebuilds the entity index
"""

# Backward compat alias
SCHEMA_MD = AGENTS_MD


def get_agents_md(wiki_dir: Path) -> str:
    """Return the AGENTS.md content, reading from disk if available.

    Args:
        wiki_dir: Path to the wiki directory (containing AGENTS.md).

    Returns:
        Content of wiki_dir/AGENTS.md if it exists, otherwise the hardcoded
        AGENTS_MD default.
    """
    agents_file = wiki_dir / "AGENTS.md"
    if agents_file.exists():
        return agents_file.read_text(encoding="utf-8")
    return AGENTS_MD

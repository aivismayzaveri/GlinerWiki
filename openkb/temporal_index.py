"""Temporal indexing for OpenKB entity validity timeline.

Provides TemporalIndexer for parsing entity validity[] blocks from wiki pages,
and answering bi-temporal queries: "what changed since X", "what was true during Y".
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class EntityFact(NamedTuple):
    """A single fact asserted about an entity, anchored in time."""
    entity_slug: str
    entity_name: str
    entity_type: str
    fact: str
    valid_from: str  # partial date like "2017", "2017-03", "Q3 2025", "early 2020s"; "" = unknown
    valid_to: str    # "" = unknown, "open" = still true, "XXXX-XX" = ended
    recorded_at: str  # ISO date when this was first recorded in wiki (jj transaction_time)
    source: str      # e.g. "summaries/paper"


@dataclass
class TemporalIndex:
    """In-memory timeline index built from entity validity[] blocks.

    Supports bi-temporal queries:
    - get_facts_since(date): facts with valid_from >= date
    - get_facts_in_range(from_date, to_date): facts whose validity window overlaps range
    - get_entity_timeline(slug): all facts about one entity
    """
    entities_dir: Path
    facts: list[EntityFact] = field(default_factory=list)
    _by_slug: dict[str, list[EntityFact]] = field(default_factory=dict)

    # --------------------------------------------------------------
    # Parsing
    # --------------------------------------------------------------

    def index(self) -> None:
        """Scan all entity pages, parse their validity[] blocks, build index."""
        self.facts = []
        self._by_slug = {}
        entities_dir = self.entities_dir
        if not entities_dir.is_dir():
            return

        for path in sorted(entities_dir.glob("*.md")):
            if path.name == "index.md":
                continue
            slug = path.stem
            text = path.read_text(encoding="utf-8")
            entity_name = _extract_h1(text) or slug.replace("-", " ")
            entity_type = _extract_type(text) or "UNKNOWN"
            parsed = _parse_validity_block(text, slug, entity_name, entity_type)
            if parsed:
                self.facts.extend(parsed)
                self._by_slug[slug] = parsed

    # --------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------

    def get_facts_since(self, date: str) -> list[EntityFact]:
        """Return facts where valid_from >= date (date comparison is semantic).

        Treats partial dates as minima: "2017" >= "2016" is True, "Q3 2025" >= "2025" is True.
        """
        return [f for f in self.facts if _date_gte(f.valid_from, date)]

    def get_facts_in_range(self, from_date: str, to_date: str) -> list[EntityFact]:
        """Return facts whose validity window overlaps [from_date, to_date].

        A fact overlaps if: valid_from <= to_date AND (valid_to is empty/open OR valid_to >= from_date).
        """
        results = []
        for f in self.facts:
            if _date_lte(f.valid_from, to_date):
                if not f.valid_to or f.valid_to == "open":
                    results.append(f)
                elif _date_gte(f.valid_to, from_date):
                    results.append(f)
        return results

    def get_entity_timeline(self, slug: str) -> list[EntityFact]:
        """Return all validity facts for one entity."""
        return list(self._by_slug.get(slug, []))

    def to_json(self) -> list[dict]:
        """Serialize all facts as JSON for LLM consumption."""
        return [
            {
                "entity_slug": f.entity_slug,
                "entity_name": f.entity_name,
                "entity_type": f.entity_type,
                "fact": f.fact,
                "valid_from": f.valid_from,
                "valid_to": f.valid_to,
                "recorded_at": f.recorded_at,
                "source": f.source,
            }
            for f in self.facts
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_h1(text: str) -> str | None:
    for line in text.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return None


def _extract_type(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    for line in text[3:end].split("\n"):
        if line.startswith("type:"):
            return line[5:].strip()
    return None


_VALIDITY_ENTRY_RE = re.compile(r"^\s+-\s+fact:\s*\"(.*?)\"$")


def _parse_validity_block(
    text: str,
    entity_slug: str,
    entity_name: str,
    entity_type: str,
) -> list[EntityFact]:
    """Parse a validity[] block from an entity page.

    Looks for the YAML list under 'validity:' and extracts each entry's
    fact, valid_from, valid_to, recorded_at, source.
    """
    facts: list[EntityFact] = []
    if "validity:" not in text:
        return facts

    # Split into lines and find the validity block
    lines = text.split("\n")
    in_validity = False
    entry: dict[str, str] = {}

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "validity:":
            in_validity = True
            continue
        if in_validity:
            # End of block: unindented line
            if entry:
                fact = entry.get("fact", "")
                valid_from = entry.get("valid_from", "")
                valid_to = entry.get("valid_to", "open")
                recorded_at = entry.get("recorded_at", "")
                source = entry.get("source", "")
                if fact:
                    facts.append(EntityFact(
                        entity_slug=entity_slug,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        fact=fact,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        recorded_at=recorded_at,
                        source=source,
                    ))
                entry = {}
            if stripped and not stripped.startswith(("- ", "  ")):
                # End of validity block
                in_validity = False
                break
            if stripped.startswith("- "):
                # New entry
                continue
            if stripped.startswith("fact:"):
                entry["fact"] = stripped[5:].strip().strip('"')
            elif stripped.startswith("valid_from:"):
                entry["valid_from"] = stripped[11:].strip().strip('"')
            elif stripped.startswith("valid_to:"):
                entry["valid_to"] = stripped[10:].strip().strip('"')
            elif stripped.startswith("recorded_at:"):
                entry["recorded_at"] = stripped[13:].strip().strip('"')
            elif stripped.startswith("source:"):
                entry["source"] = stripped[7:].strip().strip('"')
    # Last entry
    if entry:
        fact = entry.get("fact", "")
        valid_from = entry.get("valid_from", "")
        valid_to = entry.get("valid_to", "open")
        recorded_at = entry.get("recorded_at", "")
        source = entry.get("source", "")
        if fact:
            facts.append(EntityFact(
                entity_slug=entity_slug,
                entity_name=entity_name,
                entity_type=entity_type,
                fact=fact,
                valid_from=valid_from,
                valid_to=valid_to,
                recorded_at=recorded_at,
                source=source,
            ))

    return facts


# ---------------------------------------------------------------------------
# Partial date comparison
# ---------------------------------------------------------------------------

def _date_gte(a: str, b: str) -> bool:
    """Return True if date string a >= b, treating partial dates as minima."""
    if not a or a == "open":
        return False
    if not b:
        return True
    norm_a = _normalize_partial_date(a)
    norm_b = _normalize_partial_date(b)
    # Pad to 10 chars (YYYY-MM-DD) with 01 for missing parts
    norm_a = _pad_partial(norm_a)
    norm_b = _pad_partial(norm_b)
    return norm_a >= norm_b


def _date_lte(a: str, b: str) -> bool:
    """Return True if date string a <= b."""
    if not a:
        return False
    if not b or b == "open":
        return True
    norm_a = _normalize_partial_date(a)
    norm_b = _normalize_partial_date(b)
    norm_a = _pad_partial(norm_a)
    norm_b = _pad_partial(norm_b)
    return norm_a <= norm_b


def _normalize_partial_date(s: str) -> str:
    """Normalize a date string to YYYY-MM-DD for comparison.

    Handles: 2017, 2017-03, 2017-03-15, Q3 2025, early 2020s, Monday, the 1990s, etc.
    Returns YYYYMMDD for exact dates, or YYYYMM00 for monthly, YYYY0000 for yearly.
    """
    s = s.strip().lower()
    # Q3 2025 → 2025-07
    m = re.match(r"q(\d)\s+(\d{4})", s)
    if m:
        quarter = int(m.group(1))
        year = int(m.group(2))
        month = (quarter - 1) * 3 + 1
        return f"{year}{month:02d}00"
    # early 2020s → 2020
    em = re.match(r"early\s+(\d{4})s?", s)
    if em:
        return f"{em.group(1)[:4]}0000"
    # late 2010s → 2019
    lm = re.match(r"late\s+(\d{4})s?", s)
    if lm:
        return f"{lm.group(1)[:4]}9999"
    # mid 2020s → 2025
    mm = re.match(r"mid\s+(\d{4})s?", s)
    if mm:
        return f"{mm.group(1)[:4]}0600"
    # "the 1990s" → 1990
    dm = re.match(r"the\s+(\d{4})s", s)
    if dm:
        return f"{dm.group(1)[:4]}0000"
    # Day of week: treat as indeterminate, return 19000101 (always <= any real date)
    if s in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        return "19000101"
    # Try dateutil for ISO parsing
    try:
        from dateutil import parser as _dateparser
        dt = _dateparser.parse(s, default=datetime(2000, 1, 1))
        return dt.strftime("%Y%m%d")
    except Exception:
        pass
    # Fallback: try YYYY-MM-DD, YYYY-MM, YYYY
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            from datetime import datetime
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y%m%d")
        except ValueError:
            pass
    return "19000101"


def _pad_partial(date_str: str) -> str:
    """Pad a YYYYMMDD string to 8 chars with zeros for missing parts."""
    date_str = date_str or ""
    if len(date_str) <= 4:
        return date_str.ljust(8, "0")
    if len(date_str) <= 6:
        return date_str.ljust(8, "0")
    return date_str.ljust(8, "0")
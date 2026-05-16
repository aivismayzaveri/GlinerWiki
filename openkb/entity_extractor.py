"""Entity extraction for OpenKB — GLiNER2 primary + LLM reviewer.

Pipeline:
  1. Sentence-aware chunking (chunks end at sentence boundaries)
  2. GLiNER2: primary NER extraction per chunk (CPU/GPU auto-detect)
  3. LLM: reviews GLiNER2 output with chunk context, corrects/merges/enriches
  4. Deduplicate across chunks → final MergedEntity list
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field

import litellm

logger = logging.getLogger(__name__)

# Entity type definitions shared by GLiNER2 schema and LLM prompt.
ENTITY_TYPES: dict[str, str] = {
    "PERSON": "Named people — individuals, fictional characters, authors, researchers",
    "ORGANIZATION": "Companies, universities, government agencies, NGOs, teams, institutions",
    "LOCATION": "Cities, countries, rivers, mountains, continents, geographical regions and features",
    "PRODUCT": "Software, hardware, products, models, versions, brands, benchmarks, datasets",
    "EVENT": "Conferences, wars, disasters, festivals, competitions, launches, announcements",
    "LAW": "Laws, acts, regulations, legal provisions, court decisions, legal frameworks",
    "FILE": "Filenames, file paths, extensions, repository names, project names",
    "MATERIAL": "Chemical elements, materials, substances, compounds, metals, minerals",
    "WORK_OF_ART": "Books, papers, songs, paintings, films, TV series, games, sculptures",
    "IDENTIFIER": "URLs, email addresses, phone numbers, ISBNs, DOIs, API keys, identifiers",
    "FACILITY": "Buildings, airports, highways, bridges, stadiums, venues, hospitals, monuments",
    "TECHNOLOGY": "Software frameworks, protocols, standards, programming languages, tools",
    "LANGUAGE": "Programming languages and natural languages",
    "PROJECT": "Research projects, software projects, initiatives, missions. Includes 'DARPA Grand Challenge', 'Project Athena', 'Apollo Program', 'Human Genome Project'.",
    "AWARD": "Prizes, grants, fellowships, recognitions. Includes 'Turing Award', 'Nobel Prize', 'ACL Best Paper', 'Google Faculty Research Award', 'MacArthur Fellowship'.",
    "METRIC": "Evaluation metrics, benchmarks, scores. Includes 'BLEU', 'ROUGE', 'F1 score', 'perplexity', 'accuracy', 'mAP', 'IoU'.",
    "SPECIES": "Biological species, organisms. Includes 'E. coli', 'Homo sapiens', 'Arabidopsis thaliana', 'Mus musculus'.",
    "DRUG": "Pharmaceutical substances, drug molecules, treatments. Includes 'aspirin', 'acetaminophen', 'ibuprofen', 'insulin', 'remdesivir'.",
}

# Types that get embedded as temporal metadata, not wiki pages.
ENTITY_TYPE_TEMPORAL: set[str] = {"DATE", "TIME"}
# Types to discard (not in our target set but may appear from GLiNER2).
ENTITY_TYPE_DISCARD: set[str] = {
    "JOB_TITLE", "NATIONALITY", "MONEY", "QUANTITY", "DATE", "TIME",
}


def _classify_category(entity_type: str) -> str:
    """Classify an entity type into a routing category."""
    etype = entity_type.upper()
    if etype in ENTITY_TYPE_TEMPORAL:
        return "temporal"
    if etype in ENTITY_TYPE_DISCARD:
        return "discard"
    return "entity"

# GLiNER2 schema — rich descriptions tuned for accurate extraction.
_GLINER_SCHEMA: dict[str, str] = {
    "person": "Named people — individuals, fictional characters, authors, researchers, inventors. Includes personal names and surnames when the person is specifically identifiable.",
    "organization": "Companies, universities, government agencies, NGOs, teams, research groups, institutions, nonprofits. Includes abbreviations like 'Google', 'MIT', 'NATO'.",
    "location": "Cities, countries, rivers, mountains, continents, geographical regions and features. Includes 'the US', 'the EU', 'the Middle East' as regions.",
    "product": "Software products, hardware devices, models, benchmarks, datasets, brands, versions. Includes 'TensorFlow', 'BERT', 'ImageNet', 'PyTorch'.",
    "event": "Conferences, wars, disasters, festivals, competitions, product launches, announcements, sports events. Includes 'WWDC', 'NeurIPS', 'World War II'.",
    "law": "Laws, acts, regulations, court decisions, legal provisions, legal frameworks. Includes 'GDPR', 'the Constitution', 'EU regulations'.",
    "file": "Filenames, file paths, repository names, project names, module names. Includes 'setup.py', 'requirements.txt', '.gitignore', 'src/'.",
    "material": "Chemical elements, materials, substances, compounds, metals, minerals. Includes 'DNA', 'RNA', 'silicon', 'gold', 'water', 'protein'.",
    "work_of_art": "Books, papers, songs, paintings, films, TV series, games, sculptures. Includes papers like 'Attention Is All You Need', books, movies, songs.",
    "identifier": "URLs, email addresses, phone numbers, ISBNs, DOIs, API keys, identifiers. Includes web URLs, 'http://...', '@email.com', 'ISBN:...'.",
    "facility": "Buildings, airports, highways, bridges, stadiums, venues, hospitals, monuments. Includes 'Google HQ', 'Heathrow Airport', 'Golden Gate Bridge'.",
    "technology": "Software frameworks, protocols, standards, tools, programming languages. Includes 'TensorFlow', 'PyTorch', 'Kubernetes', 'Docker', 'HTTP', 'TCP/IP', 'Python', 'Rust'.",
    "language": "Programming languages and natural languages. Includes 'Python', 'JavaScript', 'English', 'French', 'Mandarin'.",
    "project": "Research projects, software projects, initiatives, missions. Includes 'DARPA Grand Challenge', 'Project Athena', 'Apollo Program', 'Human Genome Project'.",
    "award": "Prizes, grants, fellowships, recognitions. Includes 'Turing Award', 'Nobel Prize', 'ACL Best Paper', 'Google Faculty Research Award', 'MacArthur Fellowship'.",
    "metric": "Evaluation metrics, benchmarks, scores. Includes 'BLEU', 'ROUGE', 'F1 score', 'perplexity', 'accuracy', 'mAP', 'IoU'.",
    "species": "Biological species, organisms, strains. Includes 'E. coli', 'Homo sapiens', 'Arabidopsis thaliana', 'Mus musculus', 'Saccharomyces cerevisiae'.",
    "drug": "Pharmaceutical substances, drug molecules, treatments, biomarkers. Includes 'aspirin', 'acetaminophen', 'ibuprofen', 'insulin', 'remdesivir', 'dexamethasone'.",
    "date": "Calendar dates, date expressions, date ranges — '2023', 'January 2024', 'Q3 2025', 'the 1990s', 'Monday'.",
    "time": "Time expressions, durations, time periods — '2 hours', '30 minutes', '3 days', 'midnight', 'every Tuesday'.",
}

# Reverse mapping: lowercase label → UPPER_CASE entity type
_LABEL_TO_TYPE: dict[str, str] = {k: k.upper() for k in _GLINER_SCHEMA}

_LLM_REVIEW_PROMPT = """\
You are an entity extraction reviewer. GLiNER2 (a named entity recognition model) \
extracted the following entities from a text chunk. Your job is to review and improve them.

## Text chunk:
{chunk_text}

## GLiNER2 extracted entities:
{entities_json}

## Your tasks:
1. **Correct types**: If an entity is misclassified, fix its type
2. **Merge duplicates**: If the same entity appears multiple times with different names, merge them
3. **Add descriptions**: Give each entity a brief description (under 100 chars)
4. **Add aliases**: List alternative names found in the text
5. **Add missing entities**: If GLiNER2 missed important entities in the text, add them

## Valid entity types:
{entity_types}

Return a JSON array of objects with keys:
- name: canonical name
- type: one of the valid types above
- description: brief description
- aliases: list of alternative names

Return ONLY valid JSON, no fences, no explanation.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """A single entity extracted by GLiNER2 or reviewed by LLM."""
    text: str
    entity_type: str
    confidence: float = 1.0
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    source: str = ""  # "gliner" or "llm-review"
    chunk_index: int = -1  # which chunk it came from
    category: str = "entity"  # "entity" | "concept" | "temporal" | "discard"
    temporal_ref: str = ""  # for DATE/TIME: what event it refers to
    related_concepts: list[str] = field(default_factory=list)  # for entities
    related_entities: list[str] = field(default_factory=list)  # for concepts
    fact_text: str = ""  # for DATE/TIME: natural-language fact this date grounds
    valid_from: str = ""  # for DATE/TIME: start date or date value
    valid_to: str = "open"  # for DATE/TIME: "open" or end date (e.g. "joined Google in 2015")


@dataclass
class MergedEntity:
    """A deduplicated entity with merged aliases and metadata."""
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 1.0
    sources: list[str] = field(default_factory=list)  # doc names
    category: str = "entity"  # "entity" | "concept" | "temporal"
    temporal_ref: str = ""  # for DATE/TIME: what event it refers to
    related_concepts: list[str] = field(default_factory=list)  # for entities
    related_entities: list[str] = field(default_factory=list)  # for concepts
    fact_text: str = ""  # for DATE/TIME: natural-language fact this date grounds
    valid_from: str = ""  # for DATE/TIME: start date or date value
    valid_to: str = "open"  # for DATE/TIME: "open" or end date


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Normalize an entity name for deduplication comparison.

    NFKC → lowercase → strip punctuation → collapse whitespace.
    """
    s = unicodedata.normalize("NFKC", name)
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_entity_slug(name: str) -> str:
    """Convert an entity name to a safe filename slug."""
    s = unicodedata.normalize("NFKC", name)
    s = s.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s or "unnamed-entity"


# ---------------------------------------------------------------------------
# Sentence-aware chunking
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Uses punctuation-based splitting: sentence ends at . ! ? followed by
    whitespace. Handles common edge cases (Mr. Dr. U.S. etc.) by requiring
    the sentence-ending punctuation to be followed by an uppercase letter
    or end of string.
    """
    # Split on sentence-ending punctuation followed by whitespace
    # This is imperfect but good enough for chunking purposes
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty strings
    return [s for s in parts if s.strip()]


def _chunk_by_sentences(
    text: str,
    max_words: int = 512,
    overlap_sentences: int = 2,
) -> list[str]:
    """Split text into overlapping chunks that end at sentence boundaries.

    Args:
        text: Input text to chunk.
        max_words: Maximum words per chunk.
        overlap_sentences: Number of sentences to overlap between chunks.

    Returns:
        List of text chunks, each ending at a sentence boundary.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    # If total text fits in one chunk, return as-is
    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= max_words:
        return [text.strip()]

    chunks = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence would exceed limit, finalize current chunk
        if current_word_count + sentence_words > max_words and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Keep overlap_sentences for next chunk
            if overlap_sentences > 0:
                overlap = current_sentences[-overlap_sentences:]
                current_sentences = list(overlap)
                current_word_count = sum(len(s.split()) for s in current_sentences)
            else:
                current_sentences = []
                current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += sentence_words

    # Add remaining sentences
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ---------------------------------------------------------------------------
# GLiNER2 extraction (primary)
# ---------------------------------------------------------------------------

_gliner_model = None


def _get_gliner_model(model_name: str = "fastino/gliner2-large-v1"):
    """Lazy-load GLiNER2 model with CPU/GPU auto-detection.

    Falls back to CPU if CUDA is not available or loading fails.
    Model stays in memory for the session duration.
    """
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    try:
        from gliner2 import GLiNER2
        logger.info("Loading GLiNER2 model: %s (device: %s)", model_name, device)
        _gliner_model = GLiNER2.from_pretrained(model_name, device=device)
        logger.info("GLiNER2 model loaded on %s.", device)
    except Exception as exc:
        logger.warning("Failed to load GLiNER2 on %s, falling back to CPU: %s", device, exc)
        if device != "cpu":
            try:
                from gliner2 import GLiNER2
                _gliner_model = GLiNER2.from_pretrained(model_name, device="cpu")
                logger.info("GLiNER2 model loaded on CPU (fallback).")
            except Exception as cpu_exc:
                logger.error("GLiNER2 failed on CPU too: %s", cpu_exc)
                raise
        else:
            raise

    return _gliner_model


def _deduplicate_gliner_spans(entities: list[dict]) -> list[dict]:
    """Deduplicate GLiNER2 results: merge overlapping spans, keep highest confidence."""
    by_key: dict[str, dict] = {}
    for ent in entities:
        norm = _normalize_name(ent["text"])
        etype = ent["entity_type"]
        key = f"{norm}|{etype}"
        if key not in by_key or ent.get("confidence", 0) > by_key[key].get("confidence", 0):
            by_key[key] = ent
    return list(by_key.values())


def extract_entities_gliner(
    text: str,
    model_name: str = "fastino/gliner2-large-v1",
    confidence_threshold: float = 0.7,
    max_words: int = 1024,
    overlap_sentences: int = 2,
) -> tuple[list[ExtractedEntity], list[str]]:
    """Extract entities from text using GLiNER2 schema-based extraction.

    Uses create_schema().entities({...}) with rich descriptions per entity
    type for significantly better accuracy than flat label lists.

    Returns:
        Tuple of (entities, chunks) — chunks are returned so the LLM reviewer
        can use them for context.
    """
    model = _get_gliner_model(model_name)
    chunks = _chunk_by_sentences(text, max_words, overlap_sentences)

    # Build schema with descriptions for better accuracy
    schema = model.create_schema().entities(_GLINER_SCHEMA)

    all_entities: list[ExtractedEntity] = []

    for chunk_idx, chunk in enumerate(chunks):
        try:
            results = model.extract(
                chunk,
                schema,
                threshold=confidence_threshold,
                include_confidence=True,
                include_spans=True,
            )
            # GLiNER2 returns {"entities": {"label": [{"text": ..., "confidence": ...}]}}
            entities_dict = results.get("entities", results) if isinstance(results, dict) else results
            for label, ent_list in entities_dict.items():
                if not isinstance(ent_list, list):
                    continue
                etype = _LABEL_TO_TYPE.get(label, label.upper())
                for ent in ent_list:
                    if not isinstance(ent, dict):
                        continue
                    text_val = ent.get("text", "").strip()
                    if text_val:
                        all_entities.append(ExtractedEntity(
                            text=text_val,
                            entity_type=etype,
                            confidence=ent.get("confidence", 0.0),
                            source="gliner",
                            chunk_index=chunk_idx,
                        ))
        except Exception as exc:
            logger.warning("GLiNER2 extraction failed on chunk %d: %s", chunk_idx, exc)

    # Deduplicate within GLiNER2 results
    deduped_raw = _deduplicate_gliner_spans([
        {"text": e.text, "entity_type": e.entity_type, "confidence": e.confidence}
        for e in all_entities
    ])

    # Rebuild ExtractedEntity list preserving chunk_index
    by_key: dict[str, ExtractedEntity] = {}
    for ent in all_entities:
        key = f"{_normalize_name(ent.text)}|{ent.entity_type}"
        if key not in by_key or ent.confidence > by_key[key].confidence:
            by_key[key] = ent

    return list(by_key.values()), chunks


# ---------------------------------------------------------------------------
# LLM review
# ---------------------------------------------------------------------------

def _format_entities_for_review(entities: list[ExtractedEntity]) -> str:
    """Format entities as JSON for the LLM review prompt."""
    items = []
    for e in entities:
        items.append({
            "name": e.text,
            "type": e.entity_type,
            "confidence": round(e.confidence, 2),
        })
    return json.dumps(items, indent=2)


_LLM_REVIEW_BATCH_PROMPT = """\
You are an entity extraction reviewer. GLiNER2 extracted the following entities from a \
document. Your job is to review, correct, merge, and enrich them.

## Document text:
{doc_text}

## GLiNER2 extracted entities:
{entities_json}

## Your tasks:
1. **Correct types**: If an entity is misclassified, fix its type using the valid types below
2. **Merge duplicates**: If the same entity appears multiple times with different names, merge them
3. **Add descriptions**: Give each entity a brief contextual description (1-2 sentences, under 100 chars) that helps distinguish it from similar entities
4. **Add aliases**: List alternative names, abbreviations, or spellings found in the text
5. **Add missing entities**: If GLiNER2 missed important named entities in the document, add them

## Valid entity types:
{entity_types}

## Relationship inference:
- For entities: set `related_entities` to any other named entity names this entity relates to (for cross-linking to entity wiki pages, e.g. "Google" relates to "Mountain View" and "Sundar Pichai")
- Do NOT use `related_concepts` — abstract concepts are created by the LLM concept planner, not by GLiNER2

## Temporal entities (DATE, TIME):
For DATE or TIME entities, ALSO extract:
- `fact`: The natural-language fact this date anchors. E.g. if the text says "Tim Cook joined Google in 2015" and the DATE entity is "2015", the fact is "joined Google". If the text says "the pandemic began in 2020" and DATE is "2020", fact is "pandemic began".
- `valid_from`: The date value itself (e.g. "2015", "Q3 2025", "early 2020s", "Monday"). Use the raw entity text.
- `valid_to`: "open" if this fact is still true, otherwise the end date if stated (e.g. "2019" for "left in 2019").

Return a JSON array of objects with keys:
- name: canonical name (use the most complete/standard form)
- type: one of the valid types above
- description: brief contextual description
- aliases: list of alternative names/abbreviations
- related_entities: list of entity names this entity relates to (for cross-linking to entity pages)
- fact: (DATE/TIME only) the natural-language fact this date anchors
- valid_from: (DATE/TIME only) the date value itself (e.g. "2015", "Q3 2025")
- valid_to: (DATE/TIME only) "open" if still true, or an end date if stated

Return ONLY valid JSON, no fences, no explanation.
"""


def review_entities_llm(
    gliner_entities: list[ExtractedEntity],
    doc_text: str,
    model: str,
    *,
    base_url: str | None = None,
) -> list[ExtractedEntity]:
    """Review GLiNER2-extracted entities using a single LLM call with full document context.

    Sends ALL GLiNER2 entities + full document text in one call, so the LLM
    can resolve cross-chunk duplicates and understand the full context.

    Args:
        gliner_entities: Entities extracted by GLiNER2.
        doc_text: Full document text for context.
        model: LLM model name (LiteLLM format).
        base_url: Optional custom endpoint URL for the entity LLM.

    Returns reviewed entities (may include new entities the LLM found).
    """
    if not gliner_entities:
        return []

    entity_types_str = ", ".join(ENTITY_TYPES.keys())
    entities_json = _format_entities_for_review(gliner_entities)

    # Truncate doc text if extremely long (>100K chars) to avoid token limits
    # The LLM has 256K+ context, but we leave room for the prompt + response
    max_doc_chars = 100_000
    if len(doc_text) > max_doc_chars:
        doc_text = doc_text[:max_doc_chars] + "\n\n[... document truncated ...]"

    prompt = _LLM_REVIEW_BATCH_PROMPT.format(
        doc_text=doc_text,
        entities_json=entities_json,
        entity_types=entity_types_str,
    )

    try:
        completion_kwargs: dict = {"max_tokens": 4096}
        if base_url:
            completion_kwargs["base_url"] = base_url
        logger.info("LLM entity review: sending %d entities + %d chars of text", len(gliner_entities), len(doc_text))
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **completion_kwargs,
        )
        msg = response.choices[0].message
        raw = (msg.content or "").strip()
        # Fallback: some proxies return content in reasoning_content
        if not raw:
            for attr in ("reasoning_content", "refusal"):
                alt = getattr(msg, attr, None)
                if alt:
                    logger.info("Entity review: empty content but %s has %d chars", attr, len(alt))
                    raw = alt.strip()
                    break
        if not raw:
            logger.warning("Entity review returned empty content (usage: %s)", response.usage)
            return gliner_entities  # Fall back to GLiNER2 entities
    except Exception as exc:
        logger.warning("LLM entity review failed: %s", exc)
        return gliner_entities  # Fall back to GLiNER2 entities

    # Parse JSON response
    cleaned = raw
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        cleaned = cleaned[first_nl + 1:] if first_nl != -1 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

    try:
        from json_repair import repair_json
        parsed = json.loads(repair_json(cleaned.strip()))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse LLM review output: %s", exc)
        return gliner_entities  # Fall back to GLiNER2 entities

    if not isinstance(parsed, list):
        logger.warning("LLM review returned non-array: %s", type(parsed).__name__)
        return gliner_entities

    reviewed: list[ExtractedEntity] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        if not name:
            continue
        etype = item.get("type", "").upper().strip()
        if etype not in ENTITY_TYPES:
            etype = "PRODUCT"  # safe fallback to a named-entity type in our schema
        category = _classify_category(etype)
        reviewed.append(ExtractedEntity(
            text=name,
            entity_type=etype,
            confidence=1.0,
            description=item.get("description", ""),
            aliases=item.get("aliases", []) or [],
            source="llm-review",
            chunk_index=0,
            category=category,
            temporal_ref=item.get("temporal_ref", "") or "",
            related_entities=item.get("related_entities", []) or [],
            fact_text=item.get("fact", "") or "",
        ))

    logger.info("LLM review: %d GLiNER2 entities → %d reviewed entities", len(gliner_entities), len(reviewed))
    return reviewed


# ---------------------------------------------------------------------------
# Final merge + deduplicate
# ---------------------------------------------------------------------------

def merge_entities(
    entities: list[ExtractedEntity],
    doc_name: str = "",
) -> list[MergedEntity]:
    """Merge and deduplicate entities from GLiNER2 + LLM review.

    Strategy:
    - Group by normalized_name (across all types — LLM may have corrected type)
    - Merge aliases from all sources
    - Keep LLM description when available
    - Keep highest confidence score
    - Prefer LLM-corrected type over GLiNER2 type
    """
    # Group by normalized name (not type — LLM may have corrected the type)
    groups: dict[str, list[ExtractedEntity]] = {}
    for ent in entities:
        key = _normalize_name(ent.text)
        groups.setdefault(key, []).append(ent)

    merged: list[MergedEntity] = []
    for key, group in groups.items():
        # Prefer LLM-reviewed entities for canonical name and type
        reviewed = [e for e in group if e.source == "llm-review"]
        if reviewed:
            canonical = reviewed[0].text
            entity_type = reviewed[0].entity_type
            description = reviewed[0].description
            category = reviewed[0].category
            temporal_ref = reviewed[0].temporal_ref
            # Merge related_entities across group
            related_entities: list[str] = []
            for e in reviewed:
                related_entities.extend(e.related_entities)
        else:
            canonical = max(group, key=lambda e: len(e.text)).text
            entity_type = group[0].entity_type
            description = ""
            category = _classify_category(entity_type)
            temporal_ref = ""
            related_entities = []

        # Merge fact_text / valid_from / valid_to for temporal entities
        fact_text = ""
        valid_from = ""
        valid_to = "open"
        if reviewed:
            for e in reviewed:
                if e.fact_text:
                    # Prefer LLM fact_text for temporal entities
                    fact_text = e.fact_text
                    valid_from = e.valid_from or e.text
                    valid_to = e.valid_to or "open"
                    break

        # Collect all aliases (excluding canonical)
        all_names: set[str] = set()
        for e in group:
            all_names.add(e.text)
            all_names.update(e.aliases)
        all_names.discard(canonical)

        # Merge confidence (highest)
        max_confidence = max(e.confidence for e in group)

        merged.append(MergedEntity(
            canonical_name=canonical,
            entity_type=entity_type,
            aliases=sorted(all_names),
            description=description,
            confidence=max_confidence,
            sources=[doc_name] if doc_name else [],
            category=category,
            temporal_ref=temporal_ref,
            related_entities=sorted(set(related_entities)),
            fact_text=fact_text,
            valid_from=valid_from,
            valid_to=valid_to,
        ))

    return sorted(merged, key=lambda e: (-e.confidence, e.canonical_name))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@dataclass
class RoutedEntities:
    """Result of routing merged entities into categories."""
    entities: list[MergedEntity]  # category == "entity" → wiki/entities/
    concepts: list[MergedEntity]  # category == "concept" → wiki/concepts/
    temporal: list[MergedEntity]  # category == "temporal" → metadata only


def _route_entities(merged: list[MergedEntity]) -> RoutedEntities:
    """Split merged entities into entity/concept/temporal categories.

    Only named entities go to wiki/entities/. Concepts are NOT extracted via
    GLiNER2 — they are created by the LLM concept planner from document context.
    DATE/TIME go to temporal metadata. Discarded types are in ENTITY_TYPE_DISCARD.
    """
    entities: list[MergedEntity] = []
    temporal: list[MergedEntity] = []

    for m in merged:
        if m.entity_type in ENTITY_TYPE_DISCARD:
            continue
        if m.category == "temporal":
            temporal.append(m)
        else:
            entities.append(m)

    return RoutedEntities(entities=entities, concepts=[], temporal=temporal)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_entities(
    text: str,
    model: str,
    doc_name: str = "",
    gliner_model: str = "fastino/gliner2-large-v1",
    confidence_threshold: float = 0.7,
    *,
    base_url: str | None = None,
) -> RoutedEntities:
    """Run entity extraction: GLiNER2 primary → LLM review → merge → route.

    Args:
        text: Document text to extract entities from.
        model: LLM model name for review (LiteLLM format).
        doc_name: Source document name for tracking.
        gliner_model: GLiNER2 model name.
        confidence_threshold: GLiNER2 confidence threshold.
        base_url: Optional custom endpoint URL for the entity LLM.

    Returns:
        RoutedEntities with .entities, .concepts, .temporal lists.
    """
    # Step 1: GLiNER2 primary extraction (runs in thread to avoid blocking)
    gliner_entities, _chunks = await asyncio.to_thread(
        extract_entities_gliner, text, gliner_model, confidence_threshold,
    )

    logger.info(
        "GLiNER2 extracted %d entities for %s",
        len(gliner_entities), doc_name,
    )

    if not gliner_entities:
        return RoutedEntities(entities=[], concepts=[], temporal=[])

    # Step 2: LLM review — single call with full document context
    reviewed_entities = await asyncio.to_thread(
        review_entities_llm, gliner_entities, text, model,
        base_url=base_url,
    )

    logger.info(
        "LLM review: %d → %d entities for %s",
        len(gliner_entities), len(reviewed_entities), doc_name,
    )

    # Step 3: Final merge + deduplicate
    merged = merge_entities(reviewed_entities, doc_name)

    # Step 4: Route into categories
    routed = _route_entities(merged)
    logger.info(
        "Routed for %s: %d entities, %d concepts, %d temporal",
        doc_name, len(routed.entities), len(routed.concepts), len(routed.temporal),
    )
    return routed

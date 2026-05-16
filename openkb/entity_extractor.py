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
    "PERSON": "People, fictional characters, named individuals",
    "ORGANIZATION": "Companies, agencies, institutions, teams",
    "LOCATION": "Cities, countries, continents, geographical features",
    "FACILITY": "Buildings, airports, highways, infrastructure, venues",
    "EVENT": "Named events, conferences, wars, disasters, meetings",
    "DATE": "Calendar dates, date expressions, date ranges",
    "TIME": "Time expressions, durations, time periods",
    "MONEY": "Monetary values with currency",
    "QUANTITY": "Measurements, counts, percentages, ratios",
    "PRODUCT": "Products, models, versions, brands",
    "WORK_OF_ART": "Books, songs, paintings, films, articles",
    "CONCEPT": "Abstract ideas, theories, methodologies, principles",
    "TECHNOLOGY": "Software, hardware, protocols, standards, frameworks",
    "JOB_TITLE": "Roles, positions, titles, professions",
    "LAW": "Laws, regulations, acts, legal provisions, standards",
    "LANGUAGE": "Programming languages or natural languages",
    "NATIONALITY": "Nationalities, ethnic groups, demographic groups",
    "IDENTIFIER": "IDs, codes, URLs, emails, phone numbers, ISBNs",
    "FILE": "Filenames, extensions, file paths",
    "MATERIAL": "Chemical elements, materials, substances, compounds",
}

# GLiNER2-friendly labels (lowercase, descriptive)
_GLINER_LABELS: dict[str, str] = {
    "PERSON": "person names, fictional characters",
    "ORGANIZATION": "companies, agencies, institutions, teams",
    "LOCATION": "cities, countries, continents, geographical features",
    "FACILITY": "buildings, airports, highways, infrastructure",
    "EVENT": "named events, conferences, wars, disasters",
    "DATE": "calendar dates, date expressions, date ranges",
    "TIME": "time expressions, durations, time periods",
    "MONEY": "monetary values with currency",
    "QUANTITY": "measurements, counts, percentages, ratios",
    "PRODUCT": "products, models, versions, brands",
    "WORK_OF_ART": "books, songs, paintings, films, articles",
    "CONCEPT": "abstract ideas, theories, methodologies",
    "TECHNOLOGY": "software, hardware, protocols, standards, frameworks",
    "JOB_TITLE": "roles, positions, titles, professions",
    "LAW": "laws, regulations, acts, legal provisions",
    "LANGUAGE": "programming languages, natural languages",
    "NATIONALITY": "nationalities, ethnic groups",
    "IDENTIFIER": "IDs, codes, URLs, emails, phone numbers",
    "FILE": "filenames, extensions, file paths",
    "MATERIAL": "chemical elements, materials, substances",
}

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


@dataclass
class MergedEntity:
    """A deduplicated entity with merged aliases and metadata."""
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 1.0
    sources: list[str] = field(default_factory=list)  # doc names


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
    confidence_threshold: float = 0.5,
    max_words: int = 512,
    overlap_sentences: int = 2,
) -> tuple[list[ExtractedEntity], list[str]]:
    """Extract entities from text using GLiNER2 with sentence-aware chunking.

    Returns:
        Tuple of (entities, chunks) — chunks are returned so the LLM reviewer
        can use them for context.
    """
    model = _get_gliner_model(model_name)
    chunks = _chunk_by_sentences(text, max_words, overlap_sentences)

    all_entities: list[ExtractedEntity] = []
    label_to_type = {v: k for k, v in _GLINER_LABELS.items()}

    for chunk_idx, chunk in enumerate(chunks):
        try:
            results = model.extract_entities(
                chunk,
                list(_GLINER_LABELS.values()),
                include_confidence=True,
                include_spans=True,
                threshold=confidence_threshold,
            )
            # GLiNER2 returns {"entities": {"label": [{"text": ..., "confidence": ...}]}}
            entities_dict = results.get("entities", results) if isinstance(results, dict) else results
            for label, ent_list in entities_dict.items():
                if not isinstance(ent_list, list):
                    continue
                etype = label_to_type.get(label, label.upper())
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


def review_entities_llm(
    gliner_entities: list[ExtractedEntity],
    chunks: list[str],
    model: str,
    *,
    base_url: str | None = None,
) -> list[ExtractedEntity]:
    """Review GLiNER2-extracted entities using an LLM with chunk context.

    For each chunk that produced entities, sends the LLM:
    - The chunk text (for context)
    - GLiNER2's extracted entities from that chunk
    - Asks LLM to correct types, merge duplicates, add descriptions, add missing

    Args:
        base_url: Optional custom endpoint URL for the entity LLM.

    Returns reviewed entities (may include new entities the LLM found).
    """
    if not gliner_entities:
        return []

    # Group entities by chunk index
    entities_by_chunk: dict[int, list[ExtractedEntity]] = {}
    for ent in gliner_entities:
        entities_by_chunk.setdefault(ent.chunk_index, []).append(ent)

    entity_types_str = ", ".join(ENTITY_TYPES.keys())
    all_reviewed: list[ExtractedEntity] = []

    for chunk_idx, chunk_entities in entities_by_chunk.items():
        chunk_text = chunks[chunk_idx] if 0 <= chunk_idx < len(chunks) else ""
        if not chunk_text:
            # No chunk context available, keep entities as-is
            all_reviewed.extend(chunk_entities)
            continue

        entities_json = _format_entities_for_review(chunk_entities)
        prompt = _LLM_REVIEW_PROMPT.format(
            chunk_text=chunk_text,
            entities_json=entities_json,
            entity_types=entity_types_str,
        )

        try:
            completion_kwargs: dict = {"max_tokens": 2048}
            if base_url:
                completion_kwargs["base_url"] = base_url
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **completion_kwargs,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("LLM entity review failed for chunk %d: %s", chunk_idx, exc)
            # Keep original GLiNER2 entities on failure
            all_reviewed.extend(chunk_entities)
            continue

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
            logger.warning("Failed to parse LLM review output for chunk %d: %s", chunk_idx, exc)
            all_reviewed.extend(chunk_entities)
            continue

        if not isinstance(parsed, list):
            all_reviewed.extend(chunk_entities)
            continue

        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            if not name:
                continue
            etype = item.get("type", "CONCEPT").upper().strip()
            if etype not in ENTITY_TYPES:
                etype = "CONCEPT"
            all_reviewed.append(ExtractedEntity(
                text=name,
                entity_type=etype,
                confidence=1.0,
                description=item.get("description", ""),
                aliases=item.get("aliases", []),
                source="llm-review",
                chunk_index=chunk_idx,
            ))

    return all_reviewed


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
        else:
            canonical = max(group, key=lambda e: len(e.text)).text
            entity_type = group[0].entity_type
            description = ""

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
        ))

    return sorted(merged, key=lambda e: (-e.confidence, e.canonical_name))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_entities(
    text: str,
    model: str,
    doc_name: str = "",
    gliner_model: str = "fastino/gliner2-large-v1",
    confidence_threshold: float = 0.5,
    *,
    base_url: str | None = None,
) -> list[MergedEntity]:
    """Run entity extraction: GLiNER2 primary → LLM review → merge.

    Args:
        text: Document text to extract entities from.
        model: LLM model name for review (LiteLLM format).
        doc_name: Source document name for tracking.
        gliner_model: GLiNER2 model name.
        confidence_threshold: GLiNER2 confidence threshold.
        base_url: Optional custom endpoint URL for the entity LLM.

    Returns:
        Deduplicated list of MergedEntity objects.
    """
    # Step 1: GLiNER2 primary extraction (runs in thread to avoid blocking)
    gliner_entities, chunks = await asyncio.to_thread(
        extract_entities_gliner, text, gliner_model, confidence_threshold,
    )

    logger.info(
        "GLiNER2 extracted %d entities for %s",
        len(gliner_entities), doc_name,
    )

    if not gliner_entities:
        return []

    # Step 2: LLM review with chunk context
    reviewed_entities = await asyncio.to_thread(
        review_entities_llm, gliner_entities, chunks, model,
        base_url=base_url,
    )

    logger.info(
        "LLM review: %d → %d entities for %s",
        len(gliner_entities), len(reviewed_entities), doc_name,
    )

    # Step 3: Final merge + deduplicate
    return merge_entities(reviewed_entities, doc_name)

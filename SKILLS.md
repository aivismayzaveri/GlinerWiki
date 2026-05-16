# OpenKB — AI Agent Skills Reference

> Complete command and architecture reference for any AI agent working with OpenKB.

---

## What OpenKB Is

OpenKB is a CLI tool that compiles raw documents (PDF, Word, Markdown, HTML, Excel, images, etc.) into a structured, interlinked wiki using LLMs. Knowledge accumulates over time — each new document enriches existing concept pages rather than sitting in isolation. Powered by PageIndex for vectorless long-document retrieval.

**Key idea**: Instead of re-deriving knowledge on every query (RAG), compile it once into a persistent wiki with summaries, concept pages, and cross-references.

---

## Install

```bash
pip install openkb                              # From PyPI
pip install git+https://github.com/aivismayzaveri/GlinerWiki.git  # Latest from GitHub (GlinerWiki fork)
pip install -e .                                 # Editable (from source clone)
```

---

## Quick Start Commands

```bash
openkb init                          # Create a new knowledge base (interactive; pre-downloads models)
openkb add paper.pdf                 # Add a single document
openkb add ~/papers/                 # Add a whole directory
openkb query "What are the findings?" # Ask a one-shot question
openkb chat                          # Interactive multi-turn chat
openkb watch                         # Watch raw/ and auto-compile new files
openkb lint                          # Structural + semantic health checks
openkb list                          # List all indexed documents
openkb status                        # Show knowledge base stats
openkb history                       # Show wiki version history (jj)
openkb diff                          # Show wiki changes at latest revision (jj)
```

---

## Directory Structure

```
my-kb/
 +-- raw/                       # Drop source files here
 +-- wiki/                      # The compiled wiki (jj version-controlled)
 |    +-- index.md              # Knowledge base catalog
 |    +-- log.md                # Operations timeline
 |    +-- AGENTS.md             # Wiki schema (LLM instructions)
 |    +-- sources/              # Full-text conversions
 |    |    +-- images/          # Extracted images from documents
 |    +-- summaries/            # Per-document summaries
 |    +-- concepts/             # Cross-document synthesis pages
 |    +-- entities/             # Named entities (people, orgs, tech, etc.)
 |    +-- explorations/         # Saved query/chat results
 |    +-- reports/              # Lint reports
 |    +-- .jj/                  # jj version control (automatic)
 +-- .openkb/                   # Internal state
 |    +-- config.yaml           # Model, language, threshold
 |    +-- hashes.json           # SHA-256 dedup registry
 |    +-- chats/                # Persisted chat sessions
 +-- .env                       # LLM API key
```

---

## Architecture

```
raw/                     You drop files here
 ├── Short docs ──→ docling ────────────→ LLM reads full text
 ├── Long PDFs ──→ PageIndex ──────────→ LLM reads document trees
 │                                              │
 │                                              ▼
 │                                  Wiki Compilation (LLM)
 │                                              │
 ▼                                              ▼
wiki/
 ├── index.md, log.md, AGENTS.md
 ├── sources/        (full-text + images/)
 ├── summaries/      (per-document)
 ├── concepts/       (cross-document synthesis)
 ├── entities/       (named entities: people, orgs, tech, etc.)
 ├── explorations/   (saved query results)
 └── reports/        (lint reports)
```

**Short vs Long docs**: PDFs >= `pageindex_threshold` pages (default 20) use PageIndex tree indexing. Everything else is converted via docling. Both paths produce summary + concept pages.

**AGENTS.md** (`wiki/AGENTS.md`): The LLM's instruction manual for maintaining the wiki — defines directory conventions, page types, wikilink rules, and index format. Editable on disk; the LLM reads it at runtime so changes take effect immediately. The default template is in `openkb/schema.py`.

---

## CLI Commands Reference

### `openkb init`

Initialize a new knowledge base. Interactive — prompts for LLM model and API key. Pre-downloads Docling and GLiNER2 models on first run so `openkb add` is fast.

```bash
openkb init
# Prompts:
#   Model (default: gpt-5.4-mini) — skipped if LLM_MODEL env var is set
#   LLM API Key (saved to .env) — skipped if LLM_API_KEY env var is set
# Creates: raw/, wiki/, .openkb/, .env
# Initializes jj version control in wiki/
# Pre-downloads: Docling (OCR + SmolVLM), GLiNER2 NER model
```

### `openkb add <path>`

Add a document or directory of documents. Hash-checked — re-adding the same file is a no-op unless the previous ingestion was incomplete (missing summaries, concepts, or entities), in which case it re-processes automatically.

```bash
openkb add paper.pdf             # Single file
openkb add ~/papers/             # Directory (recursive)
```

**Supported formats**: `.pdf`, `.md`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`, `.txt`, `.csv`, images (`.png`, `.jpg`, `.tiff`, `.bmp`, `.webp`), audio, video, and more

**What happens internally**:
1. SHA-256 hash check — skip if already known AND wiki output is complete (summary exists); re-ingest if incomplete
2. Copy original to `raw/`
3. If PDF >= 20 pages → PageIndex tree indexing (long doc path)
4. Otherwise → docling conversion (short doc path)
5. LLM generates summary, concept plan, concept pages
6. Backlinks and index updated
7. jj auto-snapshots wiki changes with message "compiled: {doc_name}"

### `openkb query "question"`

One-shot Q&A over the knowledge base.

```bash
openkb query "What is attention mechanism?"
openkb query "Summarize the findings" --save    # Save answer to wiki/explorations/
openkb query "What is X?" --raw                 # Show raw markdown (no rendering)
```

**How it works**: Agent reads index.md → reads relevant summaries → reads concepts → drills into source pages if needed → synthesizes answer.

### `openkb chat`

Interactive multi-turn chat with persistent sessions.

```bash
openkb chat                       # Start new session
openkb chat --resume              # Resume most recent session
openkb chat --resume 20260411     # Resume by session id
openkb chat --list                # List all sessions
openkb chat --delete <id>         # Delete a session
openkb chat --no-color            # Disable colors
openkb chat --raw                 # Raw markdown output
```

**Slash commands inside chat**:
| Command | Description |
|---------|-------------|
| `/help` | List available commands |
| `/status` | Show knowledge base status |
| `/list` | List all documents |
| `/add <path>` | Add a document or directory |
| `/save [name]` | Export transcript to wiki/explorations/ |
| `/clear` | Start a fresh session |
| `/lint` | Run knowledge base lint |
| `/exit` | Exit (Ctrl-D also works) |

### `openkb watch`

Watch `raw/` directory for new files and auto-compile them.

```bash
openkb watch
# Press Ctrl+C to stop
# Debounce: 2 seconds (waits for file activity to settle)
```

### `openkb lint`

Run structural + semantic health checks.

```bash
openkb lint                       # Run lint
openkb lint --fix                 # Auto-fix broken wikilinks (fuzzy match)
```

**Structural checks**:
- Broken `[[wikilinks]]` pointing to non-existent pages
- Orphaned pages with no incoming or outgoing links
- Raw files without corresponding wiki entries
- Index sync issues (index.md vs actual files)

**Semantic checks** (LLM agent):
- Contradictions between pages
- Gaps in coverage
- Staleness
- Redundancy
- Missing concept pages

### `openkb list`

List all indexed documents, summaries, concepts, and reports.

```bash
openkb list
# Output:
#   Documents (3):
#     paper.pdf                                  pdf         12
#     notes.md                                   md
#   Summaries:
#     - paper
#     - notes
#   Concepts:
#     - attention
#     - transformers
```

### `openkb status`

Show knowledge base statistics.

```bash
openkb status
# Output:
#   Directory           Files
#   sources             3
#   summaries           3
#   concepts            5
#   reports             1
#   raw                 3
#   Total indexed: 3 document(s)
#   Last compile:  2026-05-15 18:01:06
```

### `openkb use <path>`

Set a directory as the default knowledge base.

```bash
openkb use ~/my-kb
```

---

## Wiki Version Control (jj)

OpenKB uses [jj (Jujutsu)](https://github.com/jj-vcs/jj) for automatic version control of the `wiki/` folder. jj auto-snapshots file changes — no manual commit needed.

### `openkb history [file]`

Show wiki version history. Optionally filter by specific file path.

```bash
openkb history                    # Show all wiki changes
openkb history concepts/attention.md  # Show changes to a specific file
openkb history -n 50              # Show last 50 revisions
```

### `openkb diff [revision]`

Show wiki changes at a specific revision.

```bash
openkb diff                       # Show changes at latest revision
openkb diff @-                    # Show changes at the previous revision
openkb diff @---                  # Show changes 3 revisions ago
```

### Direct jj Commands (inside wiki/)

For advanced temporal queries, run jj directly inside the `wiki/` directory:

```bash
cd wiki/

# View full history
jj log

# See what a file looked like at a previous revision
jj file show concepts/attention.md -r @-
jj file show summaries/paper.md -r @---

# See changes in a specific revision
jj diff -r @-
jj diff -r @---                   # 3 revisions ago

# Find revisions by date
jj log -r 'committer_date(after:"7 days ago")'
jj log -r 'committer_date(before:"2026-05-10")'

# Find revisions that touched a specific file
jj log -r 'files("concepts/attention.md")'

# See the description (commit message) of a revision
jj show @-

# Search by description
jj log -r 'description("compiled: paper")'

# Navigate relatives
jj log -r @-                      # Parent (previous revision)
jj log -r @+                      # Child (next revision)
jj log -r @---                    # 3 parents back
jj log -r ::@                     # All ancestors

# Restore a file to a previous state
jj restore concepts/attention.md --from @---

# View file list at a revision
jj file list -r @-
```

### jj Revset Quick Reference

| Expression | Meaning |
|------------|---------|
| `@` | Current working copy |
| `@-` | Parent (previous revision) |
| `@--` | Grandparent |
| `@---` | 3 revisions back |
| `::@` | All ancestors |
| `@::` | All descendants |
| `committer_date(after:"7 days ago")` | Commits from last week |
| `committer_date(before:"2026-05-01")` | Commits before May 1 |
| `files("path")` | Commits touching a file |
| `description("text")` | Commits with matching description |
| `latest(::, 10)` | Last 10 commits |

---

## Compilation Pipeline (How Wiki Creation Works)

When `openkb add paper.pdf` runs:

```
1. convert_document() — converter.py
   |  Hash check → copy to raw/ → convert to Markdown
   |  Short: docling → wiki/sources/paper.md
   |  Long:  PageIndex → wiki/sources/paper.json + wiki/summaries/paper.md
   v
2. compile_short_doc() or compile_long_doc() — agent/compiler.py
   |
   +--> LLM: generate summary (held in memory)
   +--> GLiNER2 primary extraction (sentence-aware chunks) → LLM review + merge
   +--> LLM: concepts plan (create/update/related, with entity context)
   +--> LLM: generate N new concept pages (concurrent, max 5)
   +--> LLM: rewrite M existing concept pages (concurrent)
   +--> LLM: rewrite summary with valid wikilinks (short docs only)
   +--> Code: write entity pages, add backlinks, update index
   |
   +--> jj describe "compiled: paper" (auto-snapshot)
   +--> hash registered in hashes.json
   +--> log.md updated
```

**Short vs Long documents**:
| | Short (< 20 pages) | Long (>= 20 pages) |
|---|---|---|
| Convert | docling → .md | PageIndex → .json tree |
| LLM reads | Full text | Document tree summaries |
| Entity source | Full raw text | Summary text |
| Result | summary + concepts + entities | summary + concepts + entities |

**Prompt caching**: The compiler caches system + document text at the first LLM call. Every subsequent call (plan, each concept page, summary rewrite) reuses the cache. A document creating 10 concepts bills the document tokens only once.

**Retry**: Compilation retries once on failure (2s delay) before reporting an error.

---

## Configuration

`.openkb/config.yaml`:
```yaml
model: gpt-5.4-mini              # LLM model (any LiteLLM provider)
language: en                      # Wiki output language
pageindex_threshold: 20           # PDF pages threshold for PageIndex
entity_extraction: true           # Enable entity extraction (GLiNER2 + LLM review)
entity_confidence_threshold: 0.7  # GLiNER2 confidence threshold (0.0-1.0)
entity_gliner_model: fastino/gliner2-large-v1  # GLiNER2 model
entity_llm_model: ""             # LLM for review (empty = use main model)
```

**Model format**: `provider/model` LiteLLM format. OpenAI models can omit prefix.

**Env var overrides**: `LLM_MODEL` overrides `model`, `ENTITY_LLM_MODEL` overrides `entity_llm_model`. These take priority over config.yaml on every command.

| Provider | Example |
|----------|---------|
| OpenAI | `gpt-5.4` |
| Anthropic | `anthropic/claude-sonnet-4-6` |
| Gemini | `gemini/gemini-3.1-pro-preview` |

**API key**: See [Environment Variables](#environment-variables) for key setup and load order.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `LLM_MODEL` | Live override for model — overrides config.yaml on every command |
| `LLM_API_KEY` | Primary API key — propagated to provider-specific vars automatically |
| `OPENAI_API_KEY` | Direct OpenAI key (alternative to LLM_API_KEY) |
| `ANTHROPIC_API_KEY` | Direct Anthropic key |
| `GEMINI_API_KEY` | Direct Gemini key |
| `ENTITY_LLM_MODEL` | Entity extraction LLM (optional; falls back to `entity_llm_model` in config, then main model) |
| `ENTITY_LLM_BASE_URL` | Custom endpoint URL for entity LLM (independent of main provider) |
| `PAGEINDEX_API_KEY` | Optional — enables PageIndex Cloud (OCR, faster indexing). Base PageIndex is open-source and free, uses LLM_API_KEY |
| `OPENKB_DIR` | Override KB directory (alternative to `--kb-dir` flag) |
| `NO_COLOR` | Disable colored output |

**API key load order** (first found wins): System env → KB-local `.env` → global `~/.config/openkb/.env`

**KB auto-detection**: Commands find the KB by walking up from cwd looking for `.openkb/`, then falling back to the global `default_kb` setting.

**Independent entity provider**: Set `ENTITY_LLM_MODEL` and `ENTITY_LLM_BASE_URL` to run entity extraction on a separate, cheaper provider (e.g. `openai/gpt-4.1-nano` at `https://api.openai.com/v1`) without affecting the main compilation model. Env vars override config.yaml.

---

## Global Config

`~/.config/openkb/global.yaml` stores cross-KB settings:
```yaml
default_kb: /path/to/my-kb        # Set by `openkb use`
known_kbs: [/path/to/my-kb, ...]  # Auto-populated by `openkb init` and `openkb use`
```

---

## PageIndex Cloud (Optional)

PageIndex is open-source (MIT) and runs locally for free using your `LLM_API_KEY`. For large/complex PDFs, set `PAGEINDEX_API_KEY` in `.env` to access cloud-only features: OCR for scanned PDFs (hosted VLM models), faster structure generation, and scalable indexing. Without it, PageIndex uses local standard PDF parsing.

---

## Obsidian Integration

Wiki is plain `.md` files with `[[wikilinks]]`. Open `wiki/` as an Obsidian vault for graph view, browsing, and search. Use Obsidian Web Clipper to add web articles to `raw/`.

---

## Entity Extraction

GLiNER2 (local NER model) as primary extractor using **schema-based extraction with rich descriptions** per entity type for high accuracy. LLM as reviewer. Merged and deduplicated.

**20 entity types** (each with a descriptive schema for better accuracy): PERSON, ORGANIZATION, LOCATION, FACILITY, EVENT, DATE, TIME, MONEY, QUANTITY, PRODUCT, WORK_OF_ART, CONCEPT, TECHNOLOGY, JOB_TITLE, LAW, LANGUAGE, NATIONALITY, IDENTIFIER, FILE, MATERIAL

**How it works**:
1. Text split into sentence-aware chunks (no mid-sentence breaks)
2. GLiNER2 `create_schema().entities({...})` builds a schema with descriptions per entity type — descriptions significantly improve extraction accuracy
3. `model.extract(chunk, schema, threshold=0.7)` extracts entities with confidence scores and character spans
4. LLM reviews GLiNER2 output with chunk context — corrects types, merges duplicates, adds missing entities
5. Results merged by normalized name — aliases tracked, highest confidence kept
6. Entity pages written to `wiki/entities/` with type, aliases, sources, mentions
7. Bidirectional backlinks: summary ↔ entities, entities ↔ concepts

**Schema approach** (vs flat labels): Each entity type has a rich description that tells GLiNER2 what to look for. For example, `"technology": "Software, hardware, protocols, standards, frameworks"` guides the model far better than just `"technology"`. Default confidence threshold: 0.7.

**Entity page format** (`wiki/entities/tim-cook.md`):
```yaml
---
type: PERSON
aliases: [Timothy Cook, Timothy D. Cook]
sources: [summaries/apple-annual-report.md]
brief: CEO of Apple Inc.
---
```

**Entity index** (`wiki/entities/index.md`): Grouped by type with one-line descriptions.

**Long docs**: Entity extraction runs on summary text (not raw pages) to avoid duplicate explosion.

**Independent provider**: Use `ENTITY_LLM_MODEL` + `ENTITY_LLM_BASE_URL` env vars to run entity LLM review on a separate, cheaper endpoint (e.g. OpenAI for entity review while using Anthropic for main compilation). Model fallback: env var → `entity_llm_model` in config.yaml → main model. If neither env var nor config is set, the main model handles entity extraction.

**Disable**: Set `entity_extraction: false` in `.openkb/config.yaml`.

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `cli.py` | Click CLI: init, add, query, chat, watch, lint, list, status, history, diff |
| `converter.py` | Document → Markdown conversion pipeline |
| `docling_converter.py` | Docling wrapper: PDF/Office/HTML/image conversion, page counting, image description (SmolVLM-256M) |
| `indexer.py` | PageIndex integration for long PDFs |
| `entity_extractor.py` | Dual entity extraction (GLiNER2 + LLM), merge, dedup |
| `entity_writer.py` | Entity page writing, index maintenance, backlinks |
| `agent/compiler.py` | Core: LLM wiki compilation pipeline |
| `agent/query.py` | Q&A agent (single-shot) |
| `agent/chat.py` | Interactive chat REPL |
| `agent/chat_session.py` | Chat session persistence |
| `agent/tools.py` | Wiki file tools (read, write, list, get_page_content, get_image) |
| `agent/linter.py` | Semantic lint agent |
| `lint.py` | Structural lint + ghost wikilink stripping |
| `jj.py` | Jujutsu version control integration |
| `state.py` | HashRegistry — file dedup via SHA-256 |
| `config.py` | YAML config management |
| `schema.py` | AGENTS.md template (wiki schema for LLM) |
| `images.py` | Image extraction (PDF images, base64, relative paths) |
| `tree_renderer.py` | PageIndex tree → Markdown rendering |
| `watcher.py` | Filesystem watcher (watchdog, debounced) |
| `log.py` | Append-only operation log |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pageindex | Vectorless document indexing (hierarchical tree) |
| docling | Universal document-to-markdown conversion with image description via SmolVLM-256M |
| openai-agents | Agent framework (works with any LLM via LiteLLM) |
| litellm | Multi-provider LLM gateway |
| gliner2 | Local NER model for entity extraction |
| click | CLI framework |
| watchdog | Filesystem monitoring |
| pymupdf | PDF processing (transitive via pageindex) |
| prompt_toolkit | Interactive chat input |
| rich | Terminal markdown rendering |
| json-repair | Fix malformed JSON from LLMs |

---

## Supported File Types

**Documents**: PDF, Markdown, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), HTML, plain text, CSV, AsciiDoc, LaTeX

**Images**: PNG, JPG, TIFF, BMP, WEBP — extracted inline via docling with automatic text descriptions (SmolVLM-256M vision model). OCR available for scanned documents.

**Audio/Video**: WAV, MP3, M4A, MP4, AVI, MOV — transcribed via docling's ASR pipeline (requires `asr` extra)

**Long documents**: PDFs with >= 20 pages are indexed by PageIndex into a hierarchical tree for better retrieval.

---

## Agent Tools (for query/chat)

When building agents that interact with the wiki, these tools are available:

| Tool | Args | Returns |
|------|------|---------|
| `read_file(path)` | Relative path like `summaries/paper.md` | File contents |
| `get_page_content(doc_name, pages)` | Doc name + page spec like `3-5,7` | Formatted page text |
| `get_image(image_path)` | Relative path like `sources/images/doc/p1_img1.png` | Base64 image |

---

## Wiki Page Format

**Summary pages** (`summaries/`):
```yaml
---
doc_type: short                    # or "pageindex"
full_text: sources/paper.md        # path to source content
---
# Summary content with [[wikilinks]] to concepts
```

**Concept pages** (`concepts/`):
```yaml
---
sources: [summaries/paper.md, summaries/notes.md]
brief: One-sentence definition
---
# Concept content with [[wikilinks]] to other concepts and summaries

## Related Documents
- [[summaries/paper]]
```

**Index page** (`index.md`):
```markdown
# Knowledge Base Index

## Documents
- [[summaries/paper]] (short) — One-line description
- [[summaries/thesis]] (pageindex) — One-line description

## Concepts
- [[concepts/attention]] — One-line description
- [[concepts/transformers]] — One-line description
```

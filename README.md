<div align="center">

<a href="https://openkb.ai">
  <img src="https://docs.pageindex.ai/images/openkb.png" alt="OpenKB (by PageIndex)" />
</a>

# OpenKB (GlinerWiki Fork) — Open LLM Knowledge Base

<p align="center"><i>Scale to long documents&nbsp; • &nbsp;Reasoning-based retrieval&nbsp; • &nbsp;Native multi-modality&nbsp; • &nbsp;No Vector DB&nbsp; • &nbsp;GLiNER2 Entity Extraction</i></p>

</div>

---

# 📑 What is OpenKB

**OpenKB (Open Knowledge Base)** is an open-source CLI system that compiles raw documents into a structured, interlinked wiki-style knowledge base using LLMs, powered by [**PageIndex**](https://github.com/VectifyAI/PageIndex) for vectorless long document retrieval.

This **GlinerWiki fork** extends the original [OpenKB](https://github.com/VectifyAI/OpenKB) with enhanced entity extraction (GLiNER2 + LLM dual pipeline), a unified docling-based converter, structured wiki utilities, comprehensive linting, and an AI agent skills reference.

The idea is based on a [concept](https://x.com/karpathy/status/2039805659525644595) described by Andrej Karpathy: LLMs generate summaries, concept pages, and cross-references, all maintained automatically. Knowledge compounds over time instead of being re-derived on every query.

### Why not traditional RAG?

Traditional RAG rediscovers knowledge from scratch on every query. Nothing accumulates. OpenKB compiles knowledge once into a persistent wiki, then keeps it current. Cross-references already exist. Contradictions are flagged. Synthesis reflects everything consumed.

### Features

- **Broad format support** — PDF, Word, Markdown, PowerPoint, HTML, Excel, text, images, and more via a unified [docling](https://github.com/docling-project/docling) converter
- **Scale to long documents** — Long and complex documents are handled via [PageIndex](https://github.com/VectifyAI/PageIndex) tree indexing, enabling accurate, vectorless long-context retrieval
- **Native multi-modality** — Retrieves and understands figures, tables, and images, not just text. Images are described automatically via SmolVLM vision model
- **Compiled Wiki** — LLM manages and compiles your documents into summaries, concept pages, and cross-links, all kept in sync
- **Entity Extraction (GLiNER2 + LLM)** — Dual extraction pipeline: GLiNER2 identifies 20 entity types (people, orgs, technologies, concepts, etc.) at the sentence level, then LLM reviews, corrects, and enriches with context-aware deduplication. Entities get their own wiki pages with bidirectional backlinks
- **Query** — Ask questions (one-off) against your wiki. The LLM navigates your compiled knowledge to answer
- **Interactive Chat** — Multi-turn conversations with persisted sessions you can resume across runs
- **Enhanced Lint** — Structural + semantic health checks: broken wikilinks, orphaned pages, missing entries, index sync, contradictions, gaps, and stale content
- **Watch mode** — Drop files into `raw/`, wiki updates automatically
- **Wiki version control** — Every compile auto-snapshots the wiki via [jj](https://github.com/jj-vcs/jj). Browse history, diff revisions, restore files — no manual commits
- **Obsidian compatible** — Wiki is plain `.md` files with `[[wikilinks]]`. Open in Obsidian for graph view and browsing
- **AI Agent Skills Reference** — `SKILLS.md` provides a complete command and architecture reference for AI agents working with OpenKB

# 🚀 Getting Started

### Install

```bash
pip install openkb
```

<details>
<summary><i>Other install options</i></summary>

- **Latest from GitHub:**

  ```bash
  pip install git+https://github.com/aivismayzaveri/GlinerWiki.git
  ```

- **Install from source** (editable, for development):

  ```bash
  git clone https://github.com/aivismayzaveri/GlinerWiki.git
  cd GlinerWiki
  pip install -e .
  ```

</details>

### Quick Start

```bash
# 1. Create a directory for your knowledge base
mkdir my-kb && cd my-kb

# 2. Initialize the knowledge base
openkb init

# 3. Add documents
openkb add paper.pdf
openkb add ~/papers/  # Add a whole directory

# 4. Ask a question
openkb query "What are the main findings?"

# 5. Or chat interactively
openkb chat
```

### Set up your LLM

OpenKB comes with [multi-LLM support](https://docs.litellm.ai/docs/providers) (e.g., OpenAI, Claude, Gemini) via [LiteLLM](https://github.com/BerriAI/litellm) (pinned to a [safe version](https://docs.litellm.ai/blog/security-update-march-2026)).

Set your model during `openkb init`, or in [`.openkb/config.yaml`](#configuration), using `provider/model` LiteLLM format (like `anthropic/claude-sonnet-4-6`). OpenAI models can omit the prefix (like `gpt-5.4`).

Create a `.env` file with your LLM API key:

```bash
LLM_MODEL=anthropic/claude-sonnet-4-6   # optional — skips model prompt during init
LLM_API_KEY=your_llm_api_key
```

If `LLM_MODEL` is set, `openkb init` uses it automatically without prompting. If unset, init prompts interactively.

### Independent Entity Extraction Provider

Entity extraction uses a dual pipeline: GLiNER2 (local, no API) for primary NER, then an LLM to review and enrich. You can run the entity LLM on a completely separate, cheaper provider to reduce costs:

```bash
# .env
ENTITY_LLM_MODEL=openai/gpt-4.1-nano
ENTITY_LLM_BASE_URL=https://api.openai.com/v1
```

**Model fallback** (first non-empty wins):
1. `ENTITY_LLM_MODEL` env var
2. `entity_llm_model` in `config.yaml`
3. Main `model` from `config.yaml` (default)

If `ENTITY_LLM_MODEL` is not set and `entity_llm_model` is empty in config, the main model is used for entity extraction — no separate provider needed. All other compilation (summaries, concepts, queries) always uses the primary model.

| Env Variable | Purpose |
|---|---|
| `ENTITY_LLM_MODEL` | Model for entity LLM review (LiteLLM format) |
| `ENTITY_LLM_BASE_URL` | Custom endpoint URL (independent of main provider) |

# 🧩 How OpenKB Works

### Architecture

```
raw/                              You drop files here
 │
 ├─ Short docs ──→ docling ────→ LLM reads full text
 │                                     │
 ├─ Long PDFs ──→ PageIndex ────→ LLM reads document trees
 │                                     │
 │                                     ▼
 │                         Wiki Compilation (using LLM)
 │                                     │
 ▼                                     ▼
wiki/
 ├── index.md            Knowledge base overview
 ├── log.md              Operations timeline
 ├── AGENTS.md           Wiki schema (LLM instructions)
 ├── sources/            Full-text conversions
 ├── summaries/          Per-document summaries
 ├── concepts/           Cross-document synthesis ← the good stuff
 ├── entities/           Named entities (people, orgs, technologies, etc.) — auto-generated wiki pages with backlinks
 ├── explorations/       Saved query results
 └── reports/            Lint reports
```

### Short vs. Long Document Handling

| | Short documents | Long documents (PDF ≥ 20 pages) |
|---|---|---|
| **Convert** | docling → Markdown | PageIndex → tree index + summaries |
| **Images** | Extracted + described via SmolVLM-256M (docling) | Extracted by PageIndex |
| **LLM reads** | Full text | Document trees |
| **Result** | summary + concepts | summary + concepts |

Short docs are read in full by the LLM. Long PDFs are indexed by PageIndex into a hierarchical tree with summaries. The LLM reads the tree instead of the full text, enabling better retrieval from long documents.

### Knowledge Compilation

When you add a document, the LLM:

1. Generates a **summary** page
2. Reads existing **concept** pages
3. Creates or updates concepts with cross-document synthesis
4. Updates the **index** and **log**

A single source might touch 10-15 wiki pages. Knowledge accumulates: each document enriches the existing wiki rather than sitting in isolation.

# ⚙️ Usage

### Commands

| Command | Description |
|---|---|
| `openkb init` | Initialize a new knowledge base (interactive) |
| <code>openkb&nbsp;add&nbsp;&lt;file_or_dir&gt;</code> | Add documents and compile to wiki |
| <code>openkb&nbsp;query&nbsp;"question"</code> | Ask a question over the knowledge base (use `--save` to save the answer to `wiki/explorations/`) |
| `openkb chat` | Start an interactive multi-turn chat (use `--resume`, `--list`, `--delete` to manage sessions) |
| `openkb watch` | Watch `raw/` and auto-compile new files |
| `openkb lint` | Run structural + knowledge health checks |
| `openkb list` | List indexed documents and concepts |
| `openkb status` | Show knowledge base stats |
| <code>openkb&nbsp;history&nbsp;[file]</code> | Show wiki version history (optionally filter by file) |
| <code>openkb&nbsp;diff&nbsp;[revision]</code> | Show wiki changes at a revision (default: latest) |

<!-- | `openkb lint --fix` | Auto-fix what it can | -->

### Interactive Chat

`openkb chat` opens an interactive chat session over your wiki knowledge base. Unlike the one-shot `openkb query`, each turn carries the conversation history, so you can dig into a topic without re-typing context.

```bash
openkb chat                       # start a new session
openkb chat --resume              # resume the most recent session
openkb chat --resume 20260411     # resume by id (unique prefix works)
openkb chat --list                # list all sessions
openkb chat --delete <id>         # delete a session
```

Inside a chat, type `/` to access slash commands (Tab to complete):

- `/help` — list available commands
- `/status` — show knowledge base status
- `/list` — list all documents
- `/add <path>` — add a document or directory without leaving the chat
- `/save [name]` — export the transcript to `wiki/explorations/`
- `/clear` — start a fresh session (the current one stays on disk)
- `/lint` — run knowledge base lint
- `/exit` — exit (Ctrl-D also works)

### Configuration

Settings are initialized by `openkb init`, and stored in `.openkb/config.yaml`:

```yaml
model: gpt-5.4                   # LLM model (any LiteLLM-supported provider)
language: en                     # Wiki output language
pageindex_threshold: 20          # PDF pages threshold for PageIndex
entity_extraction: true          # Enable GLiNER2 + LLM entity extraction
entity_llm_model: ""             # Entity LLM model (empty = use main model)
entity_gliner_model: "fastino/gliner2-large-v1"  # GLiNER2 model for NER
entity_confidence_threshold: 0.5 # GLiNER2 confidence cutoff
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `LLM_MODEL` | Default model for `openkb init` (skips model prompt if set) |
| `LLM_API_KEY` | Universal API key (propagated to all providers) |
| `OPENAI_API_KEY` | OpenAI-specific key |
| `ANTHROPIC_API_KEY` | Anthropic-specific key |
| `GEMINI_API_KEY` | Gemini-specific key |
| `ENTITY_LLM_MODEL` | Entity extraction model (optional, falls back to main model if unset) |
| `ENTITY_LLM_BASE_URL` | Custom endpoint for entity LLM (independent provider) |
| `PAGEINDEX_API_KEY` | PageIndex Cloud key (optional, for large PDFs) |
| `OPENKB_DIR` | Override auto-detected KB directory |
| `NO_COLOR` | Disable colored output |

Model names use `provider/model` LiteLLM [format](https://docs.litellm.ai/docs/providers) (OpenAI models can omit the prefix):

| Provider | Model example |
|---|---|
| OpenAI | `gpt-5.4` |
| Anthropic | `anthropic/claude-sonnet-4-6` |
| Gemini | `gemini/gemini-3.1-pro-preview` |

### PageIndex Integration

Long documents are challenging for LLMs due to context limits, context rot, and summarization loss.
[PageIndex](https://github.com/VectifyAI/PageIndex) solves this with vectorless, reasoning-based retrieval — building a hierarchical tree index that lets LLMs reason over the index for context-aware retrieval.

PageIndex runs locally by default using the [open-source version](https://github.com/VectifyAI/PageIndex), with no external dependencies required.

#### Optional: Cloud Support

For large or complex PDFs, [PageIndex Cloud](https://docs.pageindex.ai/) can be used to access additional capabilities, including:

- OCR support for scanned PDFs (via hosted VLM models)
- Faster structure generation
- Scalable indexing for large documents

Set `PAGEINDEX_API_KEY` in your `.env` to enable cloud features:

```
PAGEINDEX_API_KEY=your_pageindex_api_key
```

### AGENTS.md

The `wiki/AGENTS.md` file defines wiki structure and conventions. It's the LLM's instruction manual for maintaining the wiki. Customize it to change how your wiki is organized.

At runtime, the LLM reads `AGENTS.md` from disk, so your edits take effect immediately.

### Using with Obsidian

OpenKB's wiki is a directory of Markdown files with `[[wikilinks]]`. Obsidian renders it natively.

1. Open `wiki/` as an Obsidian vault
2. Browse summaries, concepts, and explorations
3. Use graph view to see knowledge connections
4. Use Obsidian Web Clipper to add web articles to `raw/`

### Wiki Version Control (jj)

OpenKB uses [jj (Jujutsu)](https://github.com/jj-vcs/jj) for automatic version control of the `wiki/` folder. Every `openkb add` auto-snapshots wiki changes — no manual commits needed.

```bash
openkb history                       # Show all wiki changes
openkb history concepts/attention.md # Changes to a specific file
openkb diff                          # Changes at latest revision
openkb diff @-                       # Changes at previous revision
```

For advanced queries, run jj directly inside `wiki/`:

```bash
cd wiki/
jj log                                        # Full history
jj file show concepts/attention.md -r @-      # File at previous revision
jj log -r 'files("concepts/attention.md")'    # Revisions touching a file
jj log -r 'description("compiled: paper")'    # Search by description
jj restore concepts/attention.md --from @---  # Restore to earlier state
```

# 🧭 Learn More

### GlinerWiki Fork — What's Added

This fork extends upstream OpenKB with:

| Module | Description |
|---|---|
| `entity_extractor.py` | Dual NER pipeline: GLiNER2 (20 entity types, CPU/GPU auto-detect) + LLM reviewer for correction, merging, and enrichment |
| `entity_writer.py` | Creates/updates entity wiki pages in `wiki/entities/`, maintains entity index by type, adds bidirectional backlinks |
| `docling_converter.py` | Unified document converter via docling — replaces markitdown + pymupdf with a single pipeline supporting PDF, DOCX, PPTX, HTML, and more |
| `wiki_utils.py` | Shared Markdown section-manipulation utilities (H2 section bounds, insert entries, ensure sections) used by compiler and entity writer |
| `jj.py` | Jujutsu (jj) version control integration for automatic wiki snapshots |
| `lint.py` | Enhanced structural linting: broken wikilinks (with fuzzy normalization), orphaned pages, missing entries, index sync checks |
| `SKILLS.md` | AI agent skills reference — complete command and architecture docs for agents working with OpenKB |
| **Independent entity provider** | `ENTITY_LLM_MODEL` + `ENTITY_LLM_BASE_URL` env vars let you run entity extraction on a separate, cheaper provider without affecting the main model |

### Compared to Karpathy's Approach

| | Karpathy's workflow | OpenKB |
|---|---|---|
| Short documents | LLM reads directly | docling → LLM reads |
| Long documents | Context limits, context rot | PageIndex tree index |
| Supported formats | Web clipper → .md | PDF, Word, PPT, Excel, HTML, text, CSV, .md |
| Wiki compilation | LLM agent | LLM agent (same) |
| Q&A | Query over wiki | Wiki + PageIndex retrieval |

### The Stack

- [PageIndex](https://github.com/VectifyAI/PageIndex) — Vectorless, reasoning-based document indexing and retrieval
- [Docling](https://github.com/docling-project/docling) — Universal document-to-markdown conversion with built-in image description via SmolVLM vision model
- [GLiNER2](https://github.com/urchade/GLiNER) — Generalist NER model for entity extraction (20 entity types, CPU/GPU auto-detect)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) — Agent framework (supports non-OpenAI models via LiteLLM)
- [LiteLLM](https://github.com/BerriAI/litellm) — Multi-provider LLM gateway
- [Click](https://click.palletsprojects.com/) — CLI framework
- [watchdog](https://github.com/gorakhargosh/watchdog) — Filesystem monitoring
- [jj (Jujutsu)](https://github.com/jj-vcs/jj) — Automatic wiki version control

### Roadmap

- [ ] Extend long document handling to non-PDF formats
- [ ] Scale to large document collections with nested folder support
- [ ] Hierarchical concept (topic) indexing for massive knowledge bases
- [ ] Database-backed storage engine
- [ ] Web UI for browsing and managing wikis

### Contributing

Contributions are welcome! Please submit a pull request, or open an [issue](https://github.com/aivismayzaveri/GlinerWiki/issues) for bugs or feature requests. For larger changes, consider opening an issue first to discuss the approach.

### License

Apache 2.0. See [LICENSE](LICENSE).

### Support Us

If you find OpenKB useful, please give us a star 🌟 — and check out [PageIndex](https://github.com/VectifyAI/PageIndex) too!  

<div>

[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/PageIndexAI)&ensp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/vectify-ai/)&ensp;
[![Contact Us](https://img.shields.io/badge/Contact_Us-3B82F6?style=for-the-badge&logo=envelope&logoColor=white)](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)

</div>

"""Docling-based document conversion for OpenKB.

Replaces both markitdown (for DOCX/PPTX/HTML/etc.) and pymupdf (for PDF)
with a single unified DocumentConverter from the docling library.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

logger = logging.getLogger(__name__)

# Lazy singleton -- first call initialises the converter (loads ML models).
_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Return a cached DocumentConverter instance with OpenKB defaults."""
    global _converter
    if _converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0
        pipeline_options.do_picture_description = True  # SmolVLM-256M image descriptions
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    return _converter


def get_pdf_page_count(path: Path) -> int:
    """Return the number of pages in the PDF using docling."""
    converter = _get_converter()
    result = converter.convert(str(path))
    return len(result.document.pages)


def convert_to_markdown(src: Path, doc_name: str, images_dir: Path) -> str:
    """Convert any supported document to markdown with inline images.

    Handles PDF, DOCX, PPTX, HTML, and all other docling-supported formats.
    Images are saved to *images_dir* and referenced with wiki-root-relative paths
    (``sources/images/{doc_name}/filename.png``).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    converter = _get_converter()
    result = converter.convert(str(src))

    # --- Extract and save element-level images (pictures, tables) -----------
    img_counter = 0
    placeholder_to_relpath: dict[str, str] = {}

    for element, _level in result.document.iterate_items():
        if not isinstance(element, (PictureItem, TableItem)):
            continue

        try:
            pil_image = element.get_image(result.document)
        except Exception:
            logger.debug("Could not extract image from %s element", type(element).__name__)
            continue
        if pil_image is None:
            continue

        img_counter += 1
        kind = "picture" if isinstance(element, PictureItem) else "table"
        filename = f"{kind}_{img_counter:03d}.png"
        save_path = images_dir / filename
        pil_image.save(str(save_path), "PNG")

        rel_path = f"sources/images/{doc_name}/{filename}"
        # Build a mapping from whatever placeholder docling uses in the
        # markdown output to our wiki-relative path.  We collect all
        # possible representations and replace them below.
        placeholder_to_relpath[f"images/{filename}"] = rel_path

    # --- Export markdown with referenced images ----------------------------
    md_path = images_dir / f"{doc_name}_tmp.md"
    result.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
    markdown = md_path.read_text(encoding="utf-8")
    md_path.unlink(missing_ok=True)

    # Rewrite image paths to wiki-root-relative format.
    # Docling may emit paths like ``images/filename.png`` or absolute paths;
    # normalise them to ``sources/images/{doc_name}/filename.png``.
    for old_fragment, new_rel in placeholder_to_relpath.items():
        markdown = markdown.replace(old_fragment, new_rel)

    # Catch any remaining docling-style image refs that didn't match exactly.
    # Pattern: ![...](images/NNN.ext) or ![...](file:///.../images/NNN.ext)
    markdown = re.sub(
        r'(!\[[^\]]*\]\()([^)]*?)(images/([^/"]+\.(?:png|jpg|jpeg|gif|bmp|webp))\))',
        lambda m: f"{m.group(1)}sources/images/{doc_name}/{m.group(4)})",
        markdown,
    )

    return markdown

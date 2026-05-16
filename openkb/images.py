"""Image extraction and copy utilities for the OpenKB converter pipeline."""
from __future__ import annotations

import base64
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

from docling_core.types.doc import PictureItem, TableItem

from openkb.docling_converter import _get_converter

logger = logging.getLogger(__name__)

# Matches: ![alt](data:image/ext;base64,DATA)
_BASE64_RE = re.compile(r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)')

# Matches: ![alt](relative/path) — excludes http(s):// and data: URIs
_RELATIVE_RE = re.compile(r'!\[([^\]]*)\]\((?!https?://|data:)([^)]+)\)')


def convert_pdf_to_pages(pdf_path: Path, doc_name: str, images_dir: Path) -> list[dict]:
    """Convert a PDF to per-page dicts with text content and images.

    Each dict has ``{"page": int, "content": str, "images": [{"path": str}]}``.
    Images are saved to *images_dir* and referenced with wiki-root-relative paths.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    converter = _get_converter()
    result = converter.convert(str(pdf_path))

    # Group content by page number
    page_data: dict[int, dict] = defaultdict(lambda: {"content_parts": [], "images": []})
    img_counter = 0

    for element, _level in result.document.iterate_items():
        # Determine page number from provenance
        page_no = 1
        if hasattr(element, "prov") and element.prov:
            page_no = element.prov[0].page_no

        # Extract text content
        if hasattr(element, "text") and element.text:
            page_data[page_no]["content_parts"].append(element.text)

        # Extract images from pictures and tables
        if isinstance(element, (PictureItem, TableItem)):
            try:
                pil_image = element.get_image(result.document)
            except Exception:
                pil_image = None
            if pil_image is not None:
                img_counter += 1
                kind = "picture" if isinstance(element, PictureItem) else "table"
                filename = f"p{page_no}_{kind}_{img_counter}.png"
                save_path = images_dir / filename
                pil_image.save(str(save_path), "PNG")
                rel_path = f"sources/images/{doc_name}/{filename}"
                page_data[page_no]["images"].append({"path": rel_path})

    # Build the output list, sorted by page number
    pages: list[dict] = []
    for page_no in sorted(page_data):
        data = page_data[page_no]
        pages.append({
            "page": page_no,
            "content": "\n".join(data["content_parts"]),
            "images": data["images"],
        })
    return pages


def extract_base64_images(markdown: str, doc_name: str, images_dir: Path) -> str:
    """Decode base64-embedded images, save to disk, and rewrite markdown links.

    For each ``![alt](data:image/ext;base64,DATA)`` match:
    - Decode base64 bytes → save to ``images_dir/img_NNN.ext``
    - Replace the link with ``![alt](sources/images/{doc_name}/img_NNN.ext)``
    - On decode failure: log a warning and leave the original text unchanged.
    """
    counter = 0
    result = markdown

    for match in _BASE64_RE.finditer(markdown):
        alt, ext, b64_data = match.group(1), match.group(2), match.group(3)
        try:
            image_bytes = base64.b64decode(b64_data, validate=True)
        except Exception:
            logger.warning(
                "Failed to decode base64 image (alt=%r, ext=%r); leaving original.",
                alt,
                ext,
            )
            continue

        counter += 1
        filename = f"img_{counter:03d}.{ext}"
        dest = images_dir / filename
        images_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image_bytes)

        new_ref = f"![{alt}](sources/images/{doc_name}/{filename})"
        result = result.replace(match.group(0), new_ref, 1)

    return result


def copy_relative_images(
    markdown: str, source_dir: Path, doc_name: str, images_dir: Path
) -> str:
    """Copy locally-referenced images into the KB images directory and rewrite links.

    For each ``![alt](relative/path)`` match (skipping http/https and data URIs):
    - Resolve path relative to ``source_dir``
    - Copy to ``images_dir/{filename}``
    - Replace link with ``![alt](sources/images/{doc_name}/{filename})``
    - Missing source file: log a warning and leave the original text unchanged.
    """
    result = markdown

    for match in _RELATIVE_RE.finditer(markdown):
        alt, rel_path = match.group(1), match.group(2)
        src = (source_dir / rel_path).resolve()
        if not src.is_relative_to(source_dir.resolve()):
            logger.warning("Image path escapes source dir: %s; skipping.", rel_path)
            continue
        if not src.exists():
            logger.warning(
                "Relative image not found: %s; leaving original link.", src
            )
            continue

        filename = src.name
        dest = images_dir / filename
        images_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

        new_ref = f"![{alt}](sources/images/{doc_name}/{filename})"
        result = result.replace(match.group(0), new_ref, 1)

    return result

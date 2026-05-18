"""Pre-download all ML models needed by openkb.

Run this once after installing dependencies:
    python -m openkb.install_models

Or import and call programmatically:
    from openkb.install_models import install_all
    install_all()
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def _eprint(msg: str):
    """Print to stderr so stdout stays clean for programmatic callers."""
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def _install_easyocr():
    """Pre-download EasyOCR models via docling's download utility.

    Forces download of english_g2 + latin_g2 recognition models and the craft
    detection model into docling's cache directory.
    """
    from docling.models.stages.ocr.easyocr_model import EasyOcrModel

    _eprint("  Downloading english_g2 + latin_g2 recognition models...")
    _eprint("  Downloading craft detection model...")
    local_dir = EasyOcrModel.download_models(
        detection_models=["craft"],
        recognition_models=["english_g2", "latin_g2"],
        force=False,
        progress=True,
    )
    _eprint(f"  Done. Cached at: {local_dir}")


def _install_gliner2():
    """Load GLiNER2 to trigger HuggingFace Hub download if not already cached."""
    import torch
    from gliner2 import GLiNER2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _eprint(f"  Loading fastino/gliner2-large-v1 on {device} (downloads if needed)...")
    GLiNER2.from_pretrained("fastino/gliner2-large-v1", device=device)
    _eprint("  Done.")


def _install_docling():
    """Warm up docling converter to download layout / table / SmolVLM models."""
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        PdfFormatOption,
        PdfPipelineOptions,
    )

    _eprint("  Warming up layout + table + picture models...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0
    pipeline_options.do_picture_description = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])
    DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    _eprint("  Done.")


def install_all():
    """Download / cache all ML models required by openkb.

    Installs:
    - EasyOCR english_g2 + latin_g2 recognition + craft detection
    - GLiNER2 fastino/gliner2-large-v1
    - Docling layout-analysis, table-structure, and SmolVLM picture models
    """
    _eprint("\n[openkb] Installing ML models...")
    _eprint("=" * 50)

    installers = [
        ("EasyOCR (english_g2 + latin_g2)", _install_easyocr),
        ("GLiNER2 (fastino/gliner2-large-v1)", _install_gliner2),
        ("Docling (layout + table + SmolVLM)", _install_docling),
    ]

    for name, fn in installers:
        _eprint(f"\n[{name}]")
        try:
            fn()
        except Exception as exc:
            _eprint(f"  ERROR: {exc}")
            raise

    _eprint("\n" + "=" * 50)
    _eprint("[openkb] All models installed successfully.")


if __name__ == "__main__":
    install_all()
"""Tests for openkb.converter."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openkb.converter import ConvertResult, convert_document, get_pdf_page_count


# ---------------------------------------------------------------------------
# get_pdf_page_count
# ---------------------------------------------------------------------------


class TestGetPdfPageCount:
    def test_returns_page_count(self, tmp_path):
        """Mock docling to return a document with 5 pages."""
        fake_result = MagicMock()
        fake_result.document.pages = {i: MagicMock() for i in range(1, 6)}
        fake_converter = MagicMock()
        fake_converter.convert.return_value = fake_result
        with patch("openkb.docling_converter._get_converter", return_value=fake_converter):
            count = get_pdf_page_count(tmp_path / "fake.pdf")
        assert count == 5


# ---------------------------------------------------------------------------
# convert_document — .md input
# ---------------------------------------------------------------------------


class TestConvertDocumentMarkdown:
    def test_md_file_copied_to_wiki_sources(self, kb_dir):
        """A .md file is read and saved under wiki/sources/."""
        src = kb_dir / "raw" / "notes.md"
        src.write_text("# Notes\n\nSome content here.", encoding="utf-8")

        result = convert_document(src, kb_dir)

        assert result.skipped is False
        assert result.is_long_doc is False
        assert result.source_path is not None
        assert result.source_path.exists()
        assert result.source_path.read_text(encoding="utf-8").startswith("# Notes")

    def test_md_duplicate_skipped(self, kb_dir):
        """Second call with same file returns skipped=True when hash is registered."""
        from openkb.state import HashRegistry

        src = kb_dir / "raw" / "notes.md"
        src.write_text("# Notes\n\nSome content here.", encoding="utf-8")

        result1 = convert_document(src, kb_dir)  # first call
        # Simulate CLI registering the hash after successful compilation
        registry = HashRegistry(kb_dir / ".openkb" / "hashes.json")
        registry.add(result1.file_hash, {"name": src.name, "type": "md"})

        result2 = convert_document(src, kb_dir)  # second call
        assert result2.skipped is True
        assert result2.source_path is None
        assert result2.raw_path is None

    def test_md_raw_file_copied(self, kb_dir):
        """The original file should also be copied to raw/."""
        src = kb_dir / "input" / "notes.md"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("# Notes\n", encoding="utf-8")

        result = convert_document(src, kb_dir)

        assert result.raw_path is not None
        assert result.raw_path.exists()


# ---------------------------------------------------------------------------
# convert_document — PDF short doc
# ---------------------------------------------------------------------------


class TestConvertDocumentPdfShort:
    def test_short_pdf_converted_via_docling(self, kb_dir, tmp_path):
        """PDF under threshold is converted with docling (convert_to_markdown)."""
        src = tmp_path / "short.pdf"
        src.write_bytes(b"%PDF-1.4 fake content")

        with (
            patch("openkb.converter.get_pdf_page_count", return_value=5),
            patch("openkb.converter.convert_to_markdown", return_value="# Short PDF\n\nConverted.") as mock_ctm,
        ):
            result = convert_document(src, kb_dir)

        mock_ctm.assert_called_once()
        assert result.skipped is False
        assert result.is_long_doc is False
        assert result.source_path is not None
        assert result.source_path.exists()


# ---------------------------------------------------------------------------
# convert_document — PDF long doc
# ---------------------------------------------------------------------------


class TestConvertDocumentPdfLong:
    def test_long_pdf_returns_is_long_doc(self, kb_dir, tmp_path):
        """PDF >= threshold pages returns is_long_doc=True, source_path=None."""
        src = tmp_path / "long.pdf"
        src.write_bytes(b"%PDF-1.4 fake long content")

        with patch("openkb.converter.get_pdf_page_count", return_value=200):
            result = convert_document(src, kb_dir)

        assert result.is_long_doc is True
        assert result.source_path is None
        assert result.skipped is False
        assert result.raw_path is not None

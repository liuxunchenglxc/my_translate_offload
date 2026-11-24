from __future__ import annotations

import io
import re
from pathlib import Path

MAX_TRANSLATION_CHARS = 1800 * 7


def _split_long_paragraph(text: str, max_chars: int = MAX_TRANSLATION_CHARS) -> list[str]:
    """Split long paragraphs at sentence boundaries, then whitespace, without dropping bytes."""
    if len(text) <= max_chars:
        return [text]
    sentence_parts = re.split(r"(?<=[.!?。！？])(?=\s)", text)
    result: list[str] = []
    current = ""
    for sentence in sentence_parts:
        if len(sentence) <= max_chars:
            if current and len(current) + len(sentence) > max_chars:
                result.append(current)
                current = sentence
            else:
                current += sentence
            continue
        if current:
            result.append(current)
            current = ""
        remainder = sentence
        while len(remainder) > max_chars:
            cut = remainder.rfind(" ", 0, max_chars + 1)
            if cut < max_chars // 2:
                cut = max_chars
            result.append(remainder[:cut])
            remainder = remainder[cut:]
        current = remainder
    if current:
        result.append(current)
    return result


def _text_blocks(text: str, markdown: bool) -> list[dict]:
    """Preserve blank-line separators and fenced code blocks as untranslated units."""
    pieces = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)", text) if markdown else [text]
    blocks: list[dict] = []
    for piece in pieces:
        if not piece:
            continue
        if (markdown and (piece.startswith("```") or piece.startswith("~~~"))):
            blocks.append({"text": piece, "translate": False})
            continue
        paragraphs = re.split(r"(\r?\n[ \t]*\r?\n+)", piece)
        for paragraph in paragraphs:
            if paragraph == "":
                continue
            if re.fullmatch(r"\r?\n[ \t]*\r?\n+", paragraph):
                blocks.append({"text": paragraph, "translate": False})
            elif not paragraph.strip():
                blocks.append({"text": paragraph, "translate": False})
            else:
                blocks.append({"text": paragraph, "translate": True})
    return blocks


def _iter_docx_paragraphs(document):
    """Yield body paragraphs and table paragraphs in document order, including nested tables."""
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    def table_paragraphs(table):
        seen_cells = set()
        for row in table.rows:
            for cell in row.cells:
                cell_key = cell._tc
                try:
                    if cell_key in seen_cells:
                        continue
                    seen_cells.add(cell_key)
                except TypeError:
                    pass
                for child in cell._tc.iterchildren():
                    if isinstance(child, CT_P):
                        yield Paragraph(child, cell)
                    elif isinstance(child, CT_Tbl):
                        yield from table_paragraphs(Table(child, cell))

    for child in document._element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield from table_paragraphs(Table(child, document))


def parse_document(data: bytes, filename: str) -> dict:
    """Parse supported documents into translatable text blocks."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        text = data.decode("utf-8-sig", errors="replace")
        return {
            "format": "markdown" if suffix in (".md", ".markdown") else "text",
            "blocks": _text_blocks(text, markdown=suffix in (".md", ".markdown")),
            "output_extension": suffix if suffix != ".markdown" else ".md",
        }

    if suffix == ".docx":
        from docx import Document
        document = Document(io.BytesIO(data))
        paragraphs = list(_iter_docx_paragraphs(document))
        return {
            "format": "docx",
            "blocks": [{"text": p.text, "translate": bool(p.text.strip())} for p in paragraphs],
            "output_extension": ".docx",
        }

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        blocks = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if page_number > 1:
                blocks.append({"text": "\n\n---\n\n", "translate": False})
            if text:
                blocks.append({"text": text, "translate": True})
        return {"format": "pdf_text", "blocks": blocks, "output_extension": ".md"}

    raise ValueError("Unsupported document type. Supported formats: .txt, .md, .docx, .pdf")


def plain_text_for_glossary(document: dict) -> str:
    return "\n".join(block["text"] for block in document["blocks"] if block.get("translate"))


def translate_block(llm, text: str, target_language: str, source_language: str,
                    glossary: list[dict]) -> str:
    """Translate a paragraph in bounded requests, preserving inline code and URLs."""
    if not text.strip():
        return text
    pieces = _split_long_paragraph(text)
    return "".join(
        llm.translate(piece, target_language, source_language, glossary)
        for piece in pieces
    )


def render_document(original_bytes: bytes, filename: str, parsed: dict, translations: list[str]) -> tuple[bytes, str, str]:
    """Rebuild a translated file. DOCX paragraph/table layout and paragraph styles are retained."""
    fmt = parsed["format"]
    if fmt == "docx":
        from docx import Document
        document = Document(io.BytesIO(original_bytes))
        paragraphs = list(_iter_docx_paragraphs(document))
        if len(paragraphs) != len(translations):
            raise RuntimeError("DOCX paragraph structure changed while translating")
        for paragraph, translated in zip(paragraphs, translations):
            if paragraph.runs:
                paragraph.runs[0].text = translated
                for run in paragraph.runs[1:]:
                    run.text = ""
            else:
                paragraph.add_run(translated)
        buffer = io.BytesIO()
        document.save(buffer)
        base = Path(filename).stem
        return buffer.getvalue(), f"{base}.translated.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    text = "".join(translations)
    extension = parsed["output_extension"]
    base = Path(filename).stem
    mime = "text/markdown; charset=utf-8" if extension == ".md" else "text/plain; charset=utf-8"
    return text.encode("utf-8"), f"{base}.translated{extension}", mime

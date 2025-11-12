from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from PIL import Image

from .logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class PageImage:
    page_number: int  # 1-based
    path: Path


@dataclass
class ExtractedImage:
    page_number: int
    image_index: int
    path: Path
    width: int
    height: int


def parse_pages(pages: Optional[str], page_count: int) -> List[int]:
    """Parse a page selection string like "1,3,10-15" into 1-based indices.
    If pages is None, returns all pages.
    """
    if not pages:
        return list(range(1, page_count + 1))
    result: List[int] = []
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if re.match(r"^\d+$", part):
            n = int(part)
            if 1 <= n <= page_count:
                result.append(n)
            continue
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            a, b = map(int, m.groups())
            if a > b:
                a, b = b, a
            a = max(1, a)
            b = min(page_count, b)
            result.extend(range(a, b + 1))
    # Dedupe keeping order
    seen = set()
    ordered = []
    for n in result:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_text(doc: fitz.Document, out_dir: Path, pages: Sequence[int]) -> None:
    ensure_dir(out_dir)
    logger.info("Extraindo texto para %s", out_dir)
    for pno in pages:
        page = doc[pno - 1]
        text = page.get_text("text")
        path = out_dir / f"page-{pno:03d}.txt"
        path.write_text(text, encoding="utf-8")
        logger.debug("Página %s -> %s", pno, path)


def extract_images(doc: fitz.Document, out_dir: Path, pages: Sequence[int]) -> List[ExtractedImage]:
    """Extract embedded images. Returns metadata for each image saved."""
    ensure_dir(out_dir)
    logger.info("Extraindo imagens embutidas para %s", out_dir)
    extracted: List[ExtractedImage] = []
    for pno in pages:
        page = doc[pno - 1]
        images = page.get_images(full=True)
        if not images:
            continue
        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            ext = base.get("ext", "png")
            width = base.get("width", 0)
            height = base.get("height", 0)
            path = out_dir / f"page-{pno:03d}-img-{img_index:04d}.{ext}"
            path.write_bytes(image_bytes)
            extracted.append(ExtractedImage(pno, img_index, path, width, height))
        logger.debug("Página %s: %s imagens extraídas", pno, len(images))
    return extracted


def render_pages(doc: fitz.Document, out_dir: Path, pages: Sequence[int], dpi: int = 300) -> List[PageImage]:
    """Render pages to PNGs for OCR. Returns list of PageImage objects."""
    ensure_dir(out_dir)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    logger.info("Rasterizando %s páginas em %s dpi=%s", len(pages), out_dir, dpi)
    res: List[PageImage] = []
    for pno in pages:
        page = doc[pno - 1]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        path = out_dir / f"page-{pno:03d}.png"
        pix.save(path.as_posix())
        res.append(PageImage(pno, path))
        logger.debug("Página %s rasterizada em %s", pno, path)
    return res


def open_document(pdf_path: Path) -> fitz.Document:
    logger.info("Abrindo PDF %s", pdf_path)
    return fitz.open(pdf_path)

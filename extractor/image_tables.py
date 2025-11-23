from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Tuple
import json
import cv2
import shutil

try:
    from paddleocr import PPStructure  # type: ignore
except ImportError:  # pragma: no cover - depende de lib opcional
    PPStructure = None  # type: ignore

from .logging_utils import get_logger
from .llm_vision import call_openai_vision_json, to_table_from_llm_payload, quick_precheck_with_cheap_llm
from .pdf_utils import open_document, parse_pages, render_pages


logger = get_logger(__name__)

_layout_engine_warning_emitted = False
_SUPPORTED_LAYOUT_LANGS = {"en", "ch"}


def _layout_engine_available() -> bool:
    """Verifica se PPStructure está disponível."""
    global _layout_engine_warning_emitted
    if PPStructure is None:
        if not _layout_engine_warning_emitted:
            logger.warning(
                "PPStructure (PaddleOCR) não instalado. Fluxo voltará a enviar página inteira ao LLM."
            )
            _layout_engine_warning_emitted = True
        return False
    return True


def _normalize_ocr_lang(lang: str) -> str:
    lang = (lang or "en").strip().lower()
    if lang in _SUPPORTED_LAYOUT_LANGS:
        return lang
    logger.warning(
        "Idioma '%s' não suportado pelos modelos de layout do PaddleOCR. "
        "Alternando automaticamente para 'en'.",
        lang,
    )
    return "en"


@lru_cache(maxsize=2)
def _get_layout_engine(lang: str) -> "PPStructure":  # type: ignore[name-defined]
    """Retorna instância cacheada do PPStructure para o idioma especificado."""
    if PPStructure is None:  # pragma: no cover - guard
        raise RuntimeError("PPStructure não disponível")
    normalized_lang = _normalize_ocr_lang(lang)
    logger.info("Inicializando PPStructure (lang=%s)", normalized_lang)
    return PPStructure(
        show_log=False,
        layout=True,
        ocr=True,
        table=True,
        recover_table=True,
        lang=normalized_lang,
    )


def _reset_layout_engine_cache() -> None:
    try:
        _get_layout_engine.cache_clear()  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _cleanup_paddle_structure_cache(lang: str) -> None:
    normalized = _normalize_ocr_lang(lang)
    base_dir = Path.home() / ".paddleocr" / "whl" / "table"
    target_dir = base_dir / f"{normalized}_ppstructure_mobile_v2.0_SLANet_infer"

    tar_name = f"{normalized}_ppstructure_mobile_v2.0_SLANet_infer.tar"
    tar_inside = target_dir / tar_name
    tar_outside = base_dir / tar_name

    for candidate in (tar_inside, tar_outside):
        if candidate.exists():
            try:
                candidate.unlink()
                logger.warning("Cache PaddleOCR: apagado arquivo %s", candidate)
            except Exception as err:  # pragma: no cover - best effort
                logger.warning("Não foi possível remover %s: %s", candidate, err)

    if target_dir.exists():
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
            logger.warning("Cache PaddleOCR: diretório removido %s", target_dir)
        except Exception as err:  # pragma: no cover - best effort
            logger.warning("Não foi possível remover %s: %s", target_dir, err)


PAGE_TABLE_PROMPT = """Esta página contém {count_desc}. Extraia CADA TABELA como entrada SEPARADA.

🔥 IMPORTANTE: Se há múltiplas tabelas, retorne CADA UMA como item separado no array "tables".

**USE HTML `<table>` PARA PRESERVAR ESTRUTURA:**
- Células mescladas: colspan/rowspan
- Cabeçalhos agrupados: múltiplas linhas em `<thead>`
- Formatação: `<sup>`, `<sub>`, `<strong>`

**Formato obrigatório:**
{{
  "type": "table_set",
  "tables": [
    {{
      "title": "Título EXATO da tabela 1",
      "format": "html",
      "html": "<table>...</table>",
      "notes": "Legendas/observações da tabela 1"
    }},
    {{
      "title": "Título EXATO da tabela 2",
      "format": "html",
      "html": "<table>...</table>",
      "notes": "Legendas/observações da tabela 2"
    }}
  ]
}}

**REGRAS CRÍTICAS:**
- ✅ Uma entrada por tabela (não misture múltiplas tabelas em um único HTML)
- ✅ Preserve TODA estrutura visual (colspan, rowspan)
- ✅ Use títulos E DADOS DEVEM SER EXATAMENTE como aparecem na imagem
- ✅ Capture notas/legendas específicas de cada tabela

Retorne APENAS JSON válido."""


CHART_PROMPT = """Extraia dados deste GRÁFICO como JSON.

**CASOS:**

1. **Diagrama ternário (triângulo):**
   - Retorne tabela com ranges para cada classe/região
   - Formato: {"type": "table", "table": {"headers": ["Classe", "Comp1 (%)", "Comp2 (%)", "Comp3 (%)"], "rows": [["Nome", "70-100", "0-15", "0-15"], ...]}}

2. **Gráfico com equações (Y = a + bX ± cX²):**
   - **EXTRAIA TODAS AS EQUAÇÕES VISÍVEIS**
   - Formato: {"type": "table", "table": {"headers": ["Painel", "a", "b", "c", "R²"], "rows": [...]}}
   - Coluna "c": SEMPRE POSITIVA (ignore sinal da equação)
   - Se múltiplos painéis, uma linha por equação

3. **Gráfico de dados (linhas/barras/dispersão):**
   - Formato: {"type": "chart", "chart": {"x": {...}, "y": {...}, "series": [{"name": "...", "values": [...]}]}}
   - Extraia todos os pontos visíveis
4. **OUTROS**
   - Se encontrar outro tipo de gráfico, compreenda e extraia os dados.
   - Formato: {"type": "chart", "chart": {"x": {...}, "y": {...}, "series": [{"name": "...", "values": [...]}]}}

**Regras:**
- Use vírgula como decimal (ex: 3400,874)
- NAO OMISSA NADA OU INVENTA DADOS

Retorne APENAS JSON válido."""


SEGMENT_TABLE_PROMPT = """Esta imagem é um recorte contendo EXATAMENTE 1 tabela.

**INSTRUÇÕES:**
- Use HTML completo `<table>` preservando colspan/rowspan
- Título deve ser o TEXTO visível na própria tabela (se houver). Se não houver, infira um título breve com base no conteúdo.
- Capture notas/legendas relacionadas ao recorte.

**Formato obrigatório:**
{
  "type": "table",
  "format": "html",
  "title": "Título exato ou inferido",
  "html": "<table>...</table>",
  "notes": "Notas específicas (opcional)"
}

Retorne somente JSON válido."""


@dataclass
class ImageProcessingConfig:
    model: str
    provider: Optional[str]
    azure_endpoint: Optional[str]
    azure_api_version: Optional[str]
    api_key: Optional[str]
    openrouter_api_key: Optional[str]
    cheap_model: Optional[str] = None
    cheap_provider: Optional[str] = None
    cheap_api_key: Optional[str] = None
    cheap_azure_endpoint: Optional[str] = None
    cheap_azure_api_version: Optional[str] = None
    locale: str = "pt-BR"
    render_dpi: int = 600
    use_cheap_precheck: bool = True
    llm_max_workers: int = 6
    use_layout_ocr: bool = True
    ocr_lang: str = "en"
    segment_padding: int = 16
    max_segments: Optional[int] = None
    fallback_to_full_page: bool = True


@dataclass
class SegmentedElement:
    element_type: str  # "table" | "chart"
    image_path: Path
    bbox: Tuple[int, int, int, int]
    index: int
    score: Optional[float] = None


# =============================================================================
# FUNÇÕES PRINCIPAIS
# =============================================================================


def process_pdf_images(
    pdf_path: Path,
    output_dir: Path,
    pages: Optional[str],
    img_dir_name: str,
    tables_dir_name: str,
    config: ImageProcessingConfig,
) -> List[Path]:
    """
    Fluxo principal de extração:
    1. Renderiza páginas do PDF (alta resolução)
    2. Pre-check com GPT-4.1 (identifica tipo e quantidade de elementos)
    3. Extração com GPT-5 (envia página inteira)
    4. Salva resultados (HTML, Excel, JSON)
    """
    doc = open_document(pdf_path)
    page_nums = parse_pages(pages, doc.page_count)
    tables_dir = output_dir / tables_dir_name
    tables_dir.mkdir(parents=True, exist_ok=True)

    results: List[Path] = []
    summary_entries: List[Dict[str, str]] = []

    # SEMPRE renderiza páginas completas (sem extrair imagens embutidas)
    logger.info("Renderizando %d páginas em DPI %d", len(page_nums), config.render_dpi)
    page_imgs = render_pages(doc, output_dir / "pages", page_nums, dpi=config.render_dpi)
    
    # Processa páginas (em paralelo se configurado)
    results.extend(
        _process_rasterized_pages(
            page_imgs, tables_dir, config, summary_entries
        )
    )

    if not results:
        logger.warning("Nenhuma tabela reconhecida para %s", pdf_path)
    elif summary_entries:
        _write_summary_html(tables_dir, summary_entries)
    
    return results


def _process_rasterized_pages(
    page_images,
    tables_dir: Path,
    config: ImageProcessingConfig,
    summary_entries: List[Dict[str, str]],
) -> List[Path]:
    outputs: List[Path] = []
    if not page_images:
        return outputs

    max_workers = max(1, config.llm_max_workers)
    
    if max_workers <= 1 or len(page_images) == 1:
        # Processa sequencialmente
        for page in page_images:
            page_outputs, page_summary = _process_single_page(
                page, tables_dir, config
            )
            outputs.extend(page_outputs)
            summary_entries.extend(page_summary)
        return outputs

    # Processa em paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_single_page, page, tables_dir, config)
            for page in page_images
        ]
        for future in as_completed(futures):
            page_outputs, page_summary = future.result()
            outputs.extend(page_outputs)
            summary_entries.extend(page_summary)
    
    return outputs


def _process_single_page(
    page,
    tables_dir: Path,
    config: ImageProcessingConfig,
) -> tuple[List[Path], List[Dict[str, str]]]:
    """Processa uma única página: pre-check + extração se necessário"""
    page_outputs: List[Path] = []
    page_summary: List[Dict[str, str]] = []
    page_id = f"{page.page_number:03d}"

    page_out = tables_dir / f"page-{page.page_number:03d}"
    page_out.mkdir(parents=True, exist_ok=True)

    # Copia imagem da página para o diretório de saída
    full_page_path = page_out / "page-full.png"
    bgr = cv2.imread(page.path.as_posix())
    if bgr is None:
        logger.warning("Falha ao carregar imagem da página %s", page.page_number)
        return page_outputs, page_summary
    
    cv2.imwrite(full_page_path.as_posix(), bgr)

    # ETAPA 1: Pre-check com LLM barata (identifica tipo e quantidade)
    has_content, content_type, content_count = _page_level_precheck(full_page_path, config)
    logger.info(
        "📋 Pre-check → has_content=%s | type=%s | count=%s",
        has_content,
        content_type,
        content_count,
    )
    
    if not has_content:
        logger.info(
            "Página %s: sem conteúdo útil (type=%s), pulando",
            page.page_number,
            content_type,
        )
        return page_outputs, page_summary

    logger.info(
        "Página %s: detectado %s (count=%d), processando...",
        page.page_number,
        content_type,
        content_count,
    )

    # ETAPA 2: Extração com GPT-5 (página inteira)
    outputs, summaries = _llm_page_to_tables(
        full_page_path,
        page_out,
        page_id,
        config,
        content_type,
        content_count,
    )
    
    page_outputs.extend(outputs)
    page_summary.extend(summaries)
    
    return page_outputs, page_summary


def _page_level_precheck(
    image_path: Path,
    config: ImageProcessingConfig,
) -> tuple[bool, str, int]:
    """
    PRE-CHECK: Usa LLM barata para identificar:
    - has_content: tem tabela/gráfico?
    - content_type: 'table', 'chart', 'text_only', 'none'
    - content_count: quantas tabelas/gráficos?
    """
    if not (config.use_cheap_precheck and config.cheap_model):
        # Se não configurado, assume que tem conteúdo
        return True, "unknown", 1
    
    cheap_provider = config.cheap_provider or config.provider
    try:
        has_content, content_type, content_count = quick_precheck_with_cheap_llm(
            image_path,
            config.cheap_model,
            cheap_provider,
            config.openrouter_api_key,
            api_key=config.cheap_api_key,
            azure_endpoint=config.cheap_azure_endpoint,
            azure_api_version=config.cheap_azure_api_version,
        )
        return has_content, content_type, content_count
    except Exception as exc:
        logger.warning(
            "Pre-check falhou para página %s (%s); assumindo conteúdo",
            image_path,
            exc,
        )
        return True, "unknown", 1


def _segment_page_elements(
    page_image_path: Path,
    page_out: Path,
    config: ImageProcessingConfig,
    content_type: str,
    expected_count: int,
) -> List[SegmentedElement]:
    """
    Segmenta a página em elementos individuais (tabelas/gráficos) usando PPStructure.
    Retorna lista de segmentos recortados em disco.
    """
    if not config.use_layout_ocr:
        return []
    if not _layout_engine_available():
        return []

    bgr = cv2.imread(page_image_path.as_posix())
    if bgr is None:
        logger.warning("PPStructure: falha ao carregar imagem %s", page_image_path)
        return []

    normalized_lang = _normalize_ocr_lang(config.ocr_lang)

    try:
        engine = _get_layout_engine(config.ocr_lang)
    except Exception as exc:  # pragma: no cover - inicialização falhou
        logger.error("Falha ao inicializar PPStructure (lang=%s): %s", normalized_lang, exc)
        if "unexpected end of data" in str(exc).lower():
            logger.warning("Cache de modelo corrompido detectado na inicialização. Limpando...")
            _cleanup_paddle_structure_cache(config.ocr_lang)
            _reset_layout_engine_cache()
            try:
                engine = _get_layout_engine(config.ocr_lang)
            except Exception as exc2:  # pragma: no cover
                logger.error(
                    "Reinicialização do PPStructure falhou novamente (lang=%s): %s",
                    normalized_lang,
                    exc2,
                )
                return []
        else:
            return []

    layout_results = None
    for attempt in range(1, 3):  # tenta no máximo 2 vezes
        try:
            logger.info(
                "⚙️ PaddleOCR: layout engine pronto (lang=%s). Executando inferência (tentativa %d)...",
                normalized_lang,
                attempt,
            )
            layout_results = engine(page_image_path.as_posix())
            break
        except Exception as exc:  # pragma: no cover - depende de lib externa
            logger.error("PPStructure falhou (tentativa %d): %s", attempt, exc)
            if "unexpected end of data" in str(exc).lower() and attempt == 1:
                logger.warning(
                    "Modelo PaddleOCR corrompido detectado. Limpando cache e baixando novamente..."
                )
                _cleanup_paddle_structure_cache(config.ocr_lang)
                _reset_layout_engine_cache()
                engine = _get_layout_engine(config.ocr_lang)
                continue
            return []

    if layout_results is None:
        return []

    segments: List[SegmentedElement] = []
    keep_types = _layout_types_for_content(content_type)

    if not isinstance(layout_results, list):
        logger.warning("PPStructure retornou formato inesperado (%s)", type(layout_results))
        return []

    for item in layout_results:
        layout_type = str(item.get("type", "")).lower()
        mapped_type = _map_layout_type(layout_type)
        if mapped_type is None:
            continue
        if keep_types and mapped_type not in keep_types:
            continue

        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        padded_bbox = _apply_padding_to_bbox(
            (x1, y1, x2, y2),
            bgr.shape[1],
            bgr.shape[0],
            config.segment_padding,
        )
        crop = _crop_image(bgr, padded_bbox)
        if crop is None:
            continue

        seg_idx = len(segments) + 1
        seg_path = page_out / f"segment-{seg_idx:02d}.png"
        cv2.imwrite(seg_path.as_posix(), crop)

        segments.append(
            SegmentedElement(
                element_type=mapped_type,
                image_path=seg_path,
                bbox=padded_bbox,
                index=seg_idx,
                score=item.get("score"),
            )
        )

        if config.max_segments and len(segments) >= config.max_segments:
            break

    if not segments:
        logger.info("PPStructure não encontrou segmentos relevantes (%s)", content_type)
        return []

    if expected_count and len(segments) != expected_count:
        logger.info(
            "PPStructure detectou %d elemento(s) vs pre-check %d",
            len(segments),
            expected_count,
        )

    logger.info("PPStructure segmentou %d elemento(s)", len(segments))
    return segments


def _layout_types_for_content(content_type: str) -> set[str]:
    mapping = {
        "table": {"table"},
        "chart": {"chart"},
        "mixed": {"table", "chart"},
    }
    return mapping.get(content_type, {"table", "chart"})


def _map_layout_type(layout_type: str) -> Optional[str]:
    if layout_type == "table":
        return "table"
    if layout_type in {"figure", "chart", "graphic"}:
        return "chart"
    return None


def _apply_padding_to_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> Tuple[int, int, int, int]:
    if padding <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return x1, y1, x2, y2


def _crop_image(
    image: "cv2.Mat",  # type: ignore[name-defined]
    bbox: Tuple[int, int, int, int],
) -> Optional["cv2.Mat"]:  # type: ignore[name-defined]
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _prompt_for_segment(segment: SegmentedElement, total: int) -> str:
    bbox = ", ".join(str(v) for v in segment.bbox)
    meta = (
        f"\n\nMETADADOS:\n"
        f"- Elemento {segment.index} de {total}\n"
        f"- Tipo esperado: {segment.element_type}\n"
        f"- BBox original (x1,y1,x2,y2): [{bbox}]\n"
        "Retorne apenas JSON válido, sem comentários ou markdown."
    )
    if segment.element_type == "table":
        return SEGMENT_TABLE_PROMPT + meta
    return CHART_PROMPT + meta


def _segment_payload_to_entries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return entries

    payload_type = payload.get("type")

    if payload_type == "table_set":
        tables = payload.get("tables") or []
        if isinstance(tables, Iterable):
            for entry in tables:
                if isinstance(entry, dict):
                    entries.append(entry)
        return entries

    if payload_type == "table":
        if payload.get("format") == "html":
            entries.append(
                {
                    "type": "table",
                    "format": "html",
                    "title": payload.get("title"),
                    "html": payload.get("html"),
                    "notes": payload.get("notes"),
                }
            )
        elif "table" in payload:
            entries.append(
                {
                    "type": "table",
                    "title": payload.get("title"),
                    "notes": payload.get("notes"),
                    "table": payload.get("table"),
                }
            )
        return entries

    if payload_type == "chart":
        chart = payload.get("chart")
        if isinstance(chart, dict):
            entries.append(
                {
                    "type": "chart",
                    "title": payload.get("title"),
                    "notes": payload.get("notes"),
                    "chart": chart,
                }
            )
    return entries


def _run_segmented_flow(
    segments: List[SegmentedElement],
    page_out: Path,
    page_id: str,
    config: ImageProcessingConfig,
) -> Optional[Dict[str, Any]]:
    total = len(segments)
    combined_entries: List[Dict[str, Any]] = []

    for segment in segments:
        instructions = _prompt_for_segment(segment, total)
        logger.info(
            "🤖 Extraindo elemento %02d/%02d (%s) via GPT-5",
            segment.index,
            total,
            segment.element_type,
        )
        logger.debug(
            "Prompt do segmento %02d:\n%s",
            segment.index,
            instructions,
        )
        payload = call_openai_vision_json(
            segment.image_path,
            model=config.model,
            provider=config.provider,
            api_key=config.api_key,
            azure_endpoint=config.azure_endpoint,
            azure_api_version=config.azure_api_version,
            openrouter_api_key=config.openrouter_api_key,
            locale=config.locale,
            instructions=instructions,
            max_retries=2,
        )

        if not payload:
            logger.warning(
                "Segmento %02d (%s) não retornou dados, ignorando.",
                segment.index,
                segment.element_type,
            )
            continue

        logger.info(
            "📩 Payload do segmento %02d recebido com %d chaves.",
            segment.index,
            len(payload.keys()) if isinstance(payload, dict) else 0,
        )
        (page_out / f"segment-{segment.index:02d}.json").write_text(
            _json_dumps(payload),
            encoding="utf-8",
        )

        entries = _segment_payload_to_entries(payload)
        if not entries:
            logger.warning(
                "Segmento %02d (%s) retornou payload sem entradas utilizáveis.",
                segment.index,
                segment.element_type,
            )
            continue

        for entry in entries:
            entry.setdefault("type", entry.get("type") or "table")
            entry.setdefault("notes", entry.get("notes"))
            entry["source"] = segment.image_path.name
            entry["bbox"] = list(segment.bbox)
            if segment.score is not None:
                entry["confidence"] = float(segment.score)
            combined_entries.append(entry)

    if not combined_entries:
        logger.warning("Fluxo segmentado não retornou dados utilizáveis na página %s", page_id)
        return None

    return {
        "type": "table_set",
        "tables": combined_entries,
        "segmentation": [
            {
                "index": segment.index,
                "type": segment.element_type,
                "bbox": list(segment.bbox),
                "image": segment.image_path.name,
                "score": segment.score,
            }
            for segment in segments
        ],
        "mode": "segmented",
    }


def _call_full_page_llm(
    page_image_path: Path,
    page_id: str,
    config: ImageProcessingConfig,
    content_type: str,
    content_count: int,
) -> Optional[Dict[str, Any]]:
    logger.info(
        "📄 Página %s: enviando imagem inteira ao GPT-5 (esperado %d %s)",
        page_id,
        content_count,
        "elemento(s)",
    )

    if content_type == "chart":
        prompt = CHART_PROMPT
    elif content_type == "mixed":
        logger.info("🔀 Conteúdo MISTO detectado - usando prompt combinado")
        prompt = f"""Esta página contém TABELAS E GRÁFICOS ({content_count} elementos no total).

**EXTRAIA TODOS OS ELEMENTOS SEPARADAMENTE:**

Para TABELAS:
- Use HTML `<table>` com colspan/rowspan
- Formato: {{"title": "...", "format": "html", "html": "<table>...</table>", "notes": "..."}}

Para GRÁFICOS:
- Extraia dados numéricos ou equações
- Formato: {{"title": "...", "type": "chart", "chart": {{...}}}}

**Formato obrigatório:**
{{
  "type": "table_set",
  "tables": [
    {{"title": "Tabela X", "format": "html", "html": "...", "notes": "..."}},
    {{"title": "Gráfico Y", "type": "chart", "chart": {{...}}}}
  ]
}}

Retorne TODAS as {content_count} elementos como entradas separadas no array "tables"."""
    else:
        count_desc = _format_count_description("table", content_count or 1)
        prompt = PAGE_TABLE_PROMPT.format(count_desc=count_desc)

    return call_openai_vision_json(
        page_image_path,
        model=config.model,
        provider=config.provider,
        api_key=config.api_key,
        azure_endpoint=config.azure_endpoint,
        azure_api_version=config.azure_api_version,
        openrouter_api_key=config.openrouter_api_key,
        locale=config.locale,
        instructions=prompt,
        max_retries=2,
    )


def _llm_page_to_tables(
    page_image_path: Path,
    page_out: Path,
    page_id: str,
    config: ImageProcessingConfig,
    content_type: str,
    content_count: int,
) -> tuple[List[Path], List[Dict[str, str]]]:
    """
    Extração via fluxo segmentado (OCR + LLM por recorte) com fallback para página inteira.
    """
    outputs: List[Path] = []
    summaries: List[Dict[str, str]] = []

    needs_review = content_count > 2
    expected_elements = max(1, content_count)

    payload: Optional[Dict[str, Any]] = None
    logger.info(
        "🧪 Iniciando segmentação PaddleOCR (type=%s, esperado=%d)",
        content_type,
        expected_elements,
    )
    segments = _segment_page_elements(
        page_image_path,
        page_out,
        config,
        content_type,
        expected_elements,
    )

    if segments:
        logger.info("📐 Fluxo segmentado: %d recorte(s) identificado(s)", len(segments))
        payload = _run_segmented_flow(segments, page_out, page_id, config)
        if not payload:
            logger.warning(
                "Fluxo segmentado não retornou dados utilizáveis na página %s.",
                page_id,
            )

    if payload is None:
        logger.info(
            "🚨 Segmentação indisponível/sem dados para página %s (segments=%s).",
            page_id,
            len(segments) if segments else 0,
        )
        if not config.fallback_to_full_page:
            logger.error(
                "Fallback desabilitado - abortando processamento da página %s.",
                page_id,
            )
            return outputs, summaries
        logger.info("🔁 Executando fallback com página inteira para a página %s.", page_id)
        payload = _call_full_page_llm(
            page_image_path,
            page_id,
            config,
            content_type,
            expected_elements,
        )

    if not payload:
        logger.warning("GPT-5 não retornou dados para página %s", page_id)
        return outputs, summaries

    if segments and payload.get("mode") is None:
        payload["mode"] = "segmented"
    elif payload.get("mode") is None:
        payload["mode"] = "fullpage"

    (page_out / "page-full.json").write_text(_json_dumps(payload), encoding="utf-8")

    if needs_review:
        extracted_count = len(_extract_tables_from_payload(payload))
        if extracted_count == 0 and payload.get("type") in ("table", "table_set", "chart"):
            extracted_count = 1

        review_file = page_out / "⚠️-CONFERIR-MANUALMENTE.txt"
        elemento_label = "tabelas" if content_type == "table" else "elementos"
        review_msg = f"""╔══════════════════════════════════════════════════════════════╗
║          ⚠️  ATENÇÃO: CONFERÊNCIA MANUAL NECESSÁRIA  ⚠️          ║
╚══════════════════════════════════════════════════════════════╝

Página: {page_id}
Detectadas pelo pre-check: {expected_elements} {elemento_label}
Extraídas pelo GPT-5: {extracted_count} elemento(s)

{'✅ OK - Quantidade bate!' if extracted_count == expected_elements else '❌ DIVERGÊNCIA - Verificar manualmente!'}

AÇÕES NECESSÁRIAS:
1. Abrir page-full.json e verificar se TODAS as tabelas foram extraídas
2. Comparar com a imagem original (page-full.png)
3. Se faltou alguma tabela, anotar para correção
4. Conferir valores nas células (principalmente números)

ARQUIVOS PARA CONFERIR:
- page-full.png ........... Imagem original
- page-full.json .......... Dados extraídos (JSON bruto)
- table-XX.xlsx ........... Tabelas formatadas (Excel)
- summary.html ............ Resumo visual

═══════════════════════════════════════════════════════════════
"""
        review_file.write_text(review_msg, encoding="utf-8")

        if extracted_count != expected_elements:
            logger.error(
                "❌ Página %s: DIVERGÊNCIA! Esperado %d elementos, extraído %d",
                page_id,
                expected_elements,
                extracted_count,
            )
            logger.error("📋 Arquivo de conferência salvo: %s", review_file.name)
        else:
            logger.info(
                "✅ Página %s: Quantidade OK (%d elementos)",
                page_id,
                extracted_count,
            )
            logger.info("📋 Conferir manualmente: %s", review_file.name)

    # Processa resposta baseado no tipo
    if payload.get("type") == "chart":
        rows = to_table_from_llm_payload(payload)
        if not rows:
            logger.warning("Gráfico sem séries interpretáveis em página %s", page_id)
            return outputs, summaries

        # Augmenta com métricas calculadas (X*, Y_max, etc) se for equação quadrática
        rows = _augment_rows_with_quadratic_metrics(rows)

        base_name = "chart-01"
        html = _save_table_outputs(rows, page_out, base_name)
        outputs.append(page_out / f"{base_name}.xlsx")
        if html:
            summaries.append({"page": page_id, "table": base_name, "html": html})
        return outputs, summaries

    # Tabela(s)
    tables = _extract_tables_from_payload(payload)
    if not tables:
        # Tenta como tabela simples
        rows = to_table_from_llm_payload(payload)
        if not rows:
            logger.warning("Nenhuma tabela interpretável em página %s", page_id)
            return outputs, summaries
        html = _save_table_outputs(rows, page_out, "table-01", notes=payload.get("notes"))
        outputs.append(page_out / "table-01.xlsx")
        if html:
            summaries.append({"page": page_id, "table": "table-01", "html": html})
        return outputs, summaries

    # Múltiplas tabelas
    chart_counter = 0
    table_counter = 0
    for info in tables:
        # NOVO: Detecta se é GRÁFICO (conteúdo misto)
        if info.get("type") == "chart":
            logger.info("📊 Gráfico extraído em conteúdo misto")
            chart_counter += 1
            chart_base = f"chart-{chart_counter:02d}"
            
            # Salva JSON do gráfico
            chart_payload = {
                "type": "chart",
                "title": info.get("title"),
                "notes": info.get("notes"),
                "chart": info.get("chart", {}),
            }
            if info.get("bbox"):
                chart_payload["bbox"] = info.get("bbox")
            if info.get("source"):
                chart_payload["source"] = info.get("source")
            (page_out / f"{chart_base}.json").write_text(_json_dumps(chart_payload), encoding="utf-8")
            
            if info.get("title"):
                (page_out / f"{chart_base}-title.txt").write_text(info["title"], encoding="utf-8")
            
            # Gráficos não geram Excel, apenas JSON
            logger.info("✅ Gráfico salvo como %s.json", chart_base)
            continue
        
        table_counter += 1
        base_name = f"table-{table_counter:02d}"

        # NOVO: Detecta se é formato HTML
        if info.get("format") == "html" and info.get("html"):
            logger.info("✅ Tabela %d em formato HTML (estrutura complexa preservada)", table_counter)
            html = _save_html_table(
                html_content=info["html"],
                out_dir=page_out,
                base_name=base_name,
                title=info.get("title"),
                notes=info.get("notes")
            )
            
            # Tenta encontrar Excel gerado (conversão automática em _save_html_table)
            excel_path = page_out / f"{base_name}.xlsx"
            if excel_path.exists():
                outputs.append(excel_path)
            
            if html:
                summaries.append({"page": page_id, "table": base_name, "html": html})
            
            # Salva JSON individual
            single_payload = {
                "type": "table",
                "format": "html",
                "title": info.get("title"),
                "notes": info.get("notes"),
                "html": info.get("html"),
            }
            if info.get("bbox"):
                single_payload["bbox"] = info.get("bbox")
            if info.get("source"):
                single_payload["source"] = info.get("source")
            (page_out / f"{base_name}.json").write_text(_json_dumps(single_payload), encoding="utf-8")
            if info.get("title"):
                (page_out / f"{base_name}-title.txt").write_text(info["title"], encoding="utf-8")
            continue
        
        # LEGADO: Formato JSON array (tabelas simples)
        rows = _normalize_table_rows(info.get("headers"), info.get("rows"))
        if not rows:
            logger.warning("Tabela %s da página %s vazia após normalização", table_counter, page_id)
            continue
        html = _save_table_outputs(rows, page_out, base_name, notes=info.get("notes"))
        outputs.append(page_out / f"{base_name}.xlsx")
        if html:
            summaries.append({"page": page_id, "table": base_name, "html": html})
        
        # Salva JSON individual
        single_payload = {
            "type": "table",
            "title": info.get("title"),
            "notes": info.get("notes"),
            "table": {
                "headers": info.get("headers"),
                "rows": info.get("rows"),
            },
        }
        if info.get("bbox"):
            single_payload["bbox"] = info.get("bbox")
        if info.get("source"):
            single_payload["source"] = info.get("source")
        (page_out / f"{base_name}.json").write_text(_json_dumps(single_payload), encoding="utf-8")
        if info.get("title"):
            (page_out / f"{base_name}-title.txt").write_text(info["title"], encoding="utf-8")
    
    return outputs, summaries


def _format_count_description(content_type: str, count: int) -> str:
    """Formata descrição de quantidade para o prompt"""
    count = max(1, int(count))
    noun = "tabela" if content_type == "table" else "gráfico"
    if count == 1:
        return f"1 {noun}"
    return f"{count} {noun}s"


def _augment_rows_with_quadratic_metrics(rows: List[List[str]]) -> List[List[str]]:
    """
    Adiciona colunas calculadas (X*, Y_max, X_90%, Y_90%) para equações quadráticas.
    Detecta se há colunas a, b, c separadas (formato novo).
    """
    if not rows or len(rows) < 2:
        return rows
    
    header = rows[0]
    header_lower = [h.lower() if isinstance(h, str) else "" for h in header]
    
    # Procura colunas de coeficientes separados
    try:
        a_idx = next(i for i, h in enumerate(header_lower) if h.strip() == "a")
        b_idx = next(i for i, h in enumerate(header_lower) if h.strip() == "b")
        c_idx = next(i for i, h in enumerate(header_lower) if h.strip() == "c")
    except StopIteration:
        # Sem colunas de coeficientes, retorna original
        return rows

    new_cols = [
        "X* (kg N ha⁻¹)",
        "Y_max (kg ha⁻¹)",
        "X_90% (kg N ha⁻¹)",
        "Y_90% (kg ha⁻¹)",
    ]
    augmented = [header + new_cols]
    have_values = [False, False, False, False]

    for row in rows[1:]:
        row_copy = list(row)
        
        # Extrai coeficientes
        try:
            a = _parse_float(row[a_idx] if a_idx < len(row) else "")
            b = _parse_float(row[b_idx] if b_idx < len(row) else "")
            c = _parse_float(row[c_idx] if c_idx < len(row) else "")
            
            if a is not None and b is not None and c is not None:
                # c é sempre positivo no formato, mas equação é Y = a + bX - cX²
                coeffs = (a, b, -c)
            else:
                coeffs = None
        except (ValueError, IndexError):
            coeffs = None
        
        metrics = ["", "", "", ""]
        if coeffs:
            a, b, c = coeffs
            if c != 0:
                x_max = -b / (2 * c)
                y_max = a + b * x_max + c * (x_max ** 2)
                metrics[0] = f"{x_max:.1f}"
                metrics[1] = f"{y_max:.0f}"
                
                # Calcula X para 90% do máximo
                target = 0.9 * y_max
                A = c
                B = b
                C = a - target
                disc = B ** 2 - 4 * A * C
                if disc >= 0 and A != 0:
                    sqrt_disc = sqrt(disc)
                    roots = [(-B - sqrt_disc) / (2 * A), (-B + sqrt_disc) / (2 * A)]
                    roots = [r for r in roots if r >= 0]
                    if roots:
                        x90 = min(roots)
                        y90 = a + b * x90 + c * x90 ** 2
                        metrics[2] = f"{x90:.1f}"
                        metrics[3] = f"{y90:.0f}"
            
            for idx, val in enumerate(metrics):
                if val:
                    have_values[idx] = True
        
        row_copy.extend(metrics)
        augmented.append(row_copy)

    if not any(have_values):
        # Nenhum valor calculado, retorna original
        return rows
    
    logger.info("Colunas calculadas adicionadas: %s", new_cols)
    return augmented


def _parse_float(value: str) -> Optional[float]:
    """Converte string para float, aceitando vírgula ou ponto"""
    if not value or not isinstance(value, str):
        return None
    try:
        cleaned = value.strip().replace(",", ".")
        return float(cleaned)
    except ValueError:
        return None


def _save_html_table(
    html_content: str,
    out_dir: Path,
    base_name: str,
    title: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[str]:
    """Salva tabela HTML com estrutura complexa preservada"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from html import escape
    
    # Monta HTML completo
    notes_clean = notes.strip() if isinstance(notes, str) and notes.strip() else None
    title_clean = title.strip() if isinstance(title, str) and title.strip() else None
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #fff; color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; color: #333; }}
        th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .notes {{ margin-top: 20px; padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107; color: #856404; }}
        .title {{ font-size: 1.5em; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }}
    </style>
</head>
<body>
"""
    
    if title_clean:
        full_html += f'    <div class="title">{escape(title_clean)}</div>\n'
    
    full_html += f'    {html_content}\n'
    
    if notes_clean:
        full_html += f'    <div class="notes"><strong>Notas:</strong> {escape(notes_clean)}</div>\n'
    
    full_html += """</body>
</html>"""
    
    # Salva HTML
    html_path = out_dir / f"{base_name}.html"
    html_path.write_text(full_html, encoding="utf-8")
    logger.info("✅ HTML salvo: %s", html_path.name)
    
    # Tenta converter HTML para Excel (parsing básico)
    try:
        import pandas as pd
        from io import StringIO
        
        # Pandas pode ler HTML table direto
        dfs = pd.read_html(StringIO(html_content))
        if dfs:
            df = dfs[0]  # Primeira tabela encontrada
            excel_path = out_dir / f"{base_name}.xlsx"
            df.to_excel(excel_path, index=False)
            logger.info("✅ Excel convertido: %s", excel_path.name)
    except Exception as e:
        logger.warning("⚠️  Não foi possível converter HTML para Excel: %s", e)
    
    # Retorna HTML inline para sumário
    return html_content


def _save_table_outputs(
    rows: List[List[str]],
    out_dir: Path,
    base_name: str,
    notes: Optional[str] = None,
) -> Optional[str]:
    """Salva tabela em múltiplos formatos (Excel, HTML, JSON) - LEGADO para JSON simples"""
    if not rows:
        logger.warning("Sem dados para salvar em %s/%s", out_dir, base_name)
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Identifica header e body
    header: Optional[List[str]] = None
    body = rows
    if rows and isinstance(rows[0], list):
        candidate = rows[0]
        if all(isinstance(v, str) for v in candidate) and len(rows) > 1:
            same_len = all(len(r) == len(candidate) for r in rows[1:])
            if same_len:
                header = candidate
                body = rows[1:]
    
    # Salva Excel
    try:
        import pandas as pd
        if header:
            df = pd.DataFrame(body, columns=header)
        else:
            df = pd.DataFrame(rows)
        df.to_excel(out_dir / f"{base_name}.xlsx", index=False)
    except Exception as e:
        logger.warning("Erro ao salvar Excel: %s", e)
    
    # Salva notas se houver
    notes_clean = notes.strip() if isinstance(notes, str) and notes.strip() else None
    if notes_clean:
        (out_dir / f"{base_name}-notes.txt").write_text(notes_clean + "\n", encoding="utf-8")
    
    # Salva HTML
    try:
        import pandas as pd
        from html import escape
        if header:
            df = pd.DataFrame(body, columns=header)
        else:
            df = pd.DataFrame(rows)
        html = df.to_html(index=False, header=bool(header))
        if notes_clean:
            html += f'\n<p><strong>Notas:</strong> {escape(notes_clean)}</p>'
        (out_dir / f"{base_name}.html").write_text(html, encoding="utf-8")
        return html
    except Exception as e:
        logger.warning("Erro ao salvar HTML: %s", e)
        return None


def _write_summary_html(base_dir: Path, entries: List[Dict[str, str]]) -> None:
    """Escreve summary.html com merge de execuções anteriores"""
    summary_path = base_dir / "summary.html"
    
    # Carrega entradas existentes
    existing_entries: Dict[tuple[str, str], Dict[str, str]] = {}
    if summary_path.exists():
        try:
            import re
            content = summary_path.read_text(encoding="utf-8")
            pattern = r"<section class='table-block'><h3>Página\s+(\S+)\s+-\s+(\S+)</h3>(.*?)</section>"
            matches = re.findall(pattern, content, re.DOTALL)
            for page, table, html_content in matches:
                existing_entries[(page, table)] = {
                    "page": page,
                    "table": table,
                    "html": html_content
                }
            logger.info("Carregadas %s entradas existentes do summary.html", len(existing_entries))
        except Exception as e:
            logger.warning("Erro ao ler summary.html existente: %s", e)
    
    # Merge: novas entradas sobrescrevem existentes
    all_entries: Dict[tuple[str, str], Dict[str, str]] = existing_entries.copy()
    for entry in entries:
        key = (entry["page"], entry["table"])
        all_entries[key] = entry
    
    logger.info("Total no summary.html: %s (%s novas)", len(all_entries), len(entries))
    
    # Gera HTML
    rows = []
    for entry in sorted(all_entries.values(), key=lambda e: (e["page"], e["table"])):
        rows.append(
            f"<section class='table-block'><h3>Página {entry['page']} - {entry['table']}</h3>{entry['html']}</section>"
        )
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    template = (
        "<html><head><meta charset='utf-8'>"
        "<style>body{{font-family:Arial,sans-serif;padding:20px;background:#f9f9f9;color:#333;}}"
        "section.table-block{{background:#fff;border:1px solid #ddd;margin-bottom:20px;padding:15px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}}"
        "section.table-block h3{{margin-top:0;font-size:16px;color:#333;}}"
        "table{{border-collapse:collapse;width:100%;margin-top:10px;}}table,th,td{{border:1px solid #ccc;}}"
        "th,td{{padding:6px;font-size:13px;text-align:left;color:#333;}}thead tr{{background:#eee;}}thead th{{color:#333;font-weight:bold;}}"
        "h1{{color:#2c3e50;margin-top:0;}}"
        "</style></head><body>"
        "<h1>Resumo das tabelas/gráficos gerados via LLM</h1>"
        "<p style='color:#666;font-size:14px;'>Total: {total} | Última atualização: {timestamp}</p>"
        "{content}</body></html>"
    )
    
    html = template.format(
        total=len(all_entries),
        timestamp=timestamp,
        content="\n".join(rows)
    )
    
    summary_path.write_text(html, encoding="utf-8")
    logger.info("Summary.html atualizado: %s entradas", len(all_entries))


def _extract_tables_from_payload(payload: dict) -> List[Dict[str, Any]]:
    """Extrai tabelas do payload JSON retornado pelo GPT-5 (suporta HTML e JSON)"""
    tables: List[Dict[str, Any]] = []
    t = payload.get("type")
    
    if t == "table":
        # Formato HTML
        if payload.get("format") == "html":
            html = payload.get("html")
            if html:
                tables.append({
                    "format": "html",
                    "html": html,
                    "title": payload.get("title"),
                    "notes": payload.get("notes"),
                })
        else:
            # Formato JSON legado
            table = payload.get("table") or {}
            rows = table.get("rows") or []
            if rows:
                tables.append({
                    "title": payload.get("title"),
                    "headers": table.get("headers"),
                    "rows": rows,
                    "notes": payload.get("notes"),
                })
    elif t == "table_set":
        for entry in payload.get("tables") or []:
            # Entrada de gráfico (conteúdo misto)
            if (entry or {}).get("type") == "chart":
                chart = (entry or {}).get("chart")
                if chart:
                    tables.append({
                        "type": "chart",
                        "chart": chart,
                        "title": entry.get("title"),
                        "notes": entry.get("notes"),
                    })
            # Formato HTML
            elif (entry or {}).get("format") == "html":
                html = (entry or {}).get("html")
                if html:
                    tables.append({
                        "format": "html",
                        "html": html,
                        "title": entry.get("title"),
                        "notes": entry.get("notes"),
                    })
            else:
                # Formato JSON legado
                table = (entry or {}).get("table") or {}
                rows = table.get("rows") or []
                if not rows:
                    continue
                tables.append({
                    "title": entry.get("title"),
                    "headers": table.get("headers"),
                    "rows": rows,
                    "notes": entry.get("notes"),
                })
    
    return tables


def _normalize_table_rows(headers: Optional[List[Any]], rows: List[List[Any]]) -> List[List[str]]:
    """Normaliza linhas de tabela para strings"""
    normalized: List[List[str]] = []
    if headers:
        normalized.append(["" if h is None else str(h) for h in headers])
    for row in rows:
        normalized.append(["" if cell is None else str(cell) for cell in row])
    # Remove linhas vazias
    normalized = [r for r in normalized if any(str(cell).strip() for cell in r)]
    return normalized


def _json_dumps(payload: dict) -> str:
    """Converte dict para JSON formatado"""
    return json.dumps(payload, ensure_ascii=False, indent=2)

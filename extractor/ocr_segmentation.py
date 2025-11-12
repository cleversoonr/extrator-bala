"""
Módulo de segmentação de páginas usando PaddleOCR.

Este módulo isola toda a complexidade de OCR/layout detection,
permitindo processar páginas com múltiplos elementos separadamente.

Uso:
- Automaticamente ativado quando pre-check detecta 2+ elementos
- Processa cada tabela/gráfico individualmente (melhor precisão)
- Fallback para página inteira se segmentação falhar
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np
import shutil

try:
    from paddleocr import PPStructure  # type: ignore
except ImportError:  # pragma: no cover - depende de lib opcional
    PPStructure = None  # type: ignore

from .logging_utils import get_logger
from .llm_vision import call_openai_vision_json

logger = get_logger(__name__)

_layout_engine_warning_emitted = False
_SUPPORTED_LAYOUT_LANGS = {"en", "ch"}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class SegmentedElement:
    """Representa um elemento segmentado (tabela/gráfico) da página"""
    element_type: str  # "table" | "chart"
    image_path: Path
    bbox: Tuple[int, int, int, int]
    index: int
    score: Optional[float] = None


# =============================================================================
# PROMPTS (específicos para segmentos)
# =============================================================================

SEGMENT_TABLE_PROMPT = """⚠️ PRINCÍPIO: Extraia EXATAMENTE como está. Use a LEGENDA como referência.

Esta imagem é um recorte contendo EXATAMENTE 1 tabela.

**INSTRUÇÕES GERAIS:**
- Use HTML completo `<table>` preservando colspan/rowspan
- Título deve ser o TEXTO visível na própria tabela (se houver). Se não houver, infira um título breve com base no conteúdo.
- **CAPTURE A LEGENDA/NOTA** que está VISÍVEL neste recorte (ex: "C = verde", "Fonte: JACOB 1989", etc.) e coloque em `"notes"`

**⚠️ TRANSCRIÇÃO EXATA - REGRA ABSOLUTA:**

**VOCÊ DEVE LER E TRANSCREVER O TEXTO ESCRITO EM CADA CÉLULA.**

**ATENÇÃO: LETRAS PEQUENAS!**
- Células podem ter letras MUITO PEQUENAS (C, CL, I, etc.)
- Amplie mentalmente, foque, leia com cuidado
- TODAS as células coloridas geralmente têm texto - não deixe vazias sem verificar!

**PROCEDIMENTO OBRIGATÓRIO:**

1. **CONTE** quantas linhas e colunas a tabela tem

2. **LEIA CADA CÉLULA COM ATENÇÃO**:
   - Se vê letra "C" (mesmo pequena) → `<td>C</td>`
   - Se vê letras "CL" → `<td>CL</td>`
   - Se vê letra "I" → `<td>I</td>`
   - Se vê "—" (travessão/hífen) → `<td>—</td>`
   - Se vê número/fórmula/texto → transcreva EXATAMENTE
   - Se célula está REALMENTE vazia → `<td></td>`

3. **TRABALHE DEVAGAR**:
   - Uma célula por vez
   - NÃO copie linhas inteiras
   - NÃO assuma padrões
   - Cada célula é independente
   - Zoom mental nas células pequenas

**PROIBIDO:**
- Deixar TODAS células vazias (improvável que tabela inteira seja vazia)
- Copiar linhas
- Assumir que cor = ausência de texto

**Formato obrigatório:**
{
  "type": "table",
  "format": "html",
  "title": "Título exato ou inferido",
  "html": "<table>...</table>",
  "notes": "Notas específicas (opcional)"
}

Retorne somente JSON válido."""


CHART_PROMPT = """⚠️ PRINCÍPIO: Extraia EXATAMENTE os dados visíveis. NÃO invente, NÃO force padrões.

Extraia dados deste GRÁFICO como JSON.

**CASOS:**

1. **Diagrama ternário (triângulo de textura):**
   - Identifique cada classe/região do triângulo (ex.: Arenosa, Média, Argilosa, Muito argilosa, Siltosa).
   - Leia os valores dos vértices e bordas e retorne faixas completas de areia, argila e silte para cada classe.
   - Formato obrigatório:
     {
       "type": "table",
       "table": {
         "headers": ["Classe de Textura", "Areia (%)", "Argila (%)", "Silte (%)"],
         "rows": [
           ["Arenosa", "70-100", "0-15", "0-30"],
           ...
         ]
       }
     }
   - Nunca deixe valores como null; se o gráfico mostra uma faixa, escreva "min-max". Se o valor é fixo, escreva "valor".
   - NÃO responda com descrições textuais; converta o triângulo inteiro em tabela.

2. **Gráfico com equações (Y = a + bX ± cX²):**
   - **EXTRAIA TODAS AS EQUAÇÕES VISÍVEIS**
   - Formato: {"type": "table", "table": {"headers": ["Painel", "a", "b", "c", "R²"], "rows": [...]}}
   - Coluna "c": SEMPRE POSITIVA (ignore sinal da equação)
   - Se múltiplos painéis, uma linha por equação

3. **Gráfico de dados (linhas/barras/dispersão):**
   ⚠️ **CRÍTICO - EXTRAÇÃO DE VALORES:**
   - Você DEVE ler os valores numéricos de CADA PONTO visível no gráfico
   - Procedimento obrigatório:
     1. Conte QUANTAS marcas/valores existem no EIXO X (ex: 12 datas)
     2. Identifique TODOS os valores do EIXO X (datas, categorias ou números)
     3. Identifique a escala do EIXO Y (min, max) lendo os números à esquerda
     4. Para CADA série/linha do gráfico:
        a) Identifique a série pela cor/símbolo (●, ○, etc) ou legenda
        b) Localize CADA ponto dessa série no gráfico
        c) Leia o valor Y aproximado para cada ponto (COM DECIMAIS!)
        d) Se um ponto não existe para aquele X, use null
   
   ⚠️ **REGRA CRÍTICA - QUANTIDADE DE VALORES:**
   - Se o eixo X tem N valores → CADA série DEVE ter EXATAMENTE N valores
   - Exemplo: 12 datas no eixo X → cada série TEM que ter 12 valores
   - NUNCA retorne arrays com mais ou menos valores que o eixo X!
   
   ⚠️ **REGRA CRÍTICA - VALORES DECIMAIS:**
   - Gráficos podem mostrar vírgula (0,4) ou ponto (0.4) como decimal
   - No JSON, sempre use PONTO: 0.4 (padrão JSON)
   - CUIDADO: "0,4" é UM número (0.4), não dois ([0, 4])
   - Conte seus valores: se eixo X tem 12 pontos, cada série deve ter 12 números
   
   - Formato obrigatório:
     {
       "type": "chart",
       "chart": {
         "x": {
           "type": "category",  // ou "numeric"
           "values": [...]  // valores do eixo X (datas, categorias, números)
         },
         "y": {
           "label": "...",  // rótulo do eixo Y
           "min": 0,
           "max": 10
         },
         "series": [
           {
             "name": "...",  // nome da série (leia da legenda)
             "values": [1.5, 2.3, null, 4.7, ...]  // valores Y (null se ponto não existe)
           }
         ]
       }
     }
   - Leia TODOS os valores visíveis - não deixe arrays de null
   - Se múltiplos painéis (a, b), inclua no nome da série

4. **OUTROS**
   - Se encontrar outro tipo de gráfico, compreenda e extraia os dados.
   - Formato: {"type": "chart", "chart": {"x": {...}, "y": {...}, "series": [{"name": "...", "values": [...]}]}}

**Regras Finais:**
- Use PONTO decimal no JSON (padrão: 3.4)
- Extraia EXATAMENTE o que está visível - não invente dados
- Cada série deve ter EXATAMENTE quantos valores o eixo X tem
- Use null APENAS se um ponto específico não existe (não preencha série inteira de null)
- **CAPTURE LEGENDA/NOTA** visível no gráfico (ex: "● = Sem P", "Fonte: Silva 2020") e coloque em `"notes"`

Retorne APENAS JSON válido."""


# =============================================================================
# FUNÇÕES DE INICIALIZAÇÃO E CACHE
# =============================================================================

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
    """Normaliza idioma para PaddleOCR (apenas 'en' e 'ch' suportados)"""
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
    """Limpa cache de instâncias do PPStructure"""
    try:
        _get_layout_engine.cache_clear()  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _cleanup_paddle_structure_cache(lang: str) -> None:
    """Remove cache corrompido do PaddleOCR"""
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


# =============================================================================
# FUNÇÕES DE PROCESSAMENTO DE IMAGEM
# =============================================================================

def _apply_padding_to_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> Tuple[int, int, int, int]:
    """Adiciona padding ao bbox sem ultrapassar limites da imagem"""
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
    """Recorta região da imagem"""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _enhance_segment_image(image: "cv2.Mat") -> "cv2.Mat":  # type: ignore[name-defined]
    """
    Aplica processamento balanceado para legibilidade sem explodir o tamanho:
    1. Denoise leve (h=6)
    2. Contraste adaptativo moderado (CLAHE clipLimit=2.5)
    3. Sharpening moderado (50% blend)
    4. Upscaling/Downscaling para ~1600px (lado maior)
    """
    
    # ETAPA 1: Denoise LEVE (remove ruído sem aumentar muito o processamento)
    try:
        image = cv2.fastNlMeansDenoisingColored(image, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=15)
    except Exception as e:
        logger.debug("Denoise falhou (não crítico): %s", e)
    
    # ETAPA 2: Contraste adaptativo MODERADO
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # Reduzido para evitar oversaturation
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        enhanced = image.copy()

    # ETAPA 3: Sharpening MODERADO
    try:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        # Blend: 50% sharpened + 50% enhanced (balanceado)
        sharpened = cv2.addWeighted(sharpened, 0.5, enhanced, 0.5, 0)
    except Exception:
        sharpened = enhanced.copy()

    # ETAPA 4: Upscaling MODERADO (balanceado entre qualidade e tamanho)
    try:
        h, w = sharpened.shape[:2]
        target_max = 1600  # Reduzido para evitar arquivos gigantes
        max_side = max(h, w)
        
        if max_side < target_max:
            scale = target_max / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info("📐 Upscaling segmento: %dx%d → %dx%d (%.1fx)", w, h, new_w, new_h, scale)
            sharpened = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif max_side > 3000:
            # Se imagem já muito grande, reduz para evitar processamento lento
            scale = 3000 / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info("📐 Downscaling segmento muito grande: %dx%d → %dx%d", w, h, new_w, new_h)
            sharpened = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Garante mínimo 800px no lado menor (suficiente para ler texto)
        h, w = sharpened.shape[:2]
        min_side = min(h, w)
        if min_side < 800:
            scale = 800 / min_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info("📐 Upscaling (lado menor): %dx%d → %dx%d", w, h, new_w, new_h)
            sharpened = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        logger.debug("Resize falhou (não crítico): %s", e)

    return sharpened


def _segment_reading_order_key(data: Dict[str, Any]) -> tuple:
    """Determina ordem de leitura dos segmentos (top-down, left-right)"""
    bbox = data.get("bbox", (0, 0, 0, 0))
    x1, y1, *_ = bbox
    row_bucket = int(y1 // 50)  # agrupa linhas aproximadas
    return (row_bucket, x1)


# =============================================================================
# FUNÇÕES DE MAPEAMENTO
# =============================================================================

def _layout_types_for_content(content_type: str) -> set[str]:
    """Mapeia tipo de conteúdo para tipos de layout aceitos"""
    mapping = {
        "table": {"table"},
        "chart": {"chart"},
        "mixed": {"table", "chart"},
    }
    return mapping.get(content_type, {"table", "chart"})


def _map_layout_type(layout_type: str) -> Optional[str]:
    """Mapeia tipo de layout do PaddleOCR para tipo interno"""
    if layout_type == "table":
        return "table"
    if layout_type in {"figure", "chart", "graphic"}:
        return "chart"
    return None


# =============================================================================
# SEGMENTAÇÃO PRINCIPAL
# =============================================================================

def segment_page_elements(
    page_image_path: Path,
    page_out: Path,
    ocr_lang: str,
    segment_padding: int,
    max_segments: Optional[int],
    content_type: str,
    expected_count: int,
) -> List[SegmentedElement]:
    """
    Segmenta a página em elementos individuais (tabelas/gráficos) usando PPStructure.
    
    Args:
        page_image_path: Caminho para imagem da página
        page_out: Diretório de saída
        ocr_lang: Idioma para OCR ('en' ou 'ch')
        segment_padding: Padding ao redor de cada segmento (px)
        max_segments: Limite máximo de segmentos (None = ilimitado)
        content_type: Tipo esperado ('table', 'chart', 'mixed')
        expected_count: Número esperado de elementos
    
    Returns:
        Lista de SegmentedElement com recortes salvos em disco
    """
    if not _layout_engine_available():
        return []

    bgr = cv2.imread(page_image_path.as_posix())
    if bgr is None:
        logger.warning("PPStructure: falha ao carregar imagem %s", page_image_path)
        return []

    normalized_lang = _normalize_ocr_lang(ocr_lang)

    try:
        engine = _get_layout_engine(ocr_lang)
    except Exception as exc:  # pragma: no cover - inicialização falhou
        logger.error("Falha ao inicializar PPStructure (lang=%s): %s", normalized_lang, exc)
        if "unexpected end of data" in str(exc).lower():
            logger.warning("Cache de modelo corrompido detectado na inicialização. Limpando...")
            _cleanup_paddle_structure_cache(ocr_lang)
            _reset_layout_engine_cache()
            try:
                engine = _get_layout_engine(ocr_lang)
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
                _cleanup_paddle_structure_cache(ocr_lang)
                _reset_layout_engine_cache()
                engine = _get_layout_engine(ocr_lang)
                continue
            return []

    if layout_results is None:
        return []

    raw_segments: List[Dict[str, Any]] = []
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
            segment_padding,
        )
        crop = _crop_image(bgr, padded_bbox)
        if crop is None:
            continue
        crop = _enhance_segment_image(crop)

        raw_segments.append(
            {
                "image": crop,
                "bbox": padded_bbox,
                "type": mapped_type,
                "score": item.get("score"),
            }
        )

        if max_segments and len(raw_segments) >= max_segments:
            break

    if not raw_segments:
        logger.info("PPStructure não encontrou segmentos relevantes (%s)", content_type)
        return []

    raw_segments.sort(key=_segment_reading_order_key)

    segments: List[SegmentedElement] = []
    for idx, data in enumerate(raw_segments, start=1):
        seg_path = page_out / f"segment-{idx:02d}.png"
        cv2.imwrite(seg_path.as_posix(), data["image"])
        segments.append(
            SegmentedElement(
                element_type=data["type"],
                image_path=seg_path,
                bbox=tuple(data["bbox"]),
                index=idx,
                score=data["score"],
            )
        )

    if expected_count and len(segments) != expected_count:
        logger.info(
            "PPStructure detectou %d elemento(s) vs pre-check %d",
            len(segments),
            expected_count,
        )

    logger.info("PPStructure segmentou %d elemento(s)", len(segments))
    return segments


# =============================================================================
# PROCESSAMENTO DE PAYLOADS
# =============================================================================

def get_prompt_for_segment(segment: SegmentedElement, total: int) -> str:
    """Retorna prompt apropriado para o tipo de segmento"""
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


def segment_payload_to_entries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converte payload do LLM em lista de entradas normalizadas"""
    from typing import Iterable
    
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


# =============================================================================
# FLUXO COMPLETO DE SEGMENTAÇÃO
# =============================================================================

def run_segmented_flow(
    segments: List[SegmentedElement],
    page_out: Path,
    page_id: str,
    llm_model: str,
    llm_provider: Optional[str],
    llm_api_key: Optional[str],
    azure_endpoint: Optional[str],
    azure_api_version: Optional[str],
    openrouter_api_key: Optional[str],
    locale: str,
) -> Optional[Dict[str, Any]]:
    """
    Executa extração LLM em cada segmento individualmente.
    
    Returns:
        Payload combinado com todos os elementos extraídos
    """
    import json
    
    total = len(segments)
    combined_entries: List[Dict[str, Any]] = []

    for segment in segments:
        instructions = get_prompt_for_segment(segment, total)
        logger.info(
            "🤖 Extraindo elemento %02d/%02d (%s) via LLM",
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
            model=llm_model,
            provider=llm_provider,
            api_key=llm_api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            openrouter_api_key=openrouter_api_key,
            locale=locale,
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
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        entries = segment_payload_to_entries(payload)
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


def write_segments_manifest(page_out: Path, segments: List[SegmentedElement]) -> None:
    """Salva manifest detalhando recortes produzidos pelo PaddleOCR."""
    import json
    
    if not segments:
        return
    manifest = {
        "count": len(segments),
        "segments": [
            {
                "index": seg.index,
                "type": seg.element_type,
                "bbox": list(seg.bbox),
                "image": seg.image_path.name,
                "score": seg.score,
            }
            for seg in segments
        ],
    }
    (page_out / "segments-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


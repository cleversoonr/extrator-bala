from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import cv2
import re
from html import escape

from .logging_utils import get_logger
from .llm_vision import call_openai_vision_json, to_table_from_llm_payload, quick_precheck_with_cheap_llm
from .pdf_utils import open_document, parse_pages, render_pages
from .ocr_segmentation import (
    SegmentedElement,
    segment_page_elements,
    run_segmented_flow,
    write_segments_manifest,
)


logger = get_logger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

PAGE_TABLE_PROMPT = """Extraia EXATAMENTE como está na imagem. Esta página contém {count_desc}.

**Formato JSON:**
{{
  "type": "table_set",
  "tables": [
    {{
      "title": "Título exato",
      "format": "html",
      "html": "<table>...</table>",
      "notes": "Legendas/observações"
    }}
  ]
}}

**HTML `<table>` preservando estrutura:**
- Estrutura: `<table><thead>...</thead><tbody>...</tbody></table>`
- Células mescladas: `colspan="N"` (horizontal) ou `rowspan="N"` (vertical)
- Cabeçalhos: `<thead>` com múltiplas linhas se necessário
- Formatação: `<sup>`, `<sub>`, `<strong>`
- Sub-cabeçalhos: Use `<tr>` adicional dentro do `<thead>`

**REGRAS CRÍTICAS:**
1. Leia CADA célula individualmente 
2. Transcreva EXATAMENTE o texto escrito (C, CL, I, números, etc.)
3. Preserve colspan/rowspan onde células ocupam múltiplas colunas/linhas
4. NÃO copie linhas (cada célula é única)
5. Se célula vazia → `<td></td>`
6. **NÃO invente linhas vazias com `<td colspan="14"></td>`** - isso quebra a estrutura
7. Se são tabelas SEPARADAS visualmente → crie objetos SEPARADOS no JSON

Retorne APENAS JSON válido."""


CHART_PROMPT = """Extraia dados deste GRÁFICO como JSON.

**Formato por tipo:**

**1. Diagrama ternário:**
{"type": "table", "table": {"headers": ["Classe", "Areia %", "Argila %", "Silte %"], "rows": [["Arenosa", "70-100", "0-15", "0-30"], ...]}}

**2. Equações (Y = a + bX ± cX²):**
{"type": "table", "table": {"headers": ["Painel", "a", "b", "c", "R²"], "rows": [...]}}
→ Coluna "c": sempre positiva

**3. Linhas/Barras/Dispersão:**
{"type": "chart", "chart": {"x": {"type": "category|numeric", "values": [...]}, "y": {"label": "...", "min": X, "max": Y}, "series": [{"name": "...", "values": [1.5, 2.3, null, ...]}]}}

**REGRAS CRÍTICAS:**
- Leia TODOS os pontos visíveis (amplie zoom mental)
- Cada série TEM que ter EXATAMENTE quantos valores o eixo X tem
- Decimais: use PONTO (0.4 não [0,4])
- Use null SÓ se ponto específico não existe
- Capture legenda/notas em "notes"

Retorne APENAS JSON válido."""


NOTES_PROMPT = """Você verá uma página inteira contendo tabelas e gráficos.

IDENTIFIQUE todas as notas de rodapé, legendas e fontes associadas a esses elementos.

**Saída obrigatória:**
{
  "notes": [
    {"label": "Legenda", "text": "Legenda: ...", "applies_to": "Tabela 3"},
    {"label": "Fonte", "text": "Fonte: ...", "applies_to": "Tabela 3"},
    {"label": "Nota geral", "text": "Nota: ..."}
  ]
}

Regras:
- Extraia o texto EXATO que aparece na página (não traduza).
- Se a nota mencionar uma tabela/figura específica, informe em "applies_to".
- Caso seja uma legenda genérica (ex.: explicação de cores), use "Legenda" em label.
- Se não encontrar notas/legendas, retorne {"notes": []}.

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
    skip_ocr_pages: List[int] = None  # Páginas específicas onde OCR deve ser desabilitado
    ocr_lang: str = "en"
    segment_padding: int = 32
    max_segments: Optional[int] = None
    fallback_to_full_page: bool = True
    force_reprocess: bool = False  # Se True, reprocessa mesmo páginas já extraídas
    convert_text_only: bool = False  # Se True, converte páginas text-only em HTML
    
    def __post_init__(self):
        if self.skip_ocr_pages is None:
            self.skip_ocr_pages = []


# =============================================================================
# CHECKPOINT (Validação de páginas já processadas)
# =============================================================================


def _is_page_already_processed(page_out: Path) -> bool:
    """
    Verifica se uma página já foi processada com sucesso.
    
    Critérios:
    - A pasta da página existe
    - Existe pelo menos um arquivo HTML válido (table-*.html ou chart-*.html)
    
    Returns:
        True se a página já foi processada, False caso contrário
    """
    if not page_out.exists():
        return False
    
    # Procura por arquivos HTML gerados
    html_files = list(page_out.glob("table-*.html")) + list(page_out.glob("chart-*.html"))
    
    # Se não encontrou nenhum HTML, página não foi processada
    if not html_files:
        return False
    
    # Verifica se pelo menos um HTML tem conteúdo válido (não vazio)
    for html_file in html_files:
        try:
            content = html_file.read_text(encoding="utf-8")
            # HTML válido deve ter pelo menos uma tag <table> ou conteúdo significativo
            if content.strip() and len(content) > 50:
                return True
        except Exception:
            continue
    
    return False


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

    # Renderiza páginas completas (checkpoint: pula páginas já renderizadas)
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

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECKPOINT: Verifica se página já foi processada
    # ═══════════════════════════════════════════════════════════════════════════
    if not config.force_reprocess and _is_page_already_processed(page_out):
        logger.info("✅ Página %s JÁ PROCESSADA (checkpoint) - pulando", page.page_number)
        
        # Retorna os arquivos HTML/Excel existentes
        html_files = list(page_out.glob("table-*.html")) + list(page_out.glob("chart-*.html"))
        excel_files = list(page_out.glob("*.xlsx"))
        
        for excel_file in excel_files:
            page_outputs.append(excel_file)
        
        # Cria summary a partir dos HTMLs existentes
        for html_file in html_files:
            try:
                html_content = html_file.read_text(encoding="utf-8")
                base_name = html_file.stem  # Remove .html
                page_summary.append({
                    "page": page_id,
                    "table": base_name,
                    "html": html_content
                })
            except Exception as e:
                logger.debug("Erro ao ler HTML %s: %s", html_file.name, e)
        
        return page_outputs, page_summary

    # Se chegou aqui, precisa processar a página
    if config.force_reprocess:
        logger.info("🔄 Página %s será REPROCESSADA (force_reprocess=True)", page.page_number)

    # ═══════════════════════════════════════════════════════════════════════════
    # ETAPA 1: Pre-check DIRETO na imagem renderizada (SEM copiar ainda!)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info(
        "🚦 Página %s → pre-check ativo? %s (modelo=%s)",
        page_id,
        "sim" if config.use_cheap_precheck and config.cheap_model else "não",
        config.cheap_model or "n/a",
    )
    
    # 🎯 PRE-CHECK USA IMAGEM ORIGINAL (pages/page-XXX.png) - NÃO COPIA AINDA!
    has_content, content_type, content_count, rotation, characteristics = _page_level_precheck(page.path, config)
    logger.info(
        "📋 Resultado do pre-check (página %s): has_content=%s | type=%s | count=%s | rotation=%s°",
        page_id,
        has_content,
        content_type,
        content_count,
        rotation,
    )
    
    # ✅ VERIFICAÇÃO CRÍTICA: Se não tem conteúdo útil, PULA ou CONVERTE TEXTO
    if not has_content or content_type == "text_only" or content_count == 0:
        # 📝 CASO ESPECIAL: Se text_only E convert_text_only=True, converte texto
        if content_type == "text_only" and config.convert_text_only:
            logger.info(
                "📄 Página %s: text-only com CONVERT_TEXT_ONLY ativado → extraindo texto completo",
                page.page_number,
            )
            
            # Importa módulo de extração de texto
            from .text_extraction import extract_text_from_page
            
            # Extrai texto completo usando GPT-5
            html_path, summary_entry = extract_text_from_page(
                page.path,  # Usa imagem original (não copia!)
                page_out,
                page_id,
                config.model,  # Usa GPT-5 (modelo principal)
                config.provider,
                config.api_key,
                config.azure_endpoint,
                config.azure_api_version,
                config.openrouter_api_key,
                config.locale,
            )
            
            if html_path and summary_entry:
                page_outputs.append(html_path)
                page_summary.append(summary_entry)
                logger.info("✅ Texto extraído com sucesso da página %s", page.page_number)
            else:
                logger.warning("⚠️ Falha ao extrair texto da página %s", page.page_number)
            
            # Salva snapshot
            precheck_snapshot = {
                "model": config.cheap_model if config.use_cheap_precheck else None,
                "provider": config.cheap_provider if config.use_cheap_precheck else None,
                "has_content": has_content,
                "content_type": content_type,
                "content_count": content_count,
                "rotation_detected": rotation,
                "characteristics": characteristics,
                "ocr_decision": "skip",
                "ocr_reason": "text_only - texto extraído",
                "text_extracted": html_path is not None,
            }
            (page_out / "precheck.json").write_text(_json_dumps(precheck_snapshot), encoding="utf-8")
            return page_outputs, page_summary
        
        # Se NÃO é text_only OU convert_text_only=False, apenas pula
        logger.info(
            "⏭️  Página %s: sem conteúdo útil (has_content=%s, type=%s, count=%d) → PULANDO (sem copiar!)",
            page.page_number,
            has_content,
            content_type,
            content_count,
        )
        # Salva snapshot de página pulada
        precheck_snapshot = {
            "model": config.cheap_model if config.use_cheap_precheck else None,
            "provider": config.cheap_provider if config.use_cheap_precheck else None,
            "has_content": has_content,
            "content_type": content_type,
            "content_count": content_count,
            "rotation_detected": rotation,
            "characteristics": characteristics,
            "ocr_decision": "skip",
            "ocr_reason": "Página sem conteúdo útil (text_only ou count=0)",
        }
        (page_out / "precheck.json").write_text(_json_dumps(precheck_snapshot), encoding="utf-8")
        return page_outputs, page_summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ETAPA 2: Página TEM CONTEÚDO → SÓ AGORA copia para processamento
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("✅ Página %s TEM CONTEÚDO → copiando para processamento", page.page_number)
    full_page_path = page_out / "page-full.png"
    bgr = cv2.imread(page.path.as_posix())
    if bgr is None:
        logger.warning("Falha ao carregar imagem da página %s", page.page_number)
        return page_outputs, page_summary
    
    cv2.imwrite(full_page_path.as_posix(), bgr)
    logger.debug("📄 Imagem copiada: %s → %s (%.1f MB)", 
                page.path.name, 
                full_page_path.name,
                full_page_path.stat().st_size / (1024 * 1024))
    
    # DECISÃO INTELIGENTE: Usar OCR ou não? (baseado na QUANTIDADE de elementos)
    use_ocr_decision, ocr_reason = _should_use_ocr(content_count, characteristics)
    logger.info("🧠 Decisão de OCR: %s → %s", "USAR" if use_ocr_decision else "NÃO USAR", ocr_reason)
    
    # Aplica decisão (pode sobrescrever config.use_layout_ocr)
    original_ocr_setting = config.use_layout_ocr
    config.use_layout_ocr = use_ocr_decision
    
    # CORREÇÃO DE ROTAÇÃO (se detectada)
    if rotation != 0 and rotation in (90, 180, 270):
        logger.warning("🔄 Rotação detectada: %s° → corrigindo imagem antes do OCR", rotation)
        full_page_path = _correct_image_rotation(full_page_path, rotation, page_out)
        logger.info("✅ Imagem corrigida salva: %s", full_page_path.name)
    
    precheck_snapshot = {
        "model": config.cheap_model if config.use_cheap_precheck else None,
        "provider": config.cheap_provider if config.use_cheap_precheck else None,
        "has_content": has_content,
        "content_type": content_type,
        "content_count": content_count,
        "rotation_detected": rotation,
        "characteristics": characteristics,
        "ocr_decision": "use" if use_ocr_decision else "skip",
        "ocr_reason": ocr_reason,
    }
    (page_out / "precheck.json").write_text(_json_dumps(precheck_snapshot), encoding="utf-8")

    # Extrai notas/legendas da PÁGINA INTEIRA (modelo barato)
    page_notes = _extract_page_notes(full_page_path, config)
    all_notes_text = ""
    if page_notes:
        (page_out / "page-notes.json").write_text(_json_dumps({"notes": page_notes}), encoding="utf-8")
        logger.info("📝 %d nota(s)/legenda(s) extraídas da página.", len(page_notes))
        
        # Concatena TODAS as notas em um único texto
        all_notes_text = "\n\n".join([note.get("text", "") for note in page_notes if note.get("text")])
        
        # Salva TODAS as notas para CADA tabela detectada (content_count)
        # Isso garante que cada tabela tenha acesso a todas as notas/legendas/fontes da página
        for i in range(1, content_count + 1):
            note_file = page_out / f"table-{i:02d}-notes.txt"
            note_file.write_text(all_notes_text, encoding="utf-8")
            logger.debug("💾 Notas completas salvas para: %s", note_file.name)

    logger.info(
        "Página %s: detectado %s (count=%d), processando...",
        page.page_number,
        content_type,
        content_count,
    )

    # ETAPA 2: Extração com GPT-5 (página inteira) + Prompt Personalizado
    outputs, summaries = _llm_page_to_tables(
        full_page_path,
        page_out,
        page_id,
        config,
        content_type,
        content_count,
        characteristics,  # ← Passa características para gerar prompt personalizado
    )
    
    # Restaura configuração original de OCR
    config.use_layout_ocr = original_ocr_setting
    
    page_outputs.extend(outputs)
    page_summary.extend(summaries)
    
    return page_outputs, page_summary


def _correct_image_rotation(image_path: Path, rotation: int, output_dir: Path) -> Path:
    """
    Corrige a rotação da imagem baseado na POSIÇÃO ATUAL do título.
    
    rotation: ÂNGULO ATUAL do título (onde está agora)
    - 0 = Título já está horizontal (não precisa girar)
    - 90 = Título está virado 90° (à direita) → gira 270° clockwise para corrigir
    - 180 = Título está de cabeça pra baixo → gira 180° para corrigir
    - 270 = Título está virado 270° (à esquerda) → gira 90° clockwise para corrigir
    
    Retorna: Path para a imagem corrigida
    """
    import cv2
    
    bgr = cv2.imread(image_path.as_posix())
    if bgr is None:
        logger.error("Falha ao carregar imagem para correção de rotação: %s", image_path)
        return image_path
    
    # Converte posição atual → rotação necessária
    if rotation == 90:
        # Título está à direita → gira 270° clockwise (ou -90°)
        corrected = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        logger.info("✓ Título está em 90° → Girando 270° clockwise (90° anti-horário)")
    elif rotation == 180:
        # Título está de cabeça pra baixo → gira 180°
        corrected = cv2.rotate(bgr, cv2.ROTATE_180)
        logger.info("✓ Título está em 180° → Girando 180°")
    elif rotation == 270:
        # Título está à esquerda → gira 90° clockwise
        corrected = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        logger.info("✓ Título está em 270° → Girando 90° clockwise")
    else:
        logger.warning("Rotação inválida: %s (esperado: 90, 180, 270)", rotation)
        return image_path
    
    # Salva imagem corrigida
    corrected_path = output_dir / f"page-full-corrected.png"
    cv2.imwrite(corrected_path.as_posix(), corrected)
    
    return corrected_path


def _should_use_ocr(content_count: int, characteristics: dict) -> tuple[bool, str]:
    """
    DECISÃO: Usar OCR para segmentar antes de enviar para LLM.
    
    MOTIVO:
    - Letras/números pequenos na página inteira são difíceis para LLM ler
    - OCR recorta e amplia a região de interesse
    - LLM recebe imagem maior e mais legível
    - Resultado: extração mais precisa
    
    Retorna: (use_ocr: bool, reason: str)
    
    NOTA: Esta função só é chamada APÓS validar que a página tem conteúdo útil
    (has_content=True, content_type != "text_only", content_count > 0)
    """
    if content_count <= 1:
        return True, f"1 elemento detectado → usar OCR para recortar e ampliar (letras pequenas ficam mais legíveis)"
    else:
        return True, f"{content_count} elementos detectados → usar OCR para segmentar e processar cada um individualmente"


def _generate_custom_prompt(characteristics: dict, base_prompt: str) -> str:
    """
    PROMPT 100% DINÂMICO: Gera instruções COMPLETAMENTE baseadas nas características detectadas.
    SEM SUPOSIÇÕES - apenas o que foi VISTO no pre-check.
    """
    if not characteristics:
        return base_prompt
    
    elements = characteristics.get("elements", [])
    if not elements:
        # Fallback: usa características antigas se não houver array de elementos
        return _generate_legacy_prompt(characteristics, base_prompt)
    
    # Para cada elemento, gera seção personalizada
    custom_sections = []
    
    for idx, element in enumerate(elements, 1):
        elem_type = element.get("type")
        description = element.get("description", "")
        structure = element.get("structure", {})
        
        if elem_type == "table":
            section = _generate_table_instructions(idx, description, structure)
            custom_sections.append(section)
        elif elem_type == "chart":
            section = _generate_chart_instructions(idx, description, structure)
            custom_sections.append(section)
    
    if custom_sections:
        header = "\n\n" + "="*70 + "\n"
        header += "INSTRUÇÕES PERSONALIZADAS (baseadas em análise visual do pre-check)\n"
        header += "="*70 + "\n\n"
        
        full_instructions = header + "\n\n".join(custom_sections) + "\n\n" + "="*70 + "\n\n"
        return full_instructions + base_prompt
    
    return base_prompt


def _generate_table_instructions(idx: int, description: str, structure: dict) -> str:
    """Gera instruções específicas para uma tabela baseado no que foi DETECTADO."""
    instructions = []
    
    instructions.append(f"📊 **TABELA {idx}**: {description}")
    instructions.append("")
    
    instructions.append("⚠️  **PROCEDIMENTO**: Leia célula por célula, linha por linha")
    instructions.append("   → Amplie zoom mental em células pequenas")
    instructions.append("   → Transcreva EXATAMENTE o texto escrito, não cores")
    instructions.append("   → NÃO copie linhas/colunas (cada célula é única)")
    instructions.append("")
    
    # Estrutura
    table_structure = structure.get("table_structure")
    rows = structure.get("rows", 0)
    cols = structure.get("columns", 0)
    
    if table_structure and rows and cols:
        instructions.append(f"📐 **ESTRUTURA DETECTADA**: {table_structure} ({rows} linhas × {cols} colunas)")
    
    # Diagonal vazia
    if structure.get("diagonal_empty"):
        instructions.append("⚠️  **DIAGONAL**: Células da diagonal principal estão VAZIAS na imagem")
        instructions.append("   → Deixe `<td></td>` vazio (não invente conteúdo)")
    
    # Cores (APENAS como contexto, não como base para decisão)
    if structure.get("has_colors"):
        color_meaning = structure.get("color_meaning", "")
        if color_meaning:
            instructions.append(f"ℹ️  **CONTEXTO** (cores na imagem): {color_meaning}")
            instructions.append("   → Use apenas como CONTEXTO para entender a tabela")
            instructions.append("   → Mas SEMPRE transcreva o TEXTO que vê escrito, NÃO a cor")
        else:
            instructions.append("ℹ️  **CONTEXTO**: Células têm cores de fundo (design)")
            instructions.append("   → Ignore cores completamente - foque no TEXTO escrito")
    
    # Conteúdo das células
    cell_content_type = structure.get("cell_content_type")
    cell_content_desc = structure.get("cell_content_description", "")
    
    if cell_content_desc:
        instructions.append(f"📝 **CONTEÚDO DAS CÉLULAS**: {cell_content_desc}")
        if cell_content_type == "symbols":
            instructions.append("   → Amplie zoom mental, letras podem ser MUITO pequenas")
            instructions.append("   → Trabalhe célula por célula com calma")
    
    # Legenda
    if structure.get("has_legend"):
        legend_content = structure.get("legend_content", "")
        if legend_content:
            instructions.append(f"📖 **LEGENDA**: {legend_content}")
            instructions.append("   → Use para entender contexto, mas transcreva o que está ESCRITO")
    
    # Células mescladas
    if structure.get("has_merged_cells"):
        merged_location = structure.get("merged_cells_location", 'unknown')
        instructions.append("")
        instructions.append(f"🔗 **CÉLULAS MESCLADAS** detectadas em: {merged_location}")
        instructions.append("   → Use colspan=\"N\" para células que ocupam N colunas")
        instructions.append("   → Use rowspan=\"N\" para células que ocupam N linhas")
        instructions.append("   → VALIDE: todas as linhas devem ter mesmo total de colunas (contando colspan)")
        instructions.append("")
    
    return "\n".join(instructions)


def _generate_chart_instructions(idx: int, description: str, structure: dict) -> str:
    """Gera instruções específicas para um gráfico baseado no que foi DETECTADO."""
    instructions = []
    
    instructions.append(f"📈 **GRÁFICO {idx}**: {description}")
    instructions.append("")
    
    chart_type = structure.get("chart_type")
    if chart_type:
        instructions.append(f"📊 **TIPO DETECTADO**: {chart_type}")
        
        # Instruções específicas por tipo
        if chart_type == "ternary":
            instructions.append("   → Extraia pontos (x, y, z) de cada série")
            instructions.append("   → Identifique os 3 eixos e seus rótulos")
        elif chart_type in ("bar", "line", "scatter"):
            instructions.append("   → Extraia valores do eixo X e Y")
            if structure.get("has_multiple_series"):
                instructions.append("   → Gráfico tem MÚLTIPLAS séries - extraia cada uma separadamente")
        elif chart_type == "pie":
            instructions.append("   → Extraia labels + percentuais/valores")
    
    # Eixos
    axis_types = structure.get("axis_types")
    if axis_types:
        instructions.append(f"📏 **EIXOS**: {axis_types}")
    
    # Pontos visíveis
    if structure.get("data_points_visible"):
        instructions.append("✓ **PONTOS VISÍVEIS**: Extraia valores exatos dos pontos marcados")
    
    return "\n".join(instructions)


def _generate_legacy_prompt(characteristics: dict, base_prompt: str) -> str:
    """
    Fallback: Gera prompt no formato antigo se 'elements' não estiver disponível.
    """
    table_type = characteristics.get("table_structure") or characteristics.get("table_type")
    has_colors = characteristics.get("has_colors", False)
    diagonal_empty = characteristics.get("diagonal_empty") or characteristics.get("has_diagonal")
    has_legend = characteristics.get("has_legend", False)
    cell_content_type = characteristics.get("cell_content_type") or characteristics.get("cell_content")
    
    custom_instructions = []
    
    if table_type == "compatibility_matrix":
        custom_instructions.append(
            "📊 **ESTRUTURA**: Matriz de compatibilidade (simétrica)\n"
            "   - Headers das linhas = headers das colunas"
        )
    
    if diagonal_empty:
        custom_instructions.append(
            "⚠️  **DIAGONAL**: Células vazias detectadas na diagonal\n"
            "   → Deixe `<td></td>` vazio"
        )
    
    if has_colors:
        if has_legend:
            custom_instructions.append(
                "🎨 **CORES + LEGENDA**: Use legenda para contexto\n"
                "   → Transcreva o TEXTO escrito, não a cor"
            )
        else:
            custom_instructions.append(
                "🎨 **CORES**: Células coloridas detectadas\n"
                "   → Leia texto/símbolos escritos"
            )
    
    if cell_content_type == "symbols":
        custom_instructions.append(
            "📝 **CONTEÚDO**: Símbolos/letras pequenos\n"
            "   → Amplie zoom mental, trabalhe com calma"
        )
    
    if custom_instructions:
        custom_section = "\n\n" + "="*60 + "\n" + "\n\n".join(custom_instructions) + "\n\n" + "="*60 + "\n\n"
        return custom_section + base_prompt
    
    return base_prompt


def _page_level_precheck(
    image_path: Path,
    config: ImageProcessingConfig,
) -> tuple[bool, str, int, int, dict]:
    """
    PRE-CHECK: Usa LLM barata para identificar:
    - has_content: tem tabela/gráfico?
    - content_type: 'table', 'chart', 'text_only', 'none'
    - content_count: quantas tabelas/gráficos?
    - rotation: rotação detectada (0, 90, 180, 270)
    - characteristics: dict com tipo, complexidade, características especiais
    """
    if not (config.use_cheap_precheck and config.cheap_model):
        logger.info(
            "🟡 Pre-check desativado (modelo barato indisponível); seguindo com página inteira."
        )
        return True, "unknown", 1, 0, {}

    cheap_provider = config.cheap_provider or config.provider
    logger.info(
        "🔍 Rodando pre-check rápido (%s via %s) para %s",
        config.cheap_model,
        cheap_provider or "default",
        image_path.name,
    )
    try:
        has_content, content_type, content_count, rotation, characteristics = quick_precheck_with_cheap_llm(
            image_path,  # ← Volta a usar imagem COLORIDA no pre-check
            config.cheap_model,
            cheap_provider,
            config.openrouter_api_key,
            api_key=config.cheap_api_key,
            azure_endpoint=config.cheap_azure_endpoint,
            azure_api_version=config.cheap_azure_api_version,  # ← FIX: nome correto do atributo
        )
        return has_content, content_type, content_count, rotation, characteristics
    except Exception as exc:
        logger.warning(
            "Pre-check falhou para página %s (%s); assumindo conteúdo",
            image_path,
            exc,
        )
        return True, "unknown", 1, 0, {}


def _extract_page_notes(
    image_path: Path,
    config: ImageProcessingConfig,
) -> list[dict[str, str]]:
    if not (config.use_cheap_precheck and config.cheap_model):
        return []

    try:
        payload = call_openai_vision_json(
            image_path,
            model=config.cheap_model,
            provider=config.cheap_provider or config.provider,
            api_key=config.cheap_api_key,
            azure_endpoint=config.cheap_azure_endpoint,
            azure_api_version=config.cheap_azure_api_version,
            openrouter_api_key=config.openrouter_api_key,
            instructions=NOTES_PROMPT,
            max_retries=0,
        )
        if not isinstance(payload, dict):
            return []
        notes = payload.get("notes")
        if isinstance(notes, list):
            cleaned: list[dict[str, str]] = []
            for note in notes:
                if not isinstance(note, dict):
                    continue
                text = str(note.get("text") or "").strip()
                if not text:
                    continue
                cleaned.append(
                    {
                        "label": str(note.get("label") or "").strip() or "Nota",
                        "text": text,
                        "applies_to": str(note.get("applies_to") or "").strip(),
                    }
                )
            return cleaned
        return []
    except Exception as exc:
        logger.warning("Extração de notas falhou (%s)", exc)
        return []


# =============================================================================
# CHAMADAS DE OCR (agora delegadas para ocr_segmentation.py)
# =============================================================================

def _segment_page_elements(
    page_image_path: Path,
    page_out: Path,
    config: ImageProcessingConfig,
    content_type: str,
    expected_count: int,
) -> List[SegmentedElement]:
    """Wrapper: Delega segmentação para ocr_segmentation.py"""
    if not config.use_layout_ocr:
        return []
    
    return segment_page_elements(
        page_image_path,
        page_out,
        config.ocr_lang,
        config.segment_padding,
        config.max_segments,
        content_type,
        expected_count,
    )


def _run_segmented_flow(
    segments: List[SegmentedElement],
    page_out: Path,
    page_id: str,
    config: ImageProcessingConfig,
) -> Optional[Dict[str, Any]]:
    """Wrapper: Delega processamento de segmentos para ocr_segmentation.py"""
    return run_segmented_flow(
        segments,
        page_out,
        page_id,
        config.model,
        config.provider,
        config.api_key,
        config.azure_endpoint,
        config.azure_api_version,
        config.openrouter_api_key,
        config.locale,
    )


def _write_segments_manifest(page_out: Path, segments: List[SegmentedElement]) -> None:
    """Wrapper: Delega para ocr_segmentation.py"""
    write_segments_manifest(page_out, segments)


# =============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO DE IMAGEM
# =============================================================================

def _convert_to_grayscale(image_path: Path) -> Path:
    """
    Converte imagem para escala de cinza (preto e branco) para melhor contraste.
    Salva no mesmo diretório com sufixo -bw.png
    
    Args:
        image_path: Caminho da imagem original
    
    Returns:
        Caminho da imagem em preto e branco
    """
    bw_path = image_path.parent / f"{image_path.stem}-bw.png"
    
    # Se já existe, retorna (checkpoint)
    if bw_path.exists() and bw_path.stat().st_size > 0:
        logger.debug("✅ Imagem P&B já existe: %s", bw_path.name)
        return bw_path
    
    try:
        # Carrega e converte para escala de cinza
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Falha ao carregar imagem para conversão P&B, usando original")
            return image_path
        
        # Converte para grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplica threshold adaptativo para melhor contraste em tabelas
        # Isso ajuda a tornar texto e bordas mais nítidos
        gray = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # Salva imagem P&B
        cv2.imwrite(str(bw_path), gray)
        logger.info("📸 Imagem convertida para P&B: %s", bw_path.name)
        return bw_path
        
    except Exception as e:
        logger.warning("Erro ao converter para P&B: %s. Usando imagem original.", e)
        return image_path


# =============================================================================
# EXTRAÇÃO PRINCIPAL
# =============================================================================

def _extract_tables_from_payload(payload: dict) -> List[dict]:
    """
    Extrai lista de tabelas do payload, independente do formato.
    
    Returns:
        Lista de dicionários representando cada tabela
    """
    if not payload:
        return []
    
    payload_type = payload.get("type")
    
    if payload_type == "table_set":
        return payload.get("tables", [])
    elif payload_type == "table":
        return [payload]
    elif payload_type == "chart":
        return [payload]
    
    return []


def _call_full_page_llm_with_retry(
    page_image_path: Path,
    page_id: str,
    config: ImageProcessingConfig,
    content_type: str,
    content_count: int,
    characteristics: dict = None,
) -> Optional[Dict[str, Any]]:
    """
    Versão com retry ultra-agressivo quando detecta mesclagem incorreta de tabelas.
    Usa prompt extremamente específico e direto.
    """
    logger.warning("🔥 RETRY COM PROMPT ULTRA-AGRESSIVO")
    
    # Converte para P&B
    bw_image_path = _convert_to_grayscale(page_image_path)
    
    # Prompt SIMPLES e DIRETO
    ultra_prompt = f"""VOCÊ ERROU. Criou 1 objeto quando tem {content_count} tabelas.

CORRIJA AGORA:

{{
  "type": "table_set",
  "tables": [
    {{
      "title": "Tabela Superior",
      "format": "html",
      "html": "<table><thead>...</thead><tbody>...</tbody></table>"
    }},
    {{
      "title": "Tabela Inferior",
      "format": "html",
      "html": "<table><thead>...</thead><tbody>...</tbody></table>"
    }}
  ]
}}

{content_count} tabelas na imagem = {content_count} objetos no JSON.

Tabela 1 = bloco superior (primeiras linhas de dados).
Tabela 2 = bloco inferior (últimas linhas de dados).

SEPARE as tabelas. NÃO junte em 1 objeto."""
    
    try:
        return call_openai_vision_json(
            bw_image_path,
            model=config.model,
            provider=config.provider,
            api_key=config.api_key,
            azure_endpoint=config.azure_endpoint,
            azure_api_version=config.azure_api_version,
            openrouter_api_key=config.openrouter_api_key,
            locale=config.locale,
            instructions=ultra_prompt,
            max_retries=1,  # Só 1 retry aqui
        )
    except Exception as e:
        logger.error("Erro no retry: %s", e)
        return None


def _call_full_page_llm(
    page_image_path: Path,
    page_id: str,
    config: ImageProcessingConfig,
    content_type: str,
    content_count: int,
    characteristics: dict = None,
) -> Optional[Dict[str, Any]]:
    logger.info(
        "📄 Página %s: enviando imagem inteira ao GPT-5 (esperado %d %s)",
        page_id,
        content_count,
        "elemento(s)",
    )
    
    # 🔧 SOLUÇÃO 1: Converte para preto e branco para melhor contraste
    bw_image_path = _convert_to_grayscale(page_image_path)

    if content_type == "chart":
        base_prompt = CHART_PROMPT
    elif content_type == "mixed":
        base_prompt = f"""Esta página contém TABELAS E GRÁFICOS ({content_count} elementos no total).

**EXTRAIA TODOS OS ELEMENTOS SEPARADAMENTE:**

Para TABELAS:
- Use HTML `<table>` com colspan/rowspan
- Formato: {{"title": "...", "format": "html", "html": "<table>...</table>", "notes": "..."}}

Para GRÁFICOS:
- Extraia dados numéricos ou equações
- Formato: {{"title": "...", "type": "chart", "chart": {{...}}}}

**Formato obrigatório:**
{{"type": "table_set", "tables": [...]}}

Retorne TODAS as {content_count} elementos como entradas separadas no array "tables"."""
    else:
        count_desc = _format_count_description("table", content_count or 1)
        base_prompt = PAGE_TABLE_PROMPT.format(count_desc=count_desc)
        
        # 🔧 SOLUÇÃO 2: Prompt SIMPLES e DIRETO quando há múltiplas tabelas
        if content_count and content_count > 1:
            base_prompt = f"""Você vê {content_count} TABELAS SEPARADAS nesta imagem.

CRIE {content_count} OBJETOS SEPARADOS:

{{
  "type": "table_set",
  "tables": [
    {{
      "title": "Tabela 1",
      "format": "html",
      "html": "<table><thead><tr><th>Col1</th><th>Col2</th></tr></thead><tbody><tr><td>val1</td><td>val2</td></tr></tbody></table>"
    }},
    {{
      "title": "Tabela 2",
      "format": "html",
      "html": "<table><thead><tr><th>ColA</th><th>ColB</th></tr></thead><tbody><tr><td>valA</td><td>valB</td></tr></tbody></table>"
    }}
  ]
}}

CADA objeto = 1 tabela completa da imagem.
Tabela 1 = bloco superior.
Tabela 2 = bloco inferior.

NÃO junte em 1 objeto.
{content_count} tabelas = {content_count} objetos."""
    
    # PROMPT PERSONALIZADO: Gera instruções específicas baseadas nas características
    if characteristics:
        prompt = _generate_custom_prompt(characteristics, base_prompt)
        logger.info("🎯 Prompt personalizado gerado baseado em: type=%s, complexity=%s", 
                   characteristics.get("table_type"), 
                   characteristics.get("complexity"))
    else:
        prompt = base_prompt

    # 🔧 USA IMAGEM EM PRETO E BRANCO para melhor contraste
    return call_openai_vision_json(
        bw_image_path,  # ← Usa versão P&B em vez da colorida
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
    characteristics: dict = None,
) -> tuple[List[Path], List[Dict[str, str]]]:
    """
    Extração via fluxo segmentado (OCR + LLM por recorte) com fallback para página inteira.
    Notas são lidas dos arquivos table-XX-notes.txt conforme necessário.
    Gera prompt personalizado baseado nas características detectadas no pre-check.
    
    Se config.use_traditional_ocr=True, usa OpenCV + Tesseract ao invés de LLM Vision.
    """
    outputs: List[Path] = []
    summaries: List[Dict[str, str]] = []

    needs_review = content_count > 2
    expected_elements = max(1, content_count)

    payload: Optional[Dict[str, Any]] = None
    segments: List[SegmentedElement] = []
    
    # Verifica se deve pular OCR nesta página específica
    page_num = int(page_id)
    skip_ocr_this_page = page_num in config.skip_ocr_pages
    use_ocr = config.use_layout_ocr and not skip_ocr_this_page
    
    if skip_ocr_this_page:
        logger.warning("⚠️  Página %s na lista SKIP_OCR_PAGES → processando sem segmentação.", page_id)

    if use_ocr:
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
            _write_segments_manifest(page_out, segments)
            logger.info("📐 Fluxo segmentado: %d recorte(s) identificado(s)", len(segments))
            payload = _run_segmented_flow(segments, page_out, page_id, config)
            if not payload:
                logger.warning(
                    "Fluxo segmentado não retornou dados utilizáveis na página %s.",
                    page_id,
                )
        else:
            logger.info("PPStructure não retornou recortes úteis; partindo para fallback.")
    else:
        logger.debug("Segmentação PaddleOCR desativada; usando página inteira.")

    if payload is None:
        if use_ocr:
            logger.info(
                "🚨 Segmentação indisponível/sem dados para página %s (segments=%s).",
                page_id,
                len(segments) if segments else 0,
            )
        else:
            logger.info("⚡ Fluxo simplificado ativo para página %s.", page_id)
        if not config.fallback_to_full_page:
            logger.error(
                "Fallback desabilitado - abortando processamento da página %s.",
                page_id,
            )
            return outputs, summaries
        logger.info(
            "🔁 Executando extração com página inteira para a página %s.",
            page_id,
        )
        payload = _call_full_page_llm(
            page_image_path,
            page_id,
            config,
            content_type,
            expected_elements,
            characteristics,  # ← Passa características para prompt personalizado
        )

    if not payload:
        logger.warning("GPT-5 não retornou dados para página %s", page_id)
        return outputs, summaries
    
    # 🔧 SOLUÇÃO 3: Validação pós-extração - detecta tabelas incorretamente mescladas
    if expected_elements > 1 and content_type == "table":
        extracted_count = len(_extract_tables_from_payload(payload))
        
        # Se esperava N tabelas mas gerou apenas 1, pode ter mesclado incorretamente
        if extracted_count == 1:
            single_table = _extract_tables_from_payload(payload)[0] if _extract_tables_from_payload(payload) else None
            if single_table:
                html_content = single_table.get("html", "")
                
                # 🚨 VALIDAÇÃO CRÍTICA: Detecta <th> no meio do <tbody> (sinal de tabelas mescladas)
                has_th_in_tbody = "<tbody>" in html_content and html_content.find("<th", html_content.find("<tbody>")) > 0
                
                # Conta células vazias
                empty_cells = html_content.count("<td></td>")
                total_cells = html_content.count("<td")
                empty_ratio = (empty_cells / total_cells) if total_cells > 0 else 0
                
                # Detecta problemas
                has_too_many_empty = empty_ratio > 0.3
                
                if has_th_in_tbody or has_too_many_empty:
                    logger.error(
                        "❌ ERRO GRAVE DETECTADO na página %s:",
                        page_id
                    )
                    if has_th_in_tbody:
                        logger.error("   → Headers (<th>) no meio do tbody (tabelas mescladas!)")
                    if has_too_many_empty:
                        logger.error(
                            "   → %.1f%% células vazias (esperava %d tabelas, gerou 1)",
                            empty_ratio * 100,
                            expected_elements
                        )
                    
                    logger.warning("🔄 Tentando novamente com prompt ULTRA-AGRESSIVO...")
                    
                    # Tenta novamente com prompt ultra-agressivo
                    retry_payload = _call_full_page_llm_with_retry(
                        page_image_path,
                        page_id,
                        config,
                        content_type,
                        expected_elements,
                        characteristics,
                    )
                    
                    if retry_payload:
                        retry_count = len(_extract_tables_from_payload(retry_payload))
                        if retry_count == expected_elements:
                            # Valida que as tabelas estão realmente separadas
                            retry_tables = _extract_tables_from_payload(retry_payload)
                            all_valid = True
                            for idx, tbl in enumerate(retry_tables):
                                retry_html = tbl.get("html", "")
                                if "<tbody>" in retry_html and retry_html.find("<th", retry_html.find("<tbody>")) > 0:
                                    logger.error("❌ Retry também tem <th> no tbody da tabela %d", idx+1)
                                    all_valid = False
                            
                            if all_valid:
                                logger.info("✅ Retry bem-sucedido! Extraídas %d tabelas VÁLIDAS", retry_count)
                                payload = retry_payload
                            else:
                                logger.error("❌ Retry falhou validação. Mantendo original (com erro).")
                        else:
                            logger.warning("⚠️ Retry gerou %d tabelas (esperava %d)", retry_count, expected_elements)
        
        elif extracted_count != expected_elements:
            logger.warning(
                "⚠️ Divergência: esperava %d tabelas, extraiu %d",
                expected_elements,
                extracted_count
            )

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

    tables = _extract_tables_from_payload(payload)
    if not tables:
        # Pode ser um gráfico puro
        if payload.get("type") == "chart":
            chart_data = payload.get("chart")
            base_name = "chart-01"
            
            # Valida dados do gráfico antes de processar
            is_valid, error_msg = _validate_chart_data(chart_data, base_name)
            if not is_valid:
                logger.error("%s", error_msg)
                # Salva JSON mesmo com erro para análise
                (page_out / f"{base_name}-ERRO.json").write_text(_json_dumps(payload), encoding="utf-8")
            
            rows = _chart_payload_to_rows(chart_data)
            if rows:
                # Lê notas do arquivo ou usa do LLM como fallback
                notes_text = _read_notes_for_table(page_out, base_name) or payload.get("notes")
                html = _save_table_outputs(rows, page_out, base_name, notes=notes_text)
                outputs.append(page_out / f"{base_name}.xlsx")
                if html:
                    summaries.append({"page": page_id, "table": base_name, "html": html})
            else:
                logger.warning("Gráfico sem linhas tabulares em página %s", page_id)
            return outputs, summaries

        rows = to_table_from_llm_payload(payload)
        if not rows:
            logger.warning("Nenhuma tabela interpretável em página %s", page_id)
            return outputs, summaries
        # Lê notas do arquivo ou usa do LLM como fallback
        notes_text = _read_notes_for_table(page_out, "table-01") or payload.get("notes")
        html = _save_table_outputs(rows, page_out, "table-01", notes=notes_text)
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
            
            # Valida dados do gráfico antes de processar
            chart_data = info.get("chart")
            is_valid, error_msg = _validate_chart_data(chart_data, chart_base)
            if not is_valid:
                logger.error("%s", error_msg)
                # Salva JSON com marcador de erro
                (page_out / f"{chart_base}-ERRO.json").write_text(_json_dumps(chart_payload), encoding="utf-8")
            
            # Gráficos não geram Excel, apenas JSON
            rows = _chart_payload_to_rows(chart_data)
            if rows:
                # Lê notas do arquivo ou usa do LLM como fallback
                notes_text = _read_notes_for_table(page_out, chart_base) or info.get("notes")
                html = _save_table_outputs(rows, page_out, chart_base, notes=notes_text)
                outputs.append(page_out / f"{chart_base}.xlsx")
                if html:
                    summaries.append({"page": page_id, "table": chart_base, "html": html})
                    logger.info("✅ Gráfico convertido em tabela (%s)", chart_base)
            else:
                logger.info("✅ Gráfico salvo como %s.json (sem conversão tabular)", chart_base)
            continue
        
        table_counter += 1
        base_name = f"table-{table_counter:02d}"

        # NOVO: Detecta se é formato HTML
        if info.get("format") == "html" and info.get("html"):
            logger.info("✅ Tabela %d em formato HTML (estrutura complexa preservada)", table_counter)
            
            # Lê notas do arquivo ou usa do LLM como fallback
            notes_text = _read_notes_for_table(page_out, base_name) or info.get("notes")
            if notes_text:
                logger.info("📝 Notas: %d caracteres", len(notes_text))
            html = _save_html_table(
                html_content=info["html"],
                out_dir=page_out,
                base_name=base_name,
                title=info.get("title"),
                notes=notes_text,
            )
            
            # Tenta encontrar Excel gerado (conversão automática em _save_html_table)
            excel_path = page_out / f"{base_name}.xlsx"
            if excel_path.exists():
                outputs.append(excel_path)
            
            if html:
                has_notes_in_html = '<div class="notes">' in html
                logger.info("📄 Adicionando %s ao summary (contém notas: %s, len: %d)", base_name, has_notes_in_html, len(html))
                if not has_notes_in_html and notes_text:
                    logger.error("❌ BUG: notes_text existia (%d chars) mas HTML retornado não tem <div class='notes'>!", len(notes_text))
                summaries.append({"page": page_id, "table": base_name, "html": html})
            
            # Salva JSON individual
            single_payload = {
                "type": "table",
                "format": "html",
                "title": info.get("title"),
                "notes": notes_text,
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
        
        # Lê notas do arquivo ou usa do LLM como fallback
        notes_text = _read_notes_for_table(page_out, base_name) or info.get("notes")
        html = _save_table_outputs(rows, page_out, base_name, notes=notes_text)
        outputs.append(page_out / f"{base_name}.xlsx")
        if html:
            summaries.append({"page": page_id, "table": base_name, "html": html})
        
        # Salva JSON individual
        single_payload = {
            "type": "table",
            "title": info.get("title"),
            "notes": notes_text,
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


def _validate_chart_data(chart: Dict[str, Any], base_name: str) -> tuple[bool, str]:
    """
    Valida se os dados do gráfico estão consistentes.
    Retorna: (is_valid: bool, warning_message: str)
    """
    if not chart or not isinstance(chart, dict):
        return True, ""
    
    # Valida estrutura x/series
    x_data = chart.get("x", {})
    series = chart.get("series", [])
    
    if not x_data or not series:
        return True, ""
    
    x_values = x_data.get("values", [])
    if not x_values:
        return True, ""
    
    expected_len = len(x_values)
    
    # Verifica cada série
    issues = []
    for idx, s in enumerate(series):
        if not isinstance(s, dict):
            continue
        
        serie_name = s.get("name", f"série {idx+1}")
        values = s.get("values", [])
        
        if not values:
            issues.append(f"'{serie_name}' está vazia")
            continue
        
        actual_len = len(values)
        if actual_len != expected_len:
            issues.append(
                f"'{serie_name}' tem {actual_len} valores (esperado: {expected_len})"
            )
        
        # Detecta provável confusão de vírgula decimal
        if actual_len > expected_len * 1.5:  # 50% a mais que o esperado
            issues.append(
                f"'{serie_name}' pode ter valores decimais lidos separadamente "
                f"(ex: '0,4' lido como [0, 4] ao invés de 0.4)"
            )
    
    if issues:
        msg = (
            f"❌ ERRO no gráfico {base_name}: Eixo X tem {expected_len} valores, mas:\n"
            + "\n".join(f"  • {issue}" for issue in issues)
        )
        return False, msg
    
    return True, ""


def _fix_table_column_order(html_content: str, title: str = "") -> str:
    """
    Detecta e corrige tabelas onde a coluna de nomes está à DIREITA.
    
    Problema: Quando a imagem original tem a coluna de nomes à direita,
    mas a LLM a coloca à esquerda no HTML padrão, os dados ficam desalinhados.
    
    Solução: Move a última coluna (nomes) para a primeira posição.
    
    Returns: HTML corrigido
    """
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return html_content
        
        thead = table.find('thead')
        tbody = table.find('tbody')
        
        if not thead or not tbody:
            return html_content
        
        # Pega o header
        header_row = thead.find('tr')
        if not header_row:
            return html_content
        
        headers = header_row.find_all(['th', 'td'])
        
        # Pega todas as linhas do corpo
        body_rows = tbody.find_all('tr')
        
        if not body_rows or len(headers) < 2:
            return html_content
        
        # HEURÍSTICA INVERTIDA: Se a PRIMEIRA coluna do header está VAZIA
        # e as LINHAS DO CORPO têm <th> (nomes) seguidos de <td> (dados),
        # significa que a LLM já reorganizou para o padrão HTML, mas os dados
        # ficaram desalinhados com a imagem original (que tinha nomes à direita)
        
        first_header = headers[0].get_text(strip=True) if headers else ""
        first_is_empty = not first_header
        
        # Verifica se as linhas do corpo começam com <th> (indicando nomes)
        has_row_headers = False
        for row in body_rows[:3]:
            first_cell = row.find(['th', 'td'])
            if first_cell and first_cell.name == 'th':
                has_row_headers = True
                break
        
        # Se primeira coluna vazia + linhas com <th>, provavelmente foi reorganizado
        # MAS está ERRADO se for matriz de compatibilidade (dados desalinhados)
        is_compatibility_matrix = "compatibilidade" in title.lower() if title else False
        
        # Se detectou que precisa inverter DE VOLTA (mover nomes da esquerda para direita)
        if first_is_empty and has_row_headers and is_compatibility_matrix:
            logger.info(f"🔄 Tabela '{title[:40]}': LLM reorganizou incorretamente")
            logger.info(f"   • Header: primeira coluna vazia={first_is_empty}, total headers={len(headers)}")
            logger.info(f"   • Body: tem row headers={has_row_headers}, total linhas={len(body_rows)}")
            
            # Remove a primeira coluna vazia do header e adiciona à direita
            empty_header = headers[0].extract()
            header_row.append(empty_header)  # Adiciona coluna vazia à direita
            logger.info(f"   ✓ Header corrigido: removida coluna vazia da esquerda, adicionada à direita")
            
            # Move PRIMEIRA coluna de cada linha (nomes em <th>) para ÚLTIMA posição
            rows_fixed = 0
            for row in body_rows:
                cells = row.find_all(['th', 'td'])
                if cells:
                    first_cell = cells[0].extract()  # Remove nome da esquerda
                    row.append(first_cell)  # Adiciona à direita
                    rows_fixed += 1
            
            logger.info(f"   ✓ {rows_fixed} linhas corrigidas: nome movido da esquerda para direita")
            
            return str(soup)
        
        return html_content
        
    except Exception as e:
        logger.warning(f"Erro ao tentar corrigir ordem de colunas: {e}")
        return html_content


def _validate_html_table_content(html_content: str) -> tuple[bool, str]:
    """
    Valida se tabela HTML tem células vazias demais (indicando possível falha na extração).
    Retorna: (is_valid: bool, warning_message: str)
    """
    if not html_content or "<table" not in html_content.lower():
        return True, ""
    
    import re
    
    # Conta células vazias vs não-vazias
    all_td = re.findall(r'<td[^>]*>(.*?)</td>', html_content, re.IGNORECASE | re.DOTALL)
    if not all_td:
        return True, ""
    
    empty_count = sum(1 for cell in all_td if not cell.strip())
    total_count = len(all_td)
    
    if total_count == 0:
        return True, ""
    
    empty_ratio = empty_count / total_count
    
    # Se >70% das células estão vazias, é suspeito (pode ser matriz de compatibilidade mal extraída)
    if empty_ratio > 0.7 and total_count > 20:  # Só alerta em tabelas grandes
        msg = (
            f"⚠️  ATENÇÃO: {empty_count}/{total_count} células vazias ({empty_ratio*100:.1f}%). "
            f"Se esta é uma matriz colorida (compatibilidade, etc), a extração pode ter falha."
        )
        return False, msg
    
    # Detecta linhas idênticas em matrizes grandes (possível cópia indevida)
    if total_count > 100:  # Só para tabelas muito grandes
        # Extrai linhas da tabela
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_content, re.IGNORECASE | re.DOTALL)
        if len(rows) > 15:
            # Compara linhas consecutivas
            row_contents = []
            for row in rows:
                cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.IGNORECASE | re.DOTALL)
                # Ignora header da linha (primeira célula)
                if len(cells) > 1:
                    row_contents.append(tuple(cells[1:]))  # Só as células de dados
            
            # Conta linhas idênticas consecutivas
            identical_pairs = 0
            for i in range(len(row_contents) - 1):
                if row_contents[i] == row_contents[i + 1]:
                    identical_pairs += 1
            
            # Se >20% das linhas são idênticas às anteriores, é suspeito
            if identical_pairs > len(row_contents) * 0.2:
                msg = (
                    f"⚠️  ATENÇÃO: Detectadas {identical_pairs} linhas idênticas consecutivas "
                    f"em matriz grande. Possível erro: modelo pode ter copiado linhas ao invés de ler cada uma."
                )
                return False, msg
    
    return True, ""


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
    
    # Corrige ordem de colunas se necessário (coluna de nomes à direita → esquerda)
    html_content = _fix_table_column_order(html_content, title or base_name)
    
    # Valida conteúdo antes de salvar
    is_valid, warning = _validate_html_table_content(html_content)
    if not is_valid:
        logger.error("❌ TABELA COM PROBLEMA: %s - %s", base_name, warning)
        logger.error("   💡 SOLUÇÃO: Re-processe esta página SEM segmentação (config.use_layout_ocr=False)")
        logger.error("   💡 OU edite manualmente o arquivo HTML/JSON e re-execute")
    
    # Monta HTML completo
    notes_clean = notes.strip() if isinstance(notes, str) and notes.strip() else None
    title_clean = title.strip() if isinstance(title, str) and title.strip() else None
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f9f9f9; }}
        .container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .title {{ font-size: 1.5em; font-weight: 600; margin-bottom: 15px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: middle; }}
        th {{ background-color: #f5f5f5; color: #333; font-weight: 600; }}
        td[colspan], td[rowspan], th[colspan], th[rowspan] {{ text-align: center; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #fafafa; }}
        .notes {{ margin-top: 20px; padding: 10px; background-color: #fff8e1; border-left: 4px solid #ffc107; font-size: 14px; color: #555; }}
        .notes strong {{ color: #333; }}
    </style>
</head>
<body>
<div class="container">
"""
    
    if title_clean:
        full_html += f'    <div class="title">{escape(title_clean)}</div>\n'
    
    full_html += f'    {html_content}\n'
    
    if notes_clean:
        full_html += f'    <div class="notes"><strong>Notas:</strong> {escape(notes_clean)}</div>\n'
    
    full_html += """</div>
</body>
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
    
    # Retorna HTML inline para sumário (COM notas DEPOIS)
    html_with_notes = html_content
    if notes_clean:
        html_with_notes += f'\n<div class="notes"><strong>Notas:</strong> {escape(notes_clean)}</div>'
        logger.debug("HTML retornado COM notas (%d chars de notas)", len(notes_clean))
    else:
        logger.debug("HTML retornado SEM notas (notes_clean está vazio)")
    return html_with_notes


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
        
        # Adiciona notas DEPOIS da tabela
        if notes_clean:
            html += f'\n<div class="notes"><strong>Notas:</strong> {escape(notes_clean)}</div>'
        
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
                # Filtra entradas antigas de "page-notes" (legado - notas agora vão junto com cada tabela)
                if table == "page-notes":
                    logger.debug("Ignorando entrada legacy 'page-notes' da página %s", page)
                    continue
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
    for entry in sorted(all_entries.values(), key=_summary_sort_key):
        has_notes = '<div class="notes">' in entry['html']
        logger.debug("Página %s - %s: contém notas no HTML? %s", entry['page'], entry['table'], has_notes)
        rows.append(
            f"<section class='table-block'><h3>Página {entry['page']} - {entry['table']}</h3>{entry['html']}</section>"
        )
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    template = (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body{{font-family:Arial,sans-serif;padding:20px;background:#f9f9f9;}}"
        "section.table-block{{background:#fff;border:1px solid #ddd;margin-bottom:20px;padding:15px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}}"
        "section.table-block h3{{margin-top:0;font-size:16px;color:#666;font-weight:600;}}"
        "table{{border-collapse:collapse;width:100%;margin-top:10px;}}table,th,td{{border:1px solid #ccc;}}"
        "th,td{{padding:8px;font-size:13px;text-align:left;vertical-align:middle;}}"
        "th{{background:#f5f5f5;font-weight:600;color:#333;}}"
        "td[colspan],td[rowspan],th[colspan],th[rowspan]{{text-align:center;font-weight:600;}}"
        "tr:nth-child(even){{background-color:#fafafa;}}"
        ".notes{{margin-top:15px;padding:10px;background-color:#fff8e1;border-left:4px solid #ffc107;font-size:12px;color:#555;}}"
        ".notes strong{{color:#333;}}"
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


def _summary_sort_key(entry: Dict[str, str]) -> tuple:
    page = entry.get("page") or "0"
    try:
        page_num = int(page)
    except ValueError:
        page_num = 0

    table_label = entry.get("table") or ""
    match = re.search(r"(\d+)", table_label)
    table_num = int(match.group(1)) if match else 0
    return (page_num, table_num, table_label)


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


def _read_notes_for_table(page_out: Path, table_name: str) -> Optional[str]:
    """Lê notas do arquivo table-XX-notes.txt se existir"""
    notes_file = page_out / f"{table_name}-notes.txt"
    if notes_file.exists():
        try:
            return notes_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("Erro ao ler %s: %s", notes_file.name, e)
    return None


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


def _chart_payload_to_rows(chart_payload: Optional[Dict[str, Any]]) -> Optional[List[List[str]]]:
    if not chart_payload:
        return None
    payload = {
        "type": "chart",
        "chart": chart_payload,
    }
    rows = to_table_from_llm_payload(payload)
    if not rows:
        return None
    # Substitui "None"/"null" por vazio, mas mantém números/strings
    cleaned: List[List[str]] = []
    for row in rows:
        cleaned.append([("" if cell is None else str(cell)) for cell in row])
    return cleaned


def _json_dumps(payload: dict) -> str:
    """Converte dict para JSON formatado"""
    return json.dumps(payload, ensure_ascii=False, indent=2)
